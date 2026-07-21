"""Batch inference pipeline + serialized worker (DESIGN.md Sec 3.3/6).

Builds the persistent ``Gst.Pipeline "batch"``: appsrc (is-live=true
format=time block=true do-timestamp=false max-bytes=268435456, RGBA WxH caps)
-> nvvideoconvert output-buffers=16 -> NVMM RGBA capsfilter -> nvstreammux
batch-size=1 (Sec 3.3 said 8; see build_batch_pipeline for why a single-pad
mux must emit one frame per batch) batched-push-timeout=100000 attach-sys-ts=false
live-source=false -> nvinfer pgie_batch (generated b8 yolo config) -> valve
v_assess drop=true -> nvinfer sgie_batch (existing injury b8 config,
process-mode=2 output-tensor-meta=true) -> fakesink sync=false async=false.
The pool sizes are load-bearing (empirically validated wedge fix, Sec 3.3) —
implement verbatim.

The worker thread ("batch", Sec 2) is the sole runner and therefore serializes
runs by construction: push k buffers (re-stamped with each item's
feeder-assigned pts) -> wait for k collected frames or 10 s -> publish via
injected callbacks -> next run. Valve toggling only happens between runs
(race-free). Service threads submit a RunRequest and block on it; the worker
dispatches manual requests ahead of continuous auto-runs, which is how manual
work gets priority on the one shared inference pipeline. Continuous auto-runs
when >= continuous.run_size frames pend in ITS deque or the oldest is > 200 ms
(Sec 4) and nothing manual is waiting.

Gst/pyds imports live inside builders/probes; result dataclasses and the
collector's join-by-pts logic are importable without them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from config import PipelineConfig
from frames import BatchItem, GrabState

_LOG = logging.getLogger("ds_ros_pipeline.batch_pipeline")

# Sec 3.3, verbatim (empirically validated — the 200 KB appsrc default blocks
# the worker after one 14.7 MB frame; the 4-buffer convert pool wedges the mux).
APPSRC_MAX_BYTES = 268_435_456
CONV_OUTPUT_BUFFERS = 16
MUX_BATCH_PUSH_TIMEOUT_US = 100_000
COLLECT_TIMEOUT_S = 10.0

# Sec 4 continuous mode: auto-run when the oldest pending item is > 200 ms.
CONTINUOUS_MAX_WAIT_S = 0.2
_POLL_INTERVAL_S = 0.05


@dataclass(frozen=True)
class Detection:
    """One pgie object meta, in source pixel coordinates."""

    left: float
    top: float
    width: float
    height: float
    confidence: float
    class_id: int
    label: str
    object_id: int


def indexed_detections(detections: "tuple[Detection, ...] | list[Detection]",
                       ) -> tuple[list[Detection], str | None]:
    """``detections`` in DeepStream index order, plus a complaint or None.

    The publishers lay TargetBoxes down in this order so that position i of
    ``uav_target_boxes`` is the detection whose ``object_id`` is i — the same
    index ``assessments`` and ``detection_id`` key on (see the note in
    install_collect_probes). det_collect already emits them in that order, so
    the sort is a no-op today; it is here so the guarantee is enforced where
    the array is built rather than inherited from a probe two modules away.

    The second element is a human-readable warning when the indices are not
    contiguous 0..n-1 — the join still follows object_id and stays correct,
    but array position no longer equals detection_id, which every consumer
    indexing the array by detection_id would need to know. Returned rather
    than logged so this stays importable without rclpy.
    """
    ordered = sorted(detections, key=lambda det: det.object_id)
    indices = [det.object_id for det in ordered]
    if indices != list(range(len(ordered))):
        return ordered, (
            f"detection indices are not contiguous 0..n-1: {indices}; "
            "annotations still follow object_id, but array position no "
            "longer equals detection_id")
    return ordered, None


@dataclass
class RunRequest:
    """One queued intent to run the batch pipeline, submitted by a service
    thread and executed by the "batch" worker.

    Deliberately an INTENT, not a batch of frames: ``source="manual"`` carries
    no items and swaps the manual deque at the moment it is dispatched, so a
    request that waits its turn picks up everything enqueued meanwhile,
    ``pending_depth()`` never reports frames that are queued-but-invisible,
    and shutdown drains no 14.75 MB arrays. ``source="item"`` (capture/vlm)
    is the exception — it already holds its one frame and bypasses both
    deques.
    """

    source: str                     # "manual" | "item"
    assess: bool
    publish: str | None = None      # None => chosen from assess
    item: "BatchItem | None" = None
    timeout: float = COLLECT_TIMEOUT_S
    done: threading.Event = field(default_factory=threading.Event)
    result: tuple[bool, str] | None = None

    def resolve(self, success: bool, message: str) -> None:
        self.result = (success, message)
        self.done.set()


@dataclass(frozen=True)
class FrameResult:
    """Everything collected for one batched frame, joined back to its item.

    ``assessments``: object_id -> the 8-head dict from
    ``deepstream_yolo.assessment_runtime.parse_assessment_tensor_meta``;
    empty for detect-only runs (valve closed).
    """

    item: BatchItem
    detections: tuple[Detection, ...]
    assessments: dict[int, dict] = field(default_factory=dict)


class ResultCollector:
    """Probe-fed store keying per-frame results by frame_meta.buf_pts (Sec 6).

    buf_pts == the feeder-assigned live pts carried through appsrc re-stamping
    — globally unique, so joins cannot collide across loop wraps (Sec 5).
    det_collect (pgie src pad) and assess_collect (sgie src pad) probes write
    into it from the batch pipeline's streaming threads; the worker thread
    waits and reads. Internal mutex + condition.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._expected: set[int] = set()
        self._assess = False
        self._detections: dict[int, tuple[Detection, ...]] = {}
        self._assessments: dict[int, dict[int, dict]] = {}
        self._stale = 0

    def start_run(self, expected_pts: list[int], assess: bool) -> None:
        """Reset state for a run expecting exactly these pts keys (worker)."""
        with self._cond:
            self._expected = set(expected_pts)
            self._assess = assess
            self._detections = {}
            self._assessments = {}

    def add_detections(self, buf_pts: int, detections: tuple[Detection, ...]) -> None:
        """det_collect probe entry point (streaming thread). Non-blocking."""
        with self._cond:
            if buf_pts not in self._expected:
                # Late arrival from a timed-out earlier run; drop, count.
                self._stale += 1
                return
            self._detections[buf_pts] = detections
            if self._complete_locked():
                self._cond.notify_all()

    def add_assessments(self, buf_pts: int, assessments: dict[int, dict]) -> None:
        """assess_collect probe entry point (streaming thread). Non-blocking."""
        with self._cond:
            if buf_pts not in self._expected:
                self._stale += 1
                return
            self._assessments[buf_pts] = assessments
            if self._complete_locked():
                self._cond.notify_all()

    def wait_complete(self, timeout: float) -> bool:
        """Block the worker until every expected frame is fully collected
        (detections, plus assessments when the run opened the valve) or
        ``timeout`` (10 s, Sec 3.3). Returns True iff complete."""
        with self._cond:
            return self._cond.wait_for(self._complete_locked, timeout)

    def results(self, items: list[BatchItem]) -> list[FrameResult]:
        """Join collected results to ``items`` by pts, in item order (worker).

        Items whose frame never traversed the pgie (collector timeout) are
        omitted rather than reported as empty detections — nvinfer emits a
        frame_meta for every frame, so absence means loss, not zero objects.
        """
        with self._cond:
            detections = dict(self._detections)
            assessments = dict(self._assessments)
        return [
            FrameResult(
                item=item,
                detections=detections[item.pts],
                assessments=assessments.get(item.pts, {}),
            )
            for item in items
            if item.pts in detections
        ]

    def stale(self) -> int:
        """Results that arrived for a pts no run was expecting (any thread).

        Non-zero means a run timed out and its frames finished traversing
        afterwards, landing while a later run owned the collector. The join is
        keyed on pts so those strays are discarded rather than mis-attributed
        — but without surfacing this counter the fact that it happened at all
        was invisible in the field. Published as /ds/status stale_results.
        """
        with self._cond:
            return self._stale

    def _complete_locked(self) -> bool:
        return all(
            pts in self._detections
            and (not self._assess or pts in self._assessments)
            for pts in self._expected
        )


@dataclass
class BatchParts:
    """Handles for wiring: the pipeline, appsrc, valve, and probe pads."""

    pipeline: "object"
    src_batch: "object"    # appsrc
    v_assess: "object"     # valve; drop=true except *_assess runs
    pgie: "object"
    sgie: "object"


def _make(factory: str, name: str):
    from gi.repository import Gst

    elem = Gst.ElementFactory.make(factory, name)
    if elem is None:
        raise RuntimeError(f"Missing GStreamer element: {factory}")
    return elem


def _link(upstream, downstream) -> None:
    if not upstream.link(downstream):
        raise RuntimeError(
            f"Failed to link {upstream.get_name()} -> {downstream.get_name()}"
        )


def build_batch_pipeline(config: PipelineConfig, width: int, height: int,
                         pgie_config: str, sgie_config: str) -> BatchParts:
    """Assemble the batch graph with Sec 3.3's exact properties.

    Called once from the main thread at startup (after infer_configs wrote
    the pgie config). Does not set state, does not install probes.
    """
    from gi.repository import Gst

    pipeline = Gst.Pipeline.new("batch")

    src_batch = _make("appsrc", "src_batch")
    src_batch.set_property("is-live", True)
    src_batch.set_property("format", Gst.Format.TIME)
    src_batch.set_property("block", True)
    src_batch.set_property("do-timestamp", False)
    src_batch.set_property("max-bytes", APPSRC_MAX_BYTES)
    src_batch.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw,format=RGBA,width={width},height={height},framerate=0/1"
        ),
    )

    conv_batch = _make("nvvideoconvert", "conv_batch")
    conv_batch.set_property("output-buffers", CONV_OUTPUT_BUFFERS)

    caps_batch = _make("capsfilter", "caps_batch")
    caps_batch.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
    )

    mux_batch = _make("nvstreammux", "mux_batch")
    # batch-size=1, NOT config.batch_engine_batch. Every frame here enters on
    # the one sink_0 pad, so the legacy nvstreammux stamps them all
    # source_id=0; when it packs several into one batch, nvinfer annotates
    # only batch_id=0 and every later frame comes out with zero objects
    # (measured: a 6-frame run published boxes on frame 0 and empty arrays on
    # frames 1-5, even when all six buffers were byte-identical copies).
    # batch-size=1 makes the mux emit one frame per batch, which is the case
    # nvinfer handles correctly for a single source. The pgie config stays
    # b8 -- the engine accepts any batch in [1, 8], and the run still streams
    # back-to-back, it just no longer infers 8-up. Recovering true batched
    # inference needs one mux sink pad per frame (see README).
    mux_batch.set_property("batch-size", 1)
    mux_batch.set_property("width", width)
    mux_batch.set_property("height", height)
    mux_batch.set_property("batched-push-timeout", MUX_BATCH_PUSH_TIMEOUT_US)
    mux_batch.set_property("attach-sys-ts", False)
    mux_batch.set_property("live-source", False)

    pgie = _make("nvinfer", "pgie_batch")
    pgie.set_property("config-file-path", pgie_config)

    v_assess = _make("valve", "v_assess")
    v_assess.set_property("drop", True)

    sgie = _make("nvinfer", "sgie_batch")
    sgie.set_property("config-file-path", sgie_config)
    sgie.set_property("process-mode", 2)
    sgie.set_property("output-tensor-meta", True)

    sink_batch = _make("fakesink", "sink_batch")
    sink_batch.set_property("sync", False)
    sink_batch.set_property("async", False)

    for elem in (src_batch, conv_batch, caps_batch, mux_batch, pgie, v_assess,
                 sgie, sink_batch):
        pipeline.add(elem)

    _link(src_batch, conv_batch)
    _link(conv_batch, caps_batch)
    mux_sink = mux_batch.request_pad_simple("sink_0")
    if mux_sink is None:
        raise RuntimeError("mux_batch refused a sink_0 request pad")
    if caps_batch.get_static_pad("src").link(mux_sink) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link caps_batch to mux_batch")
    _link(mux_batch, pgie)
    _link(pgie, v_assess)
    _link(v_assess, sgie)
    _link(sgie, sink_batch)

    return BatchParts(
        pipeline=pipeline,
        src_batch=src_batch,
        v_assess=v_assess,
        pgie=pgie,
        sgie=sgie,
    )


def install_collect_probes(parts: BatchParts, collector: ResultCollector) -> None:
    """Install det_collect (pgie src pad, BEFORE the valve) and assess_collect
    (sgie src pad, reusing parse_assessment_tensor_meta) probes feeding
    ``collector``. Main thread, during wiring."""
    import pyds
    from gi.repository import Gst

    from deepstream_yolo.assessment_runtime import (
        ASSESSMENT_GIE_ID,
        parse_assessment_tensor_meta,
    )

    # No tracker in this pipeline, so obj_meta.object_id arrives as the
    # untracked sentinel for every object. det_collect (pgie src pad, i.e.
    # upstream of the valve and therefore of the sgie) OVERWRITES it with the
    # object's ordinal position in the frame's obj_meta_list, and that
    # written-down index is the one identifier everything downstream uses:
    #
    #   Detection.object_id            == the index
    #   TargetBoxArray.uav_target_boxes[i].. == the detection whose index is i
    #   FrameResult.assessments[index] == that box's 8 clip_rgb_* heads
    #
    # assess_collect then reads obj.object_id back rather than re-deriving a
    # position by counting the list a second time. Two independent counters
    # would agree only as long as the sgie never reorders, inserts, or drops
    # an object meta; carrying the index in the meta makes the join correct
    # by construction instead of by assumption, and a reorder can no longer
    # silently attach one person's assessment to another person's box.

    def det_collect(_pad, info):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK
        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK
            frame_list = batch_meta.frame_meta_list
            while frame_list:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                detections = []
                obj_list = frame_meta.obj_meta_list
                index = 0
                while obj_list:
                    obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                    rect = obj.rect_params
                    # Stamp the index into the meta so the sgie side can read
                    # back exactly this object's identity (see note above).
                    obj.object_id = index
                    detections.append(Detection(
                        left=float(rect.left),
                        top=float(rect.top),
                        width=float(rect.width),
                        height=float(rect.height),
                        confidence=float(obj.confidence),
                        class_id=int(obj.class_id),
                        label=str(obj.obj_label),
                        object_id=index,
                    ))
                    index += 1
                    obj_list = obj_list.next
                collector.add_detections(int(frame_meta.buf_pts), tuple(detections))
                frame_list = frame_list.next
        except Exception:
            _LOG.exception("det_collect probe failed")
        return Gst.PadProbeReturn.OK

    def assess_collect(_pad, info):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK
        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK
            frame_list = batch_meta.frame_meta_list
            while frame_list:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                assessments: dict[int, dict] = {}
                obj_list = frame_meta.obj_meta_list
                while obj_list:
                    obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                    # The index det_collect stamped on this very object --
                    # not a fresh positional count.
                    index = int(obj.object_id)
                    user_meta_list = obj.obj_user_meta_list
                    while user_meta_list:
                        user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                        if (user_meta.base_meta.meta_type
                                == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
                            tensor_meta = pyds.NvDsInferTensorMeta.cast(
                                user_meta.user_meta_data)
                            if int(tensor_meta.unique_id) == ASSESSMENT_GIE_ID:
                                predictions = parse_assessment_tensor_meta(tensor_meta)
                                if predictions:
                                    assessments[index] = predictions
                        user_meta_list = user_meta_list.next
                    obj_list = obj_list.next
                collector.add_assessments(int(frame_meta.buf_pts), assessments)
                frame_list = frame_list.next
        except Exception:
            _LOG.exception("assess_collect probe failed")
        return Gst.PadProbeReturn.OK

    parts.pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, det_collect)
    parts.sgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, assess_collect)


class BatchWorker:
    """The "batch" thread (Sec 2): the SOLE runner, and owns appsrc pushing.

    Every run goes through this one thread. Service threads do not execute
    runs; they submit a RunRequest and block on its completion event. That
    makes the thread the single place where "what runs next" is decided, and
    manual work is prioritized there with one branch:

        if a manual request is queued -> run it
        else                          -> evaluate the continuous auto-run tick

    Why the arbitration lives here rather than in a priority-aware lock around
    _execute_run: the decision to start a continuous run happens in
    _continuous_tick, ABOVE the run itself, because the tick has to swap the
    continuous deque before it can run anything. A lock further down could
    only observe that decision after the frames had already been claimed —
    it would have to run them anyway or push them back and break pts order.
    Moving all arbitration onto one thread makes "manual first" a property of
    the code's shape instead of an invariant maintained by a counter, and it
    adds no second condition variable for stop() to have to notify.

    Priority is exercised only at run boundaries: nvinfer has no cancellation,
    so an in-flight run always completes. With the mux at batch-size=1 a run
    is k sequential ~35 ms inferences, so a manual request waits at most one
    continuous run (~150 ms at the default run_size=4).

    Constructed with the grab state (source of take_pending/take_continuous),
    the parts, the collector, and publish callbacks injected by ros_io — the
    run's trigger selects which fire per frame result:
      publish_detections(result) -> None   run_detect / continuous run:
                                      one TargetBoxArray, annotations empty
      publish_assessments(result) -> None  run_detect_assess / continuous:
                                      one TargetBoxArray, annotations filled
                                      on the assessed boxes
      publish_vlm_detections(result) -> None
                                      capture/vlm: one TargetBoxArray on the
                                      /vlm topic, boxes flagged
                                      use_for_assessment
    Callbacks are invoked on the worker thread; rclpy publishers are
    thread-safe (Sec 2).
    """

    def __init__(self, config: PipelineConfig, grab: GrabState,
                 parts: BatchParts, collector: ResultCollector,
                 publish_detections: Callable[[FrameResult], None],
                 publish_assessments: Callable[[FrameResult], None],
                 publish_vlm_detections: Callable[[FrameResult], None]) -> None:
        self._config = config
        self._grab = grab
        self._parts = parts
        self._collector = collector
        self._publish_detections = publish_detections
        self._publish_assessments = publish_assessments
        self._publish_vlm_detections = publish_vlm_detections
        # One condition guards everything the worker waits on: the manual
        # request queue, the continuous flags, and the stop flag. There is no
        # separate run lock — runs never overlap because only this thread
        # executes them, which is also what keeps the valve from toggling with
        # buffers in flight.
        self._cond = threading.Condition()
        self._manual: deque[RunRequest] = deque()
        self._stop = False
        self._continuous = False
        self._continuous_assess = False
        self._oldest_since: float | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the worker thread (main thread, after pipeline PLAYING)."""
        self._thread = threading.Thread(
            target=self._worker_loop, name="batch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal and join the worker (main thread, shutdown). Blocks until a
        run in flight finishes or its 10 s collector timeout fires.

        The worker resolves every still-queued request on its way out, so a
        service thread blocked in run_once/run_capture is released with a
        failure rather than waiting out its own timeout."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # start() was never called (unit tests, or a failed bring-up): nobody
        # will drain the queue, so do it here rather than leave a caller
        # blocked on an event that can no longer be set.
        self._drain_requests()

    def run_once(self, assess: bool, timeout: float = 30.0) -> tuple[bool, str]:
        """Submit one manual run and block until it finishes (grp_batch
        service thread; MutuallyExclusive so these serialize with each other
        by construction).

        Queued ahead of the continuous stream's auto-runs, so it waits at most
        for the one run already in flight. The queued request carries no
        frames: the worker calls swap_pending() at dispatch, so the run covers
        everything enqueued up to the moment it actually starts, and it never
        picks up a continuous stride frame.

        Returns (False, "batch queue empty") if nothing was enqueued, or
        (False, ...) if the run did not complete within ``timeout``.
        """
        return self._submit(RunRequest(
            source="manual", assess=assess,
            timeout=min(timeout, COLLECT_TIMEOUT_S)), timeout)

    def run_capture(self, item: BatchItem,
                    timeout: float = COLLECT_TIMEOUT_S) -> tuple[bool, str]:
        """Run detection over ONE captured frame, bypassing both pending
        queues, and publish the vlm TargetBoxArray (the capture/vlm path;
        grp_capture service thread).

        Carries its frame with it — the grab probe already served the capture
        regardless of mode — so this only waits its turn behind at most one
        in-flight run, and is prioritized over the continuous stream like
        every other manual request. Neither pending queue is touched.
        """
        return self._submit(RunRequest(
            source="item", assess=False, publish="vlm", item=item,
            timeout=timeout), timeout)

    def _submit(self, request: RunRequest, timeout: float) -> tuple[bool, str]:
        """Queue a manual request for the worker and wait for its result.

        A timeout here does NOT cancel the request — the worker may still run
        and publish it. That is deliberate: for capture/vlm publishing is the
        whole point, and a half-cancelled run would need to be torn out of the
        collector mid-flight.
        """
        with self._cond:
            if self._stop:
                return False, "batch worker stopped"
            self._manual.append(request)
            self._cond.notify_all()
        if not request.done.wait(timeout):
            return False, f"run did not complete within {timeout:.0f}s"
        return request.result or (False, "run produced no result")

    def _run_limit(self) -> int:
        """Max frames per run — the engine's batch size (batch.engine_batch).

        This bounds two things at once: how much work the sole runner commits
        to before it can re-check for a higher-priority request, and how many
        frames one nvinfer submission is asked to carry. Keeping the two equal
        means a long queue costs more runs, never a longer run.
        """
        return max(1, self._config.batch_engine_batch)

    def collector_stale(self) -> int:
        """ResultCollector.stale() for /ds/status (status timer thread)."""
        return self._collector.stale()

    def _drain_requests(self) -> None:
        """Fail every queued request (worker exit / stop before start)."""
        with self._cond:
            pending, self._manual = list(self._manual), deque()
        for request in pending:
            request.resolve(False, "batch worker stopped")

    def set_continuous(self, active: bool, assess: bool) -> None:
        """Enable/disable auto-runs (grp_fast service thread, non-blocking).

        While active the worker runs whenever grab.continuous_depth() >=
        continuous.run_size or the oldest pending item is > 200 ms old
        (Sec 4) — but only when no manual request is waiting, which is what
        gives manual work priority. Manual run_once and run_capture stay
        available throughout. Same _execute_run machinery either way — no
        second code path (Sec 6).

        ``assess`` only selects the valve position for subsequent auto-runs,
        so flipping it while active neither stops nor restarts the stream.
        """
        with self._cond:
            # The age clock is only reset when the stream itself starts or
            # stops; a mid-stream assess flip leaves it running, so an item
            # already waiting still ages out on schedule.
            if active is not self._continuous:
                self._oldest_since = None
            self._continuous = active
            self._continuous_assess = assess if active else False
            self._cond.notify_all()

    # -- worker internals ---------------------------------------------------

    def _worker_loop(self) -> None:
        """Dispatch loop: manual requests first, continuous otherwise.

        This branch IS the priority mechanism (see the class docstring). Both
        arms run on this thread, so a run never overlaps another and the valve
        is never touched with buffers in flight.
        """
        try:
            while True:
                with self._cond:
                    while (not self._stop and not self._manual
                           and not self._continuous):
                        self._cond.wait()
                    if self._stop:
                        return
                    # Manual work always wins the next slot. Continuous is only
                    # considered when nothing manual is waiting, so a stream at
                    # full tilt can delay a manual run by one run, never more.
                    request = self._manual.popleft() if self._manual else None
                    assess = self._continuous_assess
                try:
                    if request is not None:
                        self._serve(request)
                    else:
                        self._continuous_tick(assess)
                except Exception:
                    _LOG.exception("batch run failed")
                    if request is not None and not request.done.is_set():
                        request.resolve(False, "run raised")
        finally:
            # Release anyone blocked in _submit rather than making them wait
            # out their own timeout (shutdown, or a crash in the loop).
            self._drain_requests()

    def _serve(self, request: RunRequest) -> None:
        """Execute one dispatched manual request and resolve its waiter."""
        if request.source == "item":
            success, message = self._execute_run(
                [request.item], request.assess, request.timeout,
                publish=request.publish)
            request.resolve(success, message)
            return

        # Claimed HERE, not at submit time: the run covers everything enqueued
        # while the request waited, and pending_depth() never reports frames
        # that are queued-but-invisible.
        #
        # Drained a batch at a time rather than in one giant run: with a queue
        # deliberately longer than a batch, a single all-in run would occupy
        # the sole runner for depth x per-frame time, which is exactly the
        # window a manual request is not supposed to have to wait through. One
        # bounded run per iteration keeps that window at one batch, and any
        # remainder is picked up by the next iteration of the loop below.
        frames_run = 0
        runs = 0
        failures: list[str] = []
        while True:
            items = self._grab.take_pending(self._run_limit())
            if not items:
                break
            success, message = self._execute_run(
                items, request.assess, request.timeout,
                publish=request.publish)
            frames_run += len(items)
            runs += 1
            if not success:
                failures.append(message)
                break
        if not frames_run:
            request.resolve(False, "batch queue empty")
            return
        summary = f"{frames_run} frames in {runs} batch{'es' if runs > 1 else ''}"
        if failures:
            request.resolve(False, f"{summary}; {failures[-1]}")
        else:
            request.resolve(True, summary)

    def _continuous_tick(self, assess: bool) -> None:
        """One poll of the Sec 4 auto-run condition; runs when due.

        The 200 ms rule is tracked worker-side: GrabState exposes only depth,
        so the age of the oldest pending item is measured from the poll that
        first observed a non-empty queue (<= 50 ms coarse — see review notes).
        """
        depth = self._grab.continuous_depth()
        now = time.monotonic()
        if depth == 0:
            self._oldest_since = None
        elif self._oldest_since is None:
            self._oldest_since = now
        due = depth >= self._config.continuous_run_size or (
            self._oldest_since is not None
            and now - self._oldest_since >= CONTINUOUS_MAX_WAIT_S)
        if due:
            # One bounded batch per tick. The loop comes straight back here if
            # more is queued, but re-checks for manual work in between — so a
            # deep continuous backlog can never delay a manual request by more
            # than a single batch.
            items = self._grab.take_continuous(self._run_limit())
            self._oldest_since = None
            if items:
                success, message = self._execute_run(
                    items, assess, COLLECT_TIMEOUT_S)
                if not success:
                    _LOG.warning("continuous run incomplete: %s", message)
            return
        with self._cond:
            # Wake early if a manual request arrives, so it is not stuck behind
            # the remainder of this poll interval.
            if not self._stop and self._continuous and not self._manual:
                self._cond.wait(_POLL_INTERVAL_S)

    def _execute_run(self, items: list[BatchItem], assess: bool,
                     wait_timeout: float,
                     publish: str | None = None) -> tuple[bool, str]:
        """Sec 3.3/6 run: valve -> push k -> wait k or timeout -> publish.

        WORKER THREAD ONLY (or a test driving it directly). It carries no lock
        of its own: runs cannot overlap because _worker_loop is the only
        caller, which is what lets the valve be set before the first push and
        restored after collection without ever toggling mid-run.

        ``publish`` selects which callbacks are invoked per frame result —
        "detections" (default for assess=False), "assessments" (default for
        assess=True), or "vlm" (the capture/vlm path: the vlm TargetBoxArray).
        """
        if publish is None:
            publish = "assessments" if assess else "detections"
        self._collector.start_run([item.pts for item in items], assess)
        self._parts.v_assess.set_property("drop", not assess)
        try:
            pushed = self._push_items(items)
            complete = self._collector.wait_complete(wait_timeout)
        finally:
            self._parts.v_assess.set_property("drop", True)
        results = self._collector.results(items)
        boxes = 0
        assessed = 0
        for result in results:
            boxes += len(result.detections)
            if publish == "assessments":
                self._publish_assessments(result)
                assessed += len(result.assessments)
            elif publish == "vlm":
                self._publish_vlm_detections(result)
            else:
                self._publish_detections(result)
        message = f"{len(results)}/{len(items)} frames, {boxes} boxes"
        if assess:
            message += f", {assessed} assessed"
        if not pushed:
            message += " (appsrc push failed)"
        elif not complete:
            message += " (collector timeout)"
        return pushed and complete, message

    def _push_items(self, items: list[BatchItem]) -> bool:
        """Push every item re-stamped with its feeder-assigned pts (Sec 5).

        appsrc block=true + max-bytes=256 MiB: a saturated GPU blocks this
        worker thread only, never the live pipeline (Sec 6 isolation).
        """
        from gi.repository import Gst

        for item in items:
            buf = Gst.Buffer.new_wrapped(item.frame_rgba.tobytes())
            buf.pts = item.pts
            buf.dts = Gst.CLOCK_TIME_NONE
            ret = self._parts.src_batch.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                _LOG.error("appsrc push-buffer returned %s (pts=%d)",
                           ret, item.pts)
                return False
        return True
