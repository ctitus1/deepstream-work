"""Batch inference pipeline + serialized worker (DESIGN.md Sec 3.3/6).

Builds the persistent ``Gst.Pipeline "batch"``: appsrc (is-live=true
format=time block=true do-timestamp=false max-bytes=268435456, RGBA WxH caps)
-> nvvideoconvert output-buffers=16 -> NVMM RGBA capsfilter -> nvstreammux
batch-size=8 batched-push-timeout=100000 attach-sys-ts=false
live-source=false -> nvinfer pgie_batch (generated b8 yolo config) -> valve
v_assess drop=true -> nvinfer sgie_batch (existing injury b8 config,
process-mode=2 output-tensor-meta=true) -> fakesink sync=false async=false.
The pool sizes are load-bearing (empirically validated wedge fix, Sec 3.3) —
implement verbatim.

The worker thread ("batch", Sec 2) serializes runs: push k buffers
(re-stamped with each item's feeder-assigned pts) -> wait for k collected
frames or 10 s -> publish via injected callbacks -> next run. Valve toggling
only happens between runs (race-free). Continuous mode auto-runs when
>= continuous.run_size frames pend or the oldest is > 200 ms (Sec 4).

Gst/pyds imports live inside builders/probes; result dataclasses and the
collector's join-by-pts logic are importable without them.
"""

from __future__ import annotations

import logging
import threading
import time
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
    mux_batch.set_property("batch-size", config.batch_engine_batch)
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

    # No tracker in this pipeline, so obj_meta.object_id is the untracked
    # sentinel for every object. Both probes instead key objects by their
    # ordinal position in the frame's obj_meta_list — the sgie appends tensor
    # meta without reordering the list, so the join is stable.

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
                index = 0
                while obj_list:
                    obj = pyds.NvDsObjectMeta.cast(obj_list.data)
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
                    index += 1
                    obj_list = obj_list.next
                collector.add_assessments(int(frame_meta.buf_pts), assessments)
                frame_list = frame_list.next
        except Exception:
            _LOG.exception("assess_collect probe failed")
        return Gst.PadProbeReturn.OK

    parts.pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, det_collect)
    parts.sgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, assess_collect)


class BatchWorker:
    """The "batch" thread (Sec 2): serializes all runs, owns appsrc pushing.

    Constructed with the grab state (source of swap_pending), the parts, the
    collector, and publish callbacks injected by ros_io:
      publish_detections(result: FrameResult) -> None   (one TargetBoxArray)
      publish_assessment(result: FrameResult, object_id: int) -> None
                                      (one CasualtyImageCompressed per person)
    Callbacks are invoked on the worker thread; rclpy publishers are
    thread-safe (Sec 2).
    """

    def __init__(self, config: PipelineConfig, grab: GrabState,
                 parts: BatchParts, collector: ResultCollector,
                 publish_detections: Callable[[FrameResult], None],
                 publish_assessment: Callable[[FrameResult, int], None]) -> None:
        self._config = config
        self._grab = grab
        self._parts = parts
        self._collector = collector
        self._publish_detections = publish_detections
        self._publish_assessment = publish_assessment
        # _cond guards the continuous flags and stop flag; _run_lock
        # serializes run execution so the valve never toggles mid-run even in
        # the set_continuous(False)-during-auto-run / run_once race window.
        self._cond = threading.Condition()
        self._run_lock = threading.Lock()
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
        run in flight finishes or its 10 s collector timeout fires."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def run_once(self, assess: bool, timeout: float = 30.0) -> tuple[bool, str]:
        """Execute one manual run (grp_batch service thread; MutuallyExclusive
        so runs serialize with each other by construction).

        Steps (Sec 4/6): reject with (False, reason) if continuous mode is
        active or the pending queue is empty; set the valve for ``assess``;
        swap_pending(); push each item's RGBA buffer re-stamped with its
        feeder-assigned pts (appsrc block=true: a saturated GPU blocks THIS
        thread only, never the live pipeline); wait_complete; publish
        results via the callbacks; restore valve drop=true. Blocks up to
        ``timeout``. Returns (success, message: frame/box counts or reason).
        """
        with self._cond:
            if self._continuous:
                return False, "continuous mode active"
        items = self._grab.swap_pending()
        if not items:
            return False, "batch queue empty"
        return self._execute_run(items, assess, min(timeout, COLLECT_TIMEOUT_S))

    def set_continuous(self, active: bool, assess: bool) -> None:
        """Enable/disable auto-runs (grp_fast service thread, non-blocking).

        While active the worker thread runs whenever grab.pending_depth() >=
        continuous.run_size or the oldest pending item is > 200 ms old
        (Sec 4); manual run_once is rejected. Same machinery as run_once —
        no second code path (Sec 6).
        """
        with self._cond:
            self._continuous = active
            self._continuous_assess = assess if active else False
            self._oldest_since = None
            self._cond.notify_all()

    # -- worker internals ---------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._stop and not self._continuous:
                    self._cond.wait()
                if self._stop:
                    return
                assess = self._continuous_assess
            try:
                self._continuous_tick(assess)
            except Exception:
                _LOG.exception("continuous batch run failed")

    def _continuous_tick(self, assess: bool) -> None:
        """One poll of the Sec 4 auto-run condition; runs when due.

        The 200 ms rule is tracked worker-side: GrabState exposes only depth,
        so the age of the oldest pending item is measured from the poll that
        first observed a non-empty queue (<= 50 ms coarse — see review notes).
        """
        depth = self._grab.pending_depth()
        now = time.monotonic()
        if depth == 0:
            self._oldest_since = None
        elif self._oldest_since is None:
            self._oldest_since = now
        due = depth >= self._config.continuous_run_size or (
            self._oldest_since is not None
            and now - self._oldest_since >= CONTINUOUS_MAX_WAIT_S)
        if due:
            items = self._grab.swap_pending()
            self._oldest_since = None
            if items:
                success, message = self._execute_run(
                    items, assess, COLLECT_TIMEOUT_S)
                if not success:
                    _LOG.warning("continuous run incomplete: %s", message)
            return
        with self._cond:
            if not self._stop and self._continuous:
                self._cond.wait(_POLL_INTERVAL_S)

    def _execute_run(self, items: list[BatchItem], assess: bool,
                     wait_timeout: float) -> tuple[bool, str]:
        """Sec 3.3/6 run: valve -> push k -> wait k or timeout -> publish.

        Serialized by _run_lock; the valve is set before the first push and
        restored after collection, so it never toggles with buffers in
        flight (runs never overlap).
        """
        with self._run_lock:
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
                self._publish_detections(result)
                boxes += len(result.detections)
                if assess:
                    for object_id in sorted(result.assessments):
                        self._publish_assessment(result, object_id)
                        assessed += 1
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
