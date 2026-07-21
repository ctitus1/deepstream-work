"""Grab-probe state machine, batch queue, and ended-state gate.

DESIGN.md Sec 4 (race handling), Sec 5 (EOS behavior), Sec 6 (queueing):
``GrabState`` is a pure state machine — the Gst probe is a thin adapter that
calls ``on_frame`` with an injected copy callable, so arming, coalescing, the
enqueue counter, continuous-stride auto-enqueue, and snapshot-and-swap are all
unit-testable without GStreamer (Sec 10). ``Lifecycle`` is the pure
ended-state machine of Sec 5. ``copy_surface``/``make_grab_probe`` are the
only pyds/Gst-touching pieces and their imports are deferred.

Locking: one internal mutex per GrabState guards armed flags, waiter lists,
the enqueue counter, mode/stride state, and the pending deque. The probe
consumes flags under that mutex BEFORE copying (Sec 4: exactly one frame per
arming); numpy copies happen outside the mutex.
"""

from __future__ import annotations

import enum
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from config import PipelineConfig

_LOG = logging.getLogger("ds_ros_pipeline.frames")

ENDED_MESSAGE = "source ended"


class CaptureKind(enum.Enum):
    """The three independently armed one-shot captures (grp_capture, Sec 4)."""

    MOSAIC = "mosaic"          # -> /uas4/target_detections/mosaic (q90 full res)
    VLM = "vlm"                # -> detection over the captured frame, then
    #                               one CasualtyImageCompressed (PNG crop +
    #                               detection_id) per detected box
    SNAPSHOT_PNG = "snapshot"  # -> PNG file


class Mode(enum.Enum):
    """DERIVED view of the continuous state, for /ds/status only.

    The real state is two independent booleans on GrabState — detection on/off
    and assessment on/off, set by the two toggle services. This enum is what
    that pair looks like from outside; nothing stores it. Note there is no
    value for "stopped but assessment armed": while detection is off the
    assessment flag is remembered but the stream reads as ``off``, which is
    the only thing an observer can act on.
    """

    OFF = "off"
    DETECT = "detect"
    DETECT_ASSESS = "detect_assess"


@dataclass(frozen=True)
class CapturedFrame:
    """What a fulfilled capture waiter receives: the probe's RGBA copy."""

    frame_rgba: "object"   # np.ndarray HxWx4 uint8, owned copy
    pts: int               # feeder-assigned pts
    ntp_ns: int            # resolved ingest stamp


@dataclass(frozen=True)
class BatchItem:
    """One queued batch frame (Sec 5/6): stamp travels WITH the frame."""

    frame_rgba: "object"   # np.ndarray HxWx4 uint8, owned copy
    ntp_ns: int
    pts: int


class CaptureWaiter:
    """Handle returned by ``GrabState.arm``; shared by coalesced callers.

    ``wait(timeout)`` blocks the calling service thread (grp_capture,
    Reentrant) until the probe fulfills the arming or ``timeout`` (~2 s)
    elapses or the source ends. Returns the CapturedFrame, or None on
    timeout/ended (caller maps None to success=false). Multiple callers may
    wait on the same waiter (coalescing, Sec 4) and all receive the same
    frame/stamp.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: CapturedFrame | None = None

    def wait(self, timeout: float) -> CapturedFrame | None:
        if not self._event.wait(timeout):
            return None
        return self._result

    def _fulfill(self, result: CapturedFrame | None) -> None:
        """Resolve every waiting caller with ``result`` (None => failure).

        Called by GrabState outside its mutex; setting after the event fires
        is prevented by exactly-one-consumption (Sec 4) — each waiter is
        resolved once.
        """
        self._result = result
        self._event.set()


class Lifecycle:
    """Pure ended-state machine (Sec 5 "EOS behavior", Sec 10 seam).

    States: 'running' -> 'ended' (one-way, on bus EOS with loop=false).
    Thread model: ``mark_ended`` is called once from the main thread's bus
    handler; ``state``/``guard`` are called from executor threads. Internal
    mutex; no method blocks.
    """

    RUNNING = "running"
    ENDED = "ended"

    # Services that must fail fast once ended (Sec 5). clear, mode toggles,
    # run_detect* and record/stop always proceed.
    _FAIL_FAST = frozenset({"capture", "snapshot", "enqueue", "record_start"})

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ended = False

    @property
    def state(self) -> str:
        """'running' or 'ended' — published verbatim in /ds/status."""
        with self._lock:
            return self.ENDED if self._ended else self.RUNNING

    def mark_ended(self) -> None:
        """Transition to 'ended'. Idempotent."""
        with self._lock:
            self._ended = True

    def guard(self, service: str) -> str | None:
        """Fail-fast decision for a service call in the current state.

        ``service`` is the short name ('capture', 'snapshot', 'enqueue',
        'record_start', 'record_stop', 'run', 'clear', 'mode'). Returns None
        if the call may proceed; else the failure message ("source ended")
        for services that must fail fast when ended: capture/*, snapshot*,
        enqueue, record/start*. clear, mode toggles, run_detect* and
        record/stop always proceed (Sec 5). Non-blocking.
        """
        with self._lock:
            if self._ended and service in self._FAIL_FAST:
                return ENDED_MESSAGE
        return None


class GrabState:
    """Pure grab-probe state machine + batch queues (Sec 4/6).

    Constructed once with the config; wired to a Lifecycle so ending wakes
    waiters. All public methods are non-blocking except CaptureWaiter.wait.

    TWO pending deques, not one. Manually enqueued frames (``request_enqueue``,
    i.e. frames the operator deliberately picked) and continuous stride-fired
    frames are kept apart so neither run can consume the other's work — a
    continuous auto-run used to swallow hand-picked frames and publish them as
    part of the continuous stream. They differ in overflow policy too, because
    they mean different things:

      manual      -- bounded by batch.capacity; a full queue REFUSES the new
                     frame (request_enqueue returns success=false). Frames the
                     operator chose are never silently evicted.
      continuous  -- bounded by continuous.capacity; a full queue EVICTS THE
                     OLDEST (deque maxlen). It is a live buffer, so a backlog
                     must never make the continuous detections stale.

    Callers by thread:
      on_frame                 -- Gst streaming thread (grab probe), 1/frame
      arm                      -- grp_capture service threads
      request_enqueue, clear   -- grp_fast service thread
      set_continuous_detection, set_continuous_assessment
                               -- grp_fast service thread
      take_pending             -- batch worker thread (manual run dispatch)
      take_continuous, continuous_depth -- batch worker thread
      pending_depth, counters  -- /ds/status timer thread
    """

    def __init__(self, config: PipelineConfig, lifecycle: Lifecycle) -> None:
        self._config = config
        self._lifecycle = lifecycle
        self._lock = threading.Lock()
        self._armed: dict[CaptureKind, CaptureWaiter] = {}
        self._pending_enqueues = 0
        self._pending: deque[BatchItem] = deque()
        self._pending_continuous: deque[BatchItem] = deque(
            maxlen=config.continuous_capacity)
        # Two independent flags, one per toggle service. Assessment is
        # remembered while detection is off, so arming it ahead of time and
        # then starting the stream does what it looks like it does.
        self._continuous_on = False
        self._continuous_assess = False
        self._stride_count = 0
        # Manual only: it exists solely so request_enqueue's admission check
        # cannot over-admit during the copy window. Tracking continuous frames
        # here too would UNDER-admit manual ones — refusing a manual slot for a
        # continuous copy that will never occupy the manual deque (the mirror
        # of the over-admission bug this counter was added to fix).
        self._in_flight = 0
        self._ended = False
        self._enqueue_drops = 0
        self._continuous_skips = 0
        self._copy_failures = 0
        self._resolve_skips = 0

    def on_frame(self, pts: int, ntp_ns: int | None,
                 copy: Callable[[], "object"]) -> None:
        """Per-frame decision point, called by the probe adapter.

        Under the mutex: consume any armed capture flags (all armed kinds are
        served from this same frame — Sec 4 grp_capture note), consume up to
        the pending enqueue count, and evaluate continuous-stride
        auto-enqueue for the active Mode. If any consumer needs pixels, call
        ``copy()`` exactly once (outside the mutex) — it returns an owned
        RGBA numpy array — then fulfill waiters with a CapturedFrame and
        append ONE BatchItem to whichever pending deque claimed the frame,
        under that deque's overflow policy (see the class docstring: manual
        refuses and counts enqueue_drops, continuous evicts the oldest and
        counts continuous_skips). A consumed manual enqueue decrement stays
        consumed either way. When nothing is armed/pending and mode is OFF
        this returns immediately (near-zero idle cost, Sec 3.1). Never
        blocks, never raises.
        """
        if ntp_ns is None:
            # Unresolvable stamp (registry rolled + no stream NTP, Sec 5):
            # skip the frame entirely — armed flags stay armed for the next
            # frame, the enqueue counter is untouched. Logged (rate-limited)
            # so a persistent resolve miss is visible before capture timeouts.
            with self._lock:
                self._resolve_skips += 1
                skips = self._resolve_skips
            if skips == 1 or skips % 100 == 0:
                _LOG.warning(
                    "frame skipped: unresolvable ntp stamp (pts=%d, %d total)",
                    pts, skips)
            return
        with self._lock:
            waiters: list[CaptureWaiter] = []
            if self._armed:
                waiters = list(self._armed.values())
                self._armed.clear()
            take_manual = False
            take_continuous = False
            manual = self._pending_enqueues > 0
            if manual:
                self._pending_enqueues -= 1
            fire = False
            if self._continuous_on:
                # Stride clock advances on EVERY frame while mode is on,
                # regardless of who consumes the frame (steady cadence).
                fire = self._stride_count % self._config.continuous_stride == 0
                self._stride_count += 1
            if manual:
                # Manual decrement takes precedence over a same-frame stride
                # fire: at most one BatchItem per frame (unique pts, Sec 6).
                # Exactly one of take_* is ever true, so the in-flight
                # accounting below can never double-count a frame.
                if len(self._pending) >= self._config.batch_capacity:
                    self._enqueue_drops += 1
                else:
                    take_manual = True
                    # Counted in request_enqueue's depth so the copy window
                    # cannot over-admit at the capacity boundary.
                    self._in_flight += 1
            elif fire:
                # No capacity gate: the continuous deque is maxlen-bounded and
                # evicts its oldest entry on append, so a stride fire is always
                # admitted and the buffer always holds the most recent frames.
                take_continuous = True
        if not waiters and not take_manual and not take_continuous:
            return
        try:
            frame = copy()
        except Exception:
            with self._lock:
                self._copy_failures += 1
                if take_manual:
                    self._in_flight -= 1
            _LOG.exception("surface copy failed (pts=%d)", pts)
            for waiter in waiters:
                waiter._fulfill(None)
            return
        captured = CapturedFrame(frame_rgba=frame, pts=pts, ntp_ns=ntp_ns)
        for waiter in waiters:
            waiter._fulfill(captured)
        item = BatchItem(frame_rgba=frame, ntp_ns=ntp_ns, pts=pts)
        if take_manual:
            with self._lock:
                self._in_flight -= 1
                # Defensive: only take_pending/clear touch this deque and both
                # shrink it, so the reservation above is what actually keeps
                # the capacity exact — this branch is not the guard.
                if len(self._pending) < self._config.batch_capacity:
                    self._pending.append(item)
                else:
                    self._enqueue_drops += 1
        elif take_continuous:
            with self._lock:
                if len(self._pending_continuous) == self._pending_continuous.maxlen:
                    # maxlen makes the append itself evict the oldest; count it
                    # against the continuous stream, which is whose frame is
                    # being dropped (this used to charge enqueue_drops).
                    self._continuous_skips += 1
                self._pending_continuous.append(item)

    def arm(self, kind: CaptureKind) -> CaptureWaiter:
        """Arm a one-shot capture; returns the waiter to block on.

        Coalescing (Sec 4): if ``kind`` is already armed and not yet
        consumed, the existing waiter is returned (both callers get the same
        stamp). A call after consumption-but-before-publish re-arms for the
        following frame (fresh waiter). If lifecycle is ended, returns a
        waiter already resolved to None. Non-blocking.
        """
        if self._lifecycle.state == Lifecycle.ENDED:
            waiter = CaptureWaiter()
            waiter._fulfill(None)
            return waiter
        with self._lock:
            if self._ended:
                # notify_ended ran between the lifecycle check and here; a
                # waiter inserted now would never be woken (no second wakeup).
                waiter = CaptureWaiter()
                waiter._fulfill(None)
                return waiter
            existing = self._armed.get(kind)
            if existing is not None:
                return existing
            waiter = CaptureWaiter()
            self._armed[kind] = waiter
            return waiter

    def request_enqueue(self) -> tuple[bool, int]:
        """Increment the pending-enqueue counter (never coalesces, Sec 4).

        Returns (success, resulting_queue_depth_message_value): success=False
        if the MANUAL deque is already at batch_capacity. The depth counts
        both queued frames and not-yet-consumed pending requests, so N
        back-to-back calls report 1..N (Sec 10 test 5). Frames mid-copy
        (consumed but not yet appended) count too, so the copy window cannot
        over-admit at the capacity boundary. The continuous deque is not part
        of this arithmetic at all — it has its own capacity, so a busy
        continuous stream can never consume the operator's enqueue budget.
        Non-blocking.
        """
        with self._lock:
            depth = (len(self._pending) + self._pending_enqueues
                     + self._in_flight)
            if depth >= self._config.batch_capacity:
                return False, depth
            self._pending_enqueues += 1
            return True, depth + 1

    def clear(self) -> tuple[int, int]:
        """Empty BOTH pending deques; returns (manual, continuous) removed.

        The /ds/batch/clear service is a literal clear-everything button, so
        it does not privilege either queue. A snapshot already claimed by a
        run in flight is unaffected (Sec 4) — this only empties what is still
        pending. Continuous refills within a stride or two, so wiping it
        mid-stream is close to invisible.
        """
        with self._lock:
            removed = len(self._pending)
            removed_continuous = len(self._pending_continuous)
            self._pending.clear()
            self._pending_continuous.clear()
            return removed, removed_continuous

    def clear_continuous(self) -> int:
        """Empty the continuous deque only; returns frames removed.

        Used when continuous mode is switched off, so the stream's leftover
        auto-enqueued frames cannot surface later inside an unrelated manual
        run. Deliberately leaves the manual deque alone: turning a mode off
        must not discard frames the operator enqueued by hand.
        """
        with self._lock:
            removed = len(self._pending_continuous)
            self._pending_continuous.clear()
            return removed

    def set_continuous_detection(self, on: bool) -> bool:
        """Start/stop the continuous stream (/ds/mode/toggle_detection).

        Returns the previous state. The stride clock resets on a real
        transition, so a restarted stream begins a fresh cadence. Leaves the
        assessment flag untouched — that is the other toggle's business.
        """
        with self._lock:
            previous = self._continuous_on
            if on is not previous:
                self._continuous_on = on
                self._stride_count = 0
            return previous

    def set_continuous_assessment(self, assess: bool) -> bool:
        """Enable/disable assessment (/ds/mode/toggle_assessment).

        Returns the previous state. Independent of whether the stream is
        running: it only decides whether the worker opens the assess valve on
        the next run, so flipping it mid-stream must NOT disturb the stride
        clock — that would put a seam in what is one uninterrupted stream —
        and setting it while stopped is remembered for when detection starts.
        """
        with self._lock:
            previous = self._continuous_assess
            self._continuous_assess = assess
            return previous

    @property
    def continuous_on(self) -> bool:
        with self._lock:
            return self._continuous_on

    @property
    def continuous_assess(self) -> bool:
        with self._lock:
            return self._continuous_assess

    @property
    def mode(self) -> Mode:
        """The two flags as the single derived Mode /ds/status publishes."""
        with self._lock:
            if not self._continuous_on:
                return Mode.OFF
            return Mode.DETECT_ASSESS if self._continuous_assess else Mode.DETECT

    @staticmethod
    def _take_newest(queue: "deque[BatchItem]",
                     limit: int | None) -> list[BatchItem]:
        """Remove up to ``limit`` NEWEST items; return them oldest-first.

        Both ends are load-bearing, and they serve different purposes:

        *Selection* is newest-first (pop from the right, where append puts the
        most recent frame). When the queue is longer than one run — which is
        the point of a long queue — the run should infer the freshest frames
        available, not work through a stale backlog front-to-back.

        *Publish order* is oldest-first (the reverse below). All items in a
        run are inferred together, so their relative order costs nothing to
        fix, and emitting a TargetBoxArray whose header.stamp moves backwards
        while seq moves forwards is a trap for every downstream consumer.
        Selecting newest and publishing in stamp order gives both.

        Duplicate pts are dropped: the collector joins results back to items
        by pts (Sec 6), so two items sharing one would collide. Admission
        already makes this impossible (a frame is offered to exactly one
        queue, once) — this keeps the join's precondition true by
        construction rather than by that argument holding forever.
        """
        taken: list[BatchItem] = []
        seen: set[int] = set()
        while queue and (limit is None or len(taken) < limit):
            item = queue.pop()
            if item.pts in seen:
                continue
            seen.add(item.pts)
            taken.append(item)
        taken.reverse()
        return taken

    def take_pending(self, limit: int | None = None) -> list[BatchItem]:
        """Claim up to ``limit`` newest MANUAL items (see _take_newest).

        Frames enqueued after this returns land in the queue as usual and are
        never destroyed; there is no post-run clear. Continuous frames are
        never returned here — a manual run publishes exactly the frames the
        operator enqueued.
        """
        with self._lock:
            return self._take_newest(self._pending, limit)

    def take_continuous(self, limit: int | None = None) -> list[BatchItem]:
        """Claim up to ``limit`` newest CONTINUOUS items (see _take_newest).

        The counterpart of take_pending for the auto-run path; symmetrically,
        it never returns a manually enqueued frame.
        """
        with self._lock:
            return self._take_newest(self._pending_continuous, limit)

    def pending_depth(self) -> int:
        """MANUAL pending-deque depth (/ds/status queue_depth, enqueue reply).

        Deliberately still the manual depth alone: it is the number a caller
        of /ds/batch/enqueue is counting, and mixing the continuous stream's
        buffer into it would make N enqueues report something other than N.
        """
        with self._lock:
            return len(self._pending)

    def continuous_depth(self) -> int:
        """Continuous pending-deque depth (/ds/status, batch worker tick)."""
        with self._lock:
            return len(self._pending_continuous)

    def counters(self) -> dict[str, int]:
        """Drop counters for /ds/status: manual-enqueue drops (manual deque
        full), continuous-mode skips (oldest evicted from the full continuous
        deque), surface-copy failures, and frames skipped for an unresolvable
        ntp stamp."""
        with self._lock:
            return {
                "enqueue_drops": self._enqueue_drops,
                "continuous_skips": self._continuous_skips,
                "copy_failures": self._copy_failures,
                "resolve_skips": self._resolve_skips,
            }

    def notify_ended(self) -> None:
        """Wake every armed waiter with None (=> success=false "source ended").

        Called from the main thread's bus-EOS handler after
        Lifecycle.mark_ended (Sec 5).
        """
        with self._lock:
            self._ended = True
            waiters = list(self._armed.values())
            self._armed.clear()
        for waiter in waiters:
            waiter._fulfill(None)


def copy_surface(gst_buffer, batch_id: int = 0) -> "object":
    """Map the unified-memory RGBA surface and return an owned numpy copy.

    ``pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)`` ->
    ``np.array(..., copy=True)`` (~14.75 MB, a few ms — Sec 6). Called only
    from the grab probe's streaming thread, and only when a consumer is
    armed. pyds/numpy imported here, not at module top.
    """
    import numpy as np
    import pyds

    surface = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
    try:
        return np.array(surface, copy=True, order="C")
    finally:
        try:
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), batch_id)
        except Exception:
            pass


def _first_frame_meta(buffer) -> tuple["object", int]:
    """(frame_meta, batch_id) of the buffer's first NvDsFrameMeta, or (None, 0).

    Branch G's mux runs batch-size=1 (Sec 3.1) so the first entry is the only
    entry. pyds imported here, not at module top.
    """
    import pyds

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if batch_meta is None or batch_meta.frame_meta_list is None:
        return None, 0
    frame_meta = pyds.NvDsFrameMeta.cast(batch_meta.frame_meta_list.data)
    return frame_meta, frame_meta.batch_id


def make_grab_probe(state: GrabState, registry, resolve_fn) -> Callable:
    """Build the Gst pad-probe callable for caps_grab's src pad (Sec 3.1).

    The returned ``probe(pad, info)`` extracts (pts, frame_meta) from the
    buffer, resolves ntp_ns via ``resolve_fn(registry, pts, frame_meta,
    buffer)``, and calls ``state.on_frame(pts, ntp_ns, copy)`` with ``copy``
    bound to ``copy_surface`` on this buffer. Always returns
    Gst.PadProbeReturn.OK. Runs on the grab branch's streaming thread.
    """
    from gi.repository import Gst

    def probe(pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        try:
            frame_meta, batch_id = _first_frame_meta(buffer)
            # Sec 5: nvstreammux regenerates buffer.pts from zero; the
            # feeder-assigned pts survives in frame_meta.buf_pts.
            pts = (int(frame_meta.buf_pts) if frame_meta is not None
                   else int(buffer.pts))
            ntp_ns = resolve_fn(registry, pts, frame_meta, buffer)
            state.on_frame(pts, ntp_ns,
                           lambda: copy_surface(buffer, batch_id))
        except Exception:
            _LOG.exception("grab probe failed")
        return Gst.PadProbeReturn.OK

    return probe
