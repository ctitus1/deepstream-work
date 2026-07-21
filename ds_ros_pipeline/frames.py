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
    """The four independently armed one-shot captures (grp_capture, Sec 4)."""

    MOSAIC = "mosaic"          # -> /mosaic_compressed (JPEG q90, full res)
    VLM = "vlm"                # -> detection over the captured frame, then
    #                               one CasualtyImageCompressed (PNG crop +
    #                               detection_id) per detected box
    SNAPSHOT_PNG = "snapshot"  # -> PNG file
    SNAPSHOT_RAW = "snapshot_raw"  # -> .ppm + .json sidecar


class Mode(enum.Enum):
    """Continuous-mode flag (Sec 4 toggles). Turning one on turns the other off."""

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
    """Pure grab-probe state machine + batch queue (Sec 4/6).

    Constructed once with the config; wired to a Lifecycle so ending wakes
    waiters. All public methods are non-blocking except CaptureWaiter.wait.

    Callers by thread:
      on_frame            -- Gst streaming thread (grab probe), one call/frame
      arm                 -- grp_capture service threads
      request_enqueue     -- grp_fast service thread
      clear               -- grp_fast service thread
      set_mode            -- grp_fast service thread
      swap_pending        -- batch worker thread / grp_batch service thread
      pending_depth, counters -- /ds/status timer thread
    """

    def __init__(self, config: PipelineConfig, lifecycle: Lifecycle) -> None:
        self._config = config
        self._lifecycle = lifecycle
        self._lock = threading.Lock()
        self._armed: dict[CaptureKind, CaptureWaiter] = {}
        self._pending_enqueues = 0
        self._pending: deque[BatchItem] = deque()
        self._mode = Mode.OFF
        self._stride_count = 0
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
        append BatchItems to the pending deque (respecting batch_capacity:
        a full deque counts a drop instead of appending; continuous mode
        skips, manual enqueue decrements remain consumed). When nothing is
        armed/pending and mode is OFF this returns immediately (near-zero
        idle cost, Sec 3.1). Never blocks, never raises.
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
            enqueue = False
            manual = self._pending_enqueues > 0
            if manual:
                self._pending_enqueues -= 1
            fire = False
            if self._mode is not Mode.OFF:
                # Stride clock advances on EVERY frame while mode is on,
                # regardless of who consumes the frame (steady cadence).
                fire = self._stride_count % self._config.continuous_stride == 0
                self._stride_count += 1
            if manual:
                # Manual decrement takes precedence over a same-frame stride
                # fire: at most one BatchItem per frame (unique pts, Sec 6).
                if len(self._pending) >= self._config.batch_capacity:
                    self._enqueue_drops += 1
                else:
                    enqueue = True
            elif fire:
                if len(self._pending) >= self._config.batch_capacity:
                    self._continuous_skips += 1
                else:
                    enqueue = True
            if enqueue:
                # Counted in request_enqueue's depth so the copy window
                # cannot over-admit at the capacity boundary.
                self._in_flight += 1
        if not waiters and not enqueue:
            return
        try:
            frame = copy()
        except Exception:
            with self._lock:
                self._copy_failures += 1
                if enqueue:
                    self._in_flight -= 1
            _LOG.exception("surface copy failed (pts=%d)", pts)
            for waiter in waiters:
                waiter._fulfill(None)
            return
        captured = CapturedFrame(frame_rgba=frame, pts=pts, ntp_ns=ntp_ns)
        for waiter in waiters:
            waiter._fulfill(captured)
        if enqueue:
            with self._lock:
                self._in_flight -= 1
                if len(self._pending) < self._config.batch_capacity:
                    self._pending.append(
                        BatchItem(frame_rgba=frame, ntp_ns=ntp_ns, pts=pts))
                else:
                    self._enqueue_drops += 1

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
        if pending deque is already at batch_capacity. The depth counts both
        queued frames and not-yet-consumed pending requests, so N back-to-back
        calls report 1..N (Sec 10 test 5). Frames mid-copy (consumed but not
        yet appended) count too, so the copy window cannot over-admit at the
        capacity boundary. Non-blocking.
        """
        with self._lock:
            depth = (len(self._pending) + self._pending_enqueues
                     + self._in_flight)
            if depth >= self._config.batch_capacity:
                return False, depth
            self._pending_enqueues += 1
            return True, depth + 1

    def clear(self) -> int:
        """Empty the *pending* deque only (Sec 4); returns frames removed."""
        with self._lock:
            removed = len(self._pending)
            self._pending.clear()
            return removed

    def set_mode(self, mode: Mode) -> Mode:
        """Set continuous mode; turning either on turns the other off (Sec 4).

        Returns the previous mode. The stride counter resets on transition.
        """
        with self._lock:
            previous = self._mode
            if mode is not previous:
                self._mode = mode
                self._stride_count = 0
            return previous

    @property
    def mode(self) -> Mode:
        with self._lock:
            return self._mode

    def swap_pending(self) -> list[BatchItem]:
        """Atomically swap the pending deque for a fresh empty one (Sec 4/6).

        Returns the swapped-out snapshot (possibly empty). Enqueues during a
        run land in the new deque and are never destroyed; there is no
        post-run clear.
        """
        with self._lock:
            snapshot = list(self._pending)
            self._pending = deque()
            return snapshot

    def pending_depth(self) -> int:
        """Current pending-deque depth (for /ds/status and enqueue replies)."""
        with self._lock:
            return len(self._pending)

    def counters(self) -> dict[str, int]:
        """Drop counters for /ds/status: manual-enqueue drops (deque full),
        continuous-mode skips (deque full), surface-copy failures, and
        frames skipped for an unresolvable ntp stamp."""
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
