"""Ingest-timestamp registry and resolution (DESIGN.md Sec 5).

``TimestampRegistry`` maps feeder-assigned buffer pts (globally unique,
strictly monotonic across loop wraps — Sec 5) to NTP-synced wall-clock
nanoseconds captured by the ``ingest_stamp`` probe on ``pace``'s src pad.
``resolve()`` is the single stamp-resolution funnel: it prefers in-band sender
NTP (RTSP upgrade path, via ``deepstream_yolo.assessment_runtime
.frame_timestamp`` + ``deepstream_yolo.frame_wire.is_wall_clock_timestamp``)
and falls back to the registry. Helpers convert one resolved ntp_ns into every
downstream representation (ROS sec/nanosec, UTC filename fragment).

GPU/ROS-free: imports only stdlib; the deepstream_yolo helpers are imported
lazily inside ``resolve`` so unit tests run without the src/ tree's optional
dependencies.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

NS_PER_SEC = 1_000_000_000
REGISTRY_CAPACITY = 2048  # ~68 s at 30 fps (Sec 5)

_HELPERS_UNSET = object()
# Cached (frame_timestamp, is_wall_clock_timestamp) pair, or None when the
# deepstream_yolo tree is unavailable (unit tests without pyds/gi). Tests may
# assign a fake pair here to exercise the stream-NTP preference path.
_stream_helpers = _HELPERS_UNSET


def _load_stream_helpers():
    """Lazily import the stream-NTP helpers; None if unavailable."""
    global _stream_helpers
    if _stream_helpers is _HELPERS_UNSET:
        try:
            from deepstream_yolo.assessment_runtime import frame_timestamp
            from deepstream_yolo.frame_wire import is_wall_clock_timestamp
        except Exception:
            _stream_helpers = None
        else:
            _stream_helpers = (frame_timestamp, is_wall_clock_timestamp)
    return _stream_helpers


class TimestampRegistry:
    """Bounded pts -> ntp_ns map. Sec 5: OrderedDict of 2048 entries, one mutex.

    Thread model: ``stamp()`` is called only from the live pipeline's streaming
    thread (the ingest_stamp probe); ``resolve()``/``get()`` are called from
    streaming threads (preview publish, sidecar probe) and from service/worker
    threads. All methods take the single internal mutex; none ever blocks
    beyond that lock (no I/O, no waits). Never flushed on loop wrap —
    feeder-assigned pts never repeats (Sec 5).
    """

    def __init__(self, capacity: int = REGISTRY_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, int] = OrderedDict()
        self._capacity = capacity

    def stamp(self, pts: int) -> int:
        """Record ``registry[pts] = time.time_ns()`` and return that value.

        Evicts the oldest entry beyond capacity. Called from the ingest_stamp
        pad probe (Gst streaming thread) once per trunk buffer.
        """
        ntp_ns = time.time_ns()
        with self._lock:
            self._entries[pts] = ntp_ns
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        return ntp_ns

    def get(self, pts: int) -> int | None:
        """Registry-only lookup: ntp_ns for ``pts``, or None if rolled out."""
        with self._lock:
            return self._entries.get(pts)

    def __len__(self) -> int:
        """Current entry count (for tests asserting the capacity bound)."""
        with self._lock:
            return len(self._entries)


def resolve(registry: TimestampRegistry, pts: int, frame_meta=None,
            buffer=None) -> int | None:
    """Resolve the authoritative ntp_ns for a frame (Sec 5, one function).

    Preference order: (1) sender-side NTP carried in stream meta, when
    ``frame_meta``/``buffer`` are given and ``frame_timestamp`` yields a
    wall-clock source per ``is_wall_clock_timestamp``; (2) the ingest
    ``registry`` keyed by ``pts``. Returns None only if both miss (caller
    logs and skips the frame). Non-blocking; callable from any thread.
    """
    if frame_meta is not None or buffer is not None:
        helpers = _load_stream_helpers()
        if helpers is not None:
            frame_timestamp, is_wall_clock_timestamp = helpers
            source, ntp_ns = frame_timestamp(frame_meta, buffer)
            if ntp_ns is not None and is_wall_clock_timestamp(source):
                return ntp_ns
    return registry.get(pts)


def split_stamp(ntp_ns: int) -> tuple[int, int]:
    """ntp_ns -> (sec, nanosec) for ROS ``builtin_interfaces/Time`` fields."""
    return divmod(ntp_ns, NS_PER_SEC)


def utc_tag(ntp_ns: int) -> str:
    """ntp_ns -> filename fragment like ``20260720T153001.123Z`` (Sec 5 Disk)."""
    sec, nanosec = divmod(ntp_ns, NS_PER_SEC)
    base = time.strftime("%Y%m%dT%H%M%S", time.gmtime(sec))
    return f"{base}.{nanosec // 1_000_000:03d}Z"
