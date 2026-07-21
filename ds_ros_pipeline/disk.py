"""Recorder branch lifecycle, sidecars, snapshot writers, disk worker.

DESIGN.md Sec 7 (attach / momentary-block detach / async drain /
force-finalize), Sec 3.1 Branch R (exact element properties), Sec 5
(sidecars, filenames). The detach ordering is implemented ONCE in
``DetachSequencer`` as a pure state machine driven by injected callables, so
Sec 10 can assert the exact callback ordering (unlink -> send EOS -> release
pad -> REMOVE; drain wait strictly after; timeout -> force-finalize) without
GStreamer. ``Recorder`` binds that machine to real Gst elements.

Threads: attach/stop run on the grp_record service thread (MutuallyExclusive
— start/stop serialize with each other only); the IDLE probe callback runs on
the trunk streaming thread for microseconds; the EOS probe on sink_rec.sink
runs on the branch's own streaming thread; PNG/PPM/sidecar writes run on the
"disk" worker thread (Sec 2).

Gst imports are deferred into the methods that touch elements so tests.py can
import ``DetachSequencer`` and the pure path/sidecar helpers without GStreamer.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config import PipelineConfig
from live_pipeline import LiveParts
from timestamps import TimestampRegistry, utc_tag

Q_REC_MAX_BUFFERS = 30                    # Sec 3.1: 1 s at 30 fps, leaky=2
Q_DISK_BYTES_TS = 64 * 1024 * 1024        # Sec 3.1: non-leaky, byte-bounded
Q_DISK_BYTES_RAW = 256 * 1024 * 1024      # Sec 3.1 raw variant
IFRAME_INTERVAL = 30                      # Sec 3.1/7: 1 s closed GOPs
IDR_INTERVAL = 30
CONTROL_RATE_CBR = 1


@dataclass(frozen=True)
class RecordStats:
    """record/stop response payload (Sec 4): path, counts, drain outcome."""

    path: str
    frames_written: int
    frames_dropped: int
    drained: bool          # False => drain timed out, file force-finalized


class DetachSequencer:
    """Pure Sec 7 stop state machine; every side effect is an injected callable.

    Constructor callables (all invoked exactly once per stop, no arguments):
      install_idle_probe(cb) -- installs cb as a Gst.PadProbeType.IDLE probe
                                on the tee request pad (Sec 7 step 1; IDLE,
                                deliberately NOT BLOCK_DOWNSTREAM) and calls
                                it from the streaming thread (tests call it
                                inline)
      unlink       -- tee_pad.unlink(q_rec.sink)
      send_eos     -- q_rec.sink.send_event(EOS); returns immediately even
                      against a full q_rec (leaky queues drop, never block)
      release_pad  -- t_ingest.release_request_pad(tee_pad)
      wait_drain(timeout) -> bool -- waits on the drain_done event set by the
                      sink EOS probe; True iff it fired within timeout
      finalize     -- set branch elements NULL + remove from pipeline

    ``stop(timeout)`` (service thread) runs: install_idle_probe(cb) where cb
    does, in order, unlink; send_eos; release_pad; then returns REMOVE (trunk
    hold = the callback body, Sec 7 step 2); then wait_drain(timeout); then
    finalize — unconditionally, drained or not (Sec 7 steps 3-4). Returns
    drained: bool. No drain wait ever occurs before unlink+release (Sec 10).
    """

    def __init__(self, install_idle_probe: Callable[[Callable[[], None]], None],
                 unlink: Callable[[], None], send_eos: Callable[[], None],
                 release_pad: Callable[[], None],
                 wait_drain: Callable[[float], bool],
                 finalize: Callable[[], None]) -> None:
        self._install_idle_probe = install_idle_probe
        self._unlink = unlink
        self._send_eos = send_eos
        self._release_pad = release_pad
        self._wait_drain = wait_drain
        self._finalize = finalize

    def stop(self, timeout: float) -> bool:
        """Run the full detach sequence; blocks the calling service thread up
        to ``timeout`` in wait_drain only. Returns True iff drained."""
        def callback() -> None:
            self._unlink()
            self._send_eos()
            self._release_pad()

        self._install_idle_probe(callback)
        drained = self._wait_drain(timeout)
        self._finalize()
        return drained


class _Branch:
    """Live handles for one attached Branch R instance."""

    def __init__(self, elements: list, tee_pad, q_rec, sink_rec,
                 path: Path, sidecar_path: Path) -> None:
        self.elements = elements          # upstream -> downstream order
        self.tee_pad = tee_pad
        self.q_rec = q_rec
        self.sink_rec = sink_rec
        self.path = path
        self.sidecar_path = sidecar_path
        self.drain_done = threading.Event()
        self.frames_written = 0           # rec_sidecar probe (single writer)
        self.frames_dropped = 0           # q_rec overrun accounting

    def stats(self, drained: bool) -> RecordStats:
        return RecordStats(path=str(self.path),
                           frames_written=self.frames_written,
                           frames_dropped=self.frames_dropped,
                           drained=drained)


class Recorder:
    """Dynamic Branch R owner: h265/MPEG-TS and raw-I420 variants (Sec 3.1/7).

    One instance per process. State: 'idle' | 'recording' | 'finalized'
    ('finalized' = the loop=false source EOS already drained the branch —
    stop becomes idempotent, Sec 5). All public methods are called from the
    grp_record service thread except ``on_source_eos`` (main thread bus
    handler) and the internal probes (streaming threads); internal mutex.
    """

    def __init__(self, config: PipelineConfig, live: LiveParts,
                 registry: TimestampRegistry) -> None:
        self._config = config
        self._live = live
        self._registry = registry
        self._lock = threading.Lock()
        self._state = "idle"
        self._branch: _Branch | None = None
        self._source_ended = False
        self._finalize_pending = False    # force-finalize NULL still wedged
        self._final_message = ""
        self._final_stats: RecordStats | None = None

    def start(self, raw: bool) -> tuple[bool, str]:
        """Attach the recorder branch (Sec 7 Start): request a t_ingest pad,
        build Branch R (h265 variant, or raw when ``raw``) with Sec 3.1's
        exact properties, add to pipeline, sync_state_with_parent, link;
        install the rec_sidecar probe (q_rec src pad — counts/writes only
        frames that survived the leaky queue) and the EOS probe on
        sink_rec.sink (sets drain_done, Sec 7 step 3).

        Returns (True, path) or (False, reason) if already recording or the
        source ended. No live-graph state change; non-blocking apart from
        element construction.
        """
        with self._lock:
            if self._state == "recording":
                return False, "already recording"
            if self._state == "finalized" or self._source_ended:
                return False, "source ended"
            if self._finalize_pending:
                return False, "previous recording still finalizing"
            branch = self._attach(raw)
            self._branch = branch
            self._state = "recording"
            return True, str(branch.path)

    def stop(self) -> tuple[bool, str, RecordStats | None]:
        """Momentary-block detach + async drain via DetachSequencer (Sec 7).

        Blocks the grp_record thread up to record.stop_timeout in the drain
        wait only — the trunk is never stalled. Idempotent after 'finalized'
        (returns the finalized file's stats, success=True). (False, reason,
        None) if idle. Message formatting is ros_io's job; stats carry path,
        frames written (sidecar count), frames dropped (q_rec leak count via
        its 'overrun'/drop accounting), drained.
        """
        with self._lock:
            if self._state == "finalized":
                return True, self._final_message, self._final_stats
            if self._state != "recording" or self._branch is None:
                return False, "not recording", None
            branch = self._branch
            self._branch = None           # claim: on_source_eos backs off
        drained = self._detach(branch)
        with self._lock:
            self._state = "idle"
        message = "finalize_pending" if self._finalize_pending else str(branch.path)
        return True, message, branch.stats(drained)

    def on_source_eos(self) -> None:
        """loop=false end of media (Sec 5): the pipeline-wide EOS already
        drained the branch through the identical machinery; mark state
        'finalized' and capture stats. Called once from the bus handler."""
        with self._lock:
            self._source_ended = True
            branch = self._branch
            if self._state != "recording" or branch is None:
                return
            self._branch = None
        # The bus posts EOS only after every sink received it, so the branch
        # sink already saw EOS; the wait is a formality.
        drained = branch.drain_done.wait(self._config.record_stop_timeout)
        self._release_tee_pad(branch)
        self._finalize_branch(branch)
        stats = branch.stats(drained)
        with self._lock:
            self._state = "finalized"
            self._final_message = str(branch.path)
            self._final_stats = stats

    @property
    def state(self) -> str:
        """'idle' | 'recording' | 'finalized' — for /ds/status. Any thread."""
        with self._lock:
            return self._state

    # -- Gst binding ---------------------------------------------------------

    def _attach(self, raw: bool) -> _Branch:
        from gi.repository import Gst

        config = self._config
        live = self._live
        width = live.source.width
        height = live.source.height
        output_dir = Path(config.record_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = record_path(output_dir, time.time_ns(), raw, width, height)
        sidecar_path = path.with_suffix(".jsonl")

        tee_pad = live.t_ingest.request_pad_simple("src_%u")
        if tee_pad is None:
            raise RuntimeError("failed to request t_ingest src pad")

        q_rec = _make("queue", "q_rec")
        q_rec.set_property("leaky", 2)
        q_rec.set_property("max-size-buffers", Q_REC_MAX_BUFFERS)
        q_rec.set_property("max-size-bytes", 0)
        q_rec.set_property("max-size-time", 0)
        if q_rec.find_property("flush-on-eos"):
            q_rec.set_property("flush-on-eos", False)

        conv_rec = _make("nvvideoconvert", "conv_rec")
        caps_rec = _make("capsfilter", "caps_rec")
        q_disk = _make("queue", "q_disk")
        q_disk.set_property("leaky", 0)
        q_disk.set_property("max-size-buffers", 0)
        q_disk.set_property("max-size-time", 0)
        sink_rec = _make("filesink", "sink_rec")
        sink_rec.set_property("location", str(path))
        sink_rec.set_property("sync", False)
        sink_rec.set_property("async", False)

        if raw:
            caps_rec.set_property(
                "caps", Gst.Caps.from_string("video/x-raw,format=I420"))
            q_disk.set_property("max-size-bytes", Q_DISK_BYTES_RAW)
            elements = [q_rec, conv_rec, caps_rec, q_disk, sink_rec]
        else:
            caps_rec.set_property(
                "caps",
                Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
            q_disk.set_property("max-size-bytes", Q_DISK_BYTES_TS)
            enc_rec = _make("nvv4l2h265enc", "enc_rec")
            enc_rec.set_property("bitrate", config.record_bitrate)
            enc_rec.set_property("control-rate", CONTROL_RATE_CBR)
            enc_rec.set_property("iframeinterval", IFRAME_INTERVAL)
            enc_rec.set_property("idrinterval", IDR_INTERVAL)
            parse_rec = _make("h265parse", "parse_rec")
            parse_rec.set_property("config-interval", -1)
            mux_rec = _make("mpegtsmux", "mux_rec")
            elements = [q_rec, conv_rec, caps_rec, enc_rec, parse_rec,
                        mux_rec, q_disk, sink_rec]

        for element in elements:
            live.pipeline.add(element)
        for upstream, downstream in zip(elements, elements[1:]):
            if not upstream.link(downstream):
                raise RuntimeError(
                    f"failed to link {upstream.get_name()} -> "
                    f"{downstream.get_name()}")

        branch = _Branch(elements, tee_pad, q_rec, sink_rec, path, sidecar_path)
        self._install_branch_probes(branch)

        for element in reversed(elements):
            element.sync_state_with_parent()
        if tee_pad.link(q_rec.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            raise RuntimeError("failed to link t_ingest to q_rec")
        return branch

    def _install_branch_probes(self, branch: _Branch) -> None:
        from gi.repository import Gst

        registry = self._registry

        def rec_sidecar(_pad, info):
            buffer = info.get_buffer()
            pts = buffer.pts
            ntp_ns = registry.get(pts)
            if ntp_ns is None:
                ntp_ns = time.time_ns()
            append_sidecar_line(branch.sidecar_path, pts, ntp_ns)
            branch.frames_written += 1
            return Gst.PadProbeReturn.OK

        def sink_eos(_pad, info):
            if info.get_event().type == Gst.EventType.EOS:
                branch.drain_done.set()
            return Gst.PadProbeReturn.OK

        def overrun(_queue):
            branch.frames_dropped += 1

        branch.q_rec.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, rec_sidecar)
        branch.sink_rec.get_static_pad("sink").add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM, sink_eos)
        branch.q_rec.connect("overrun", overrun)

    def _detach(self, branch: _Branch) -> bool:
        from gi.repository import Gst

        tee = self._live.t_ingest
        q_rec_sink = branch.q_rec.get_static_pad("sink")

        def install_idle_probe(cb: Callable[[], None]) -> None:
            def probe(_pad, _info):
                cb()
                return Gst.PadProbeReturn.REMOVE
            branch.tee_pad.add_probe(Gst.PadProbeType.IDLE, probe)

        sequencer = DetachSequencer(
            install_idle_probe=install_idle_probe,
            unlink=lambda: branch.tee_pad.unlink(q_rec_sink),
            send_eos=lambda: q_rec_sink.send_event(Gst.Event.new_eos()),
            release_pad=lambda: tee.release_request_pad(branch.tee_pad),
            wait_drain=branch.drain_done.wait,
            finalize=lambda: self._finalize_branch(branch),
        )
        return sequencer.stop(self._config.record_stop_timeout)

    def _release_tee_pad(self, branch: _Branch) -> None:
        """Source-EOS path only: no data flows, unlink directly (Sec 5)."""
        q_rec_sink = branch.q_rec.get_static_pad("sink")
        if branch.tee_pad.is_linked():
            branch.tee_pad.unlink(q_rec_sink)
        self._live.t_ingest.release_request_pad(branch.tee_pad)

    def _finalize_branch(self, branch: _Branch) -> None:
        """Sec 7 steps 3-4 tail: NULL + remove. Drained => inline (fast,
        safe). Not drained => the NULL runs on a disposable thread with a
        second bounded wait, so a truly wedged filesink cannot hang the
        grp_record service thread forever; finalize_pending reports it."""
        if branch.drain_done.is_set():
            self._null_and_remove(branch)
            return
        with self._lock:
            self._finalize_pending = True

        def force() -> None:
            try:
                self._null_and_remove(branch)
            finally:
                with self._lock:
                    self._finalize_pending = False

        worker = threading.Thread(target=force, name="rec-force-finalize",
                                  daemon=True)
        worker.start()
        worker.join(self._config.record_stop_timeout)
        if worker.is_alive():
            print(f"WARNING: force-finalize of {branch.path} still pending "
                  "(filesink wedged)", file=sys.stderr, flush=True)

    def _null_and_remove(self, branch: _Branch) -> None:
        from gi.repository import Gst

        for element in branch.elements:
            element.set_state(Gst.State.NULL)
        for element in branch.elements:
            self._live.pipeline.remove(element)


def _make(factory: str, name: str):
    from gi.repository import Gst

    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"failed to create element {factory} ({name})")
    return element


class DiskWorker:
    """The "disk" thread (Sec 2): serializes PNG/PPM encodes and file writes
    off the streaming and service threads.

    ``submit(fn) -> wait()``-style: submit returns a handle whose
    ``wait(timeout)`` blocks the caller (a grp_capture thread for snapshots —
    the service must not return before the file is closed, Sec 4) until fn
    ran on the worker. fn runs exactly once, in submission order.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="disk",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Drain queued jobs, then join (main thread, shutdown)."""
        if self._thread is None:
            return
        self._jobs.put(None)
        self._thread.join()
        self._thread = None

    def submit(self, fn: Callable[[], None]) -> "threading.Event":
        """Enqueue fn; returns an Event set after fn completes (or raises)."""
        done = threading.Event()
        self._jobs.put((fn, done))
        return done

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            fn, done = job
            try:
                fn()
            except Exception:
                print(f"ERROR: disk job failed:\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
            finally:
                done.set()


def record_path(output_dir: Path, ntp_ns: int, raw: bool,
                width: int, height: int) -> Path:
    """Pure: recording filename (Sec 5) — ``rec_<utc_tag>.ts`` or
    ``rec_<utc_tag>_WxH_I420.yuv``."""
    tag = utc_tag(ntp_ns)
    if raw:
        return output_dir / f"rec_{tag}_{width}x{height}_I420.yuv"
    return output_dir / f"rec_{tag}.ts"


def snapshot_path(output_dir: Path, ntp_ns: int, raw: bool) -> Path:
    """Pure: ``snap_<utc_tag>.png`` or ``snap_<utc_tag>.ppm``."""
    suffix = "ppm" if raw else "png"
    return output_dir / f"snap_{utc_tag(ntp_ns)}.{suffix}"


def append_sidecar_line(jsonl_path: Path, pts: int, ntp_ns: int) -> None:
    """Append one ``{"pts":…, "ntp_ns":…, "utc":"…"}`` line (Sec 5 Disk).
    Called from the rec_sidecar probe's streaming thread; must be fast
    (buffered append)."""
    line = f'{{"pts": {pts}, "ntp_ns": {ntp_ns}, "utc": "{utc_tag(ntp_ns)}"}}\n'
    with open(jsonl_path, "a", encoding="ascii") as handle:
        handle.write(line)


def _write_json_sidecar(path: Path, pts: int, ntp_ns: int) -> None:
    sidecar = path.with_suffix(".json")
    payload = {"pts": pts, "ntp_ns": ntp_ns, "utc": utc_tag(ntp_ns)}
    sidecar.write_text(json.dumps(payload) + "\n", encoding="ascii")


def write_png(frame_rgba, path: Path, pts: int | None = None,
              ntp_ns: int | None = None) -> None:
    """cv2.imwrite lossless PNG (RGBA->BGR convert; ~200-600 ms — disk worker
    thread only, Sec 7). cv2 imported here. When ``pts``/``ntp_ns`` are given
    the Sec 5 ``.json`` sidecar is written next to the PNG."""
    import cv2

    bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    if pts is not None and ntp_ns is not None:
        _write_json_sidecar(path, pts, ntp_ns)


def write_ppm_with_sidecar(frame_rgba, path: Path, pts: int, ntp_ns: int) -> None:
    """Raw snapshot: .ppm (P6 header + RGB pixels) plus the .json sidecar
    with the same stamp fields (Sec 4/5). Disk worker thread only."""
    import numpy as np

    rgb = np.ascontiguousarray(frame_rgba[..., :3])
    height, width = rgb.shape[:2]
    with open(path, "wb") as handle:
        handle.write(b"P6\n%d %d\n255\n" % (width, height))
        handle.write(rgb.tobytes())
    _write_json_sidecar(path, pts, ntp_ns)
