"""Swappable source factory (DESIGN.md Sec 3.1 source bin, Sec 5 loop/EOS).

THE one source-swap point: ``create_source(uri, ...)`` maps a uri scheme to a
variant and returns a ``SourceBin`` — a Gst.Bin with exactly one ghost src pad
of caps ``video/x-raw(memory:NVMM), NV12, WxH`` and an ``is_live`` flag that
decides ``pace``'s sync and ``mux_grab``'s live-source.

File variant = compressed-domain AU replay (validated, Sec 5): a one-shot
extraction pipeline yields the AU list; a feeder thread pushes AUs cyclically
into ``appsrc`` (is-live=true format=time block=true do-timestamp=false
max-bytes=8388608) with self-assigned monotonic pts and dts=NONE. The feeder
is the single writer of ``n_loops`` (the /ds/status loop counter). No seeks,
no SEGMENT_DONE, no event dropping. rtsp variant never loops and has no
feeder. A future IPC variant is a registered-but-unimplemented slot (Sec 11
risk 5).

Pure seams for tests.py (no GStreamer): ``schedule_pts`` and
``compute_loop_span`` (the feeder's pts schedule), ``select_variant``.
Gst/GstPbutils imports are deferred into the functions that build bins.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from config import PipelineConfig

APPSRC_MAX_BYTES = 8 * 1024 * 1024
NUM_EXTRA_SURFACES = 40
DEFAULT_FRAME_DURATION_NS = 33_333_333
EXTRACTION_STALL_S = 30.0

_PARSERS = {"video/x-h265": "h265parse", "video/x-h264": "h264parse"}
_DEPAYS = {"H265": "rtph265depay", "H264": "rtph264depay"}

_registered_variants: dict[str, Callable[..., "SourceBin"]] = {}


@dataclass(frozen=True)
class AccessUnit:
    """One compressed access unit from the extraction pass (Sec 3.1)."""

    data: bytes
    pts: int        # container pts, ns (first AU is the file's own first pts)
    duration: int   # ns (33_333_333 for the 30 fps test clip)


@dataclass(frozen=True)
class ExtractedStream:
    """Result of the one-shot extraction pipeline (runs to EOS, then NULL)."""

    aus: tuple[AccessUnit, ...]
    caps: str          # negotiated byte-stream/au caps, reused on the appsrc
    loop_span: int     # ns; == compute_loop_span(aus)
    width: int
    height: int


def compute_loop_span(aus: Sequence[AccessUnit]) -> int:
    """Pure: ``loop_span = last_pts - first_pts + last.duration`` (Sec 3.1).

    ``aus`` in decode order; pts of the first/last AU in *presentation* pts as
    extracted (= 9.5641 s for the test clip). Raises ValueError on empty input.
    Unit-tested against synthetic AU lists (Sec 10).
    """
    if not aus:
        raise ValueError("empty access-unit list")
    first_pts = min(au.pts for au in aus)
    last = max(aus, key=lambda au: au.pts)
    return last.pts - first_pts + last.duration


def schedule_pts(au_pts: int, n_loops: int, loop_span: int) -> int:
    """Pure feeder pts schedule: ``buf.pts = au.pts + n_loops * loop_span``.

    Sec 5: strictly monotonic and globally unique across wraps for the life of
    the process; this is the key of the timestamp registry and of batch result
    joining. Unit-tested for monotonicity/uniqueness across wraps (Sec 10).
    """
    return au_pts + n_loops * loop_span


def select_variant(uri: str) -> str:
    """Pure uri -> variant name: 'file' | 'rtsp' | registered extras.

    file:// -> 'file'; rtsp:// and rtsps:// -> 'rtsp'; a scheme registered via
    ``register_variant`` -> its name; anything else raises ValueError.
    Unit-tested with bin construction mocked (Sec 10).
    """
    scheme = urlparse(uri).scheme.lower()
    if scheme == "file":
        return "file"
    if scheme in {"rtsp", "rtsps"}:
        return "rtsp"
    if scheme in _registered_variants:
        return scheme
    raise ValueError(f"No source variant registered for uri scheme {scheme!r} ({uri})")


def register_variant(scheme: str, factory: Callable[..., "SourceBin"]) -> None:
    """Extension slot (Sec 11 risk 5): register a future source variant.

    ``factory(uri, config, on_fatal)`` must return a SourceBin honoring the
    Sec 3.1 pool rule (expose a num-extra-surfaces-equivalent knob or copy
    into an owned pool before the tee). Called at import/startup time only.
    """
    _registered_variants[scheme.lower()] = factory


def _file_uri_to_path(uri: str) -> Path:
    """file:// uri -> filesystem path; a netloc-relative uri
    (``file://streams/x.mp4``) resolves against the repo root. The RFC 8089
    local-host form ``file://localhost/abs/x.mp4`` means the filesystem root."""
    parsed = urlparse(uri)
    netloc = "" if parsed.netloc.lower() == "localhost" else parsed.netloc
    raw = unquote(netloc + parsed.path)
    path = Path(raw)
    if not path.is_absolute():
        from deepstream_yolo.paths import PROJECT_DIR

        path = PROJECT_DIR / raw
    return path


def _make(factory: str, name: str):
    from gi.repository import Gst

    elem = Gst.ElementFactory.make(factory, name)
    if elem is None:
        raise RuntimeError(f"Missing GStreamer element: {factory}")
    return elem


class SourceBin:
    """A built source: Gst.Bin + ghost NVMM src pad + lifecycle hooks.

    Attributes (read-only after construction):
      bin        -- the Gst.Bin to add to the live pipeline
      is_live    -- False for file (pace must sync), True for rtsp
      width, height -- discovered source resolution
      loops      -- bool: whether this variant loops (file with loop=True)

    ``n_loops`` is a property: the feeder's wrap counter (0 forever for
    non-file variants) — read by the /ds/status timer, written only by the
    feeder thread (single writer, plain int read is safe).
    """

    is_live: bool = False
    loops: bool = False
    width: int = 0
    height: int = 0

    def __init__(self, uri: str, config: PipelineConfig) -> None:
        self.uri = uri
        self.config = config
        self._n_loops = 0

    @property
    def n_loops(self) -> int:
        """Completed loop count (Sec 3.1). Non-blocking, any thread."""
        return self._n_loops

    def start(self) -> None:
        """Start variant-owned threads (file: the feeder). Called from the
        main thread after the live pipeline reaches PLAYING. No-op for rtsp."""

    def stop(self) -> None:
        """Stop and join variant-owned threads. Called from the main thread
        during shutdown, before the pipeline goes NULL. Blocks until the
        feeder exits (its blocking appsrc push is unblocked by pipeline
        teardown or a stop flag checked between pushes)."""


def extract_access_units(uri: str, max_preload_mb: int) -> ExtractedStream:
    """Run the one-shot extraction pipeline (Sec 3.1) and return the AU list.

    filesrc ! qtdemux ! {h265parse|h264parse} config-interval=-1 !
    byte-stream/au caps ! appsink sync=false — runs to EOS (<2 s for the test
    clip), then NULL. Blocks the calling (main) thread. Raises RuntimeError if
    the accumulated AU bytes exceed ``max_preload_mb`` (Sec 11 risk 6 guard:
    refuse and suggest loop=false) or if extraction fails.
    """
    import time

    from gi.repository import Gst

    path = _file_uri_to_path(uri)
    if not path.is_file():
        raise RuntimeError(f"Source file not found: {path}")

    pipeline = Gst.Pipeline.new("extract")
    filesrc = _make("filesrc", "extract_src")
    filesrc.set_property("location", str(path))
    demux = _make("qtdemux", "extract_demux")
    appsink = _make("appsink", "extract_sink")
    appsink.set_property("sync", False)
    pipeline.add(filesrc)
    pipeline.add(demux)
    pipeline.add(appsink)
    if not filesrc.link(demux):
        raise RuntimeError("Failed to link filesrc to qtdemux")

    link_error: list[str] = []
    linked = threading.Event()

    def on_pad_added(_demux, pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        name = structure.get_name()
        if not name.startswith("video/"):
            return
        if linked.is_set():
            return  # first video track wins; ignore extra tracks
        linked.set()
        parser = _PARSERS.get(name)
        if parser is None:
            link_error.append(f"Unsupported video codec in {path}: {name}")
            return
        parse = _make(parser, "extract_parse")
        parse.set_property("config-interval", -1)
        capsfilter = _make("capsfilter", "extract_caps")
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(f"{name},stream-format=byte-stream,alignment=au"),
        )
        pipeline.add(parse)
        pipeline.add(capsfilter)
        if not (parse.link(capsfilter) and capsfilter.link(appsink)):
            link_error.append("Failed to link extraction parse chain")
            return
        parse.sync_state_with_parent()
        capsfilter.sync_state_with_parent()
        if pad.link(parse.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            link_error.append("Failed to link qtdemux pad to parser")

    demux.connect("pad-added", on_pad_added)

    max_bytes = max_preload_mb * 1024 * 1024
    aus: list[AccessUnit] = []
    caps_str = ""
    width = height = 0
    total = 0
    last_duration = DEFAULT_FRAME_DURATION_NS
    bus = pipeline.get_bus()

    try:
        pipeline.set_state(Gst.State.PLAYING)
        deadline = time.monotonic() + EXTRACTION_STALL_S
        while True:
            sample = appsink.emit("try-pull-sample", 500 * Gst.MSECOND)
            if sample is None:
                if link_error:
                    raise RuntimeError(link_error[0])
                msg = bus.pop_filtered(Gst.MessageType.ERROR)
                if msg is not None:
                    err, _debug = msg.parse_error()
                    raise RuntimeError(f"Extraction failed for {path}: {err.message}")
                if appsink.get_property("eos"):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Extraction stalled (>{EXTRACTION_STALL_S:.0f}s) for {path}")
                continue
            deadline = time.monotonic() + EXTRACTION_STALL_S
            buf = sample.get_buffer()
            if not caps_str:
                caps = sample.get_caps()
                caps_str = caps.to_string()
                structure = caps.get_structure(0)
                ok_w, width = structure.get_int("width")
                ok_h, height = structure.get_int("height")
                if not (ok_w and ok_h):
                    width = height = 0
            if buf.pts == Gst.CLOCK_TIME_NONE:
                raise RuntimeError(f"Extraction produced an AU without pts in {path}")
            if buf.duration != Gst.CLOCK_TIME_NONE:
                last_duration = buf.duration
            data = buf.extract_dup(0, buf.get_size())
            total += len(data)
            if total > max_bytes:
                raise RuntimeError(
                    f"AU preload for {path} exceeds source.max_preload_mb={max_preload_mb}"
                    " (Sec 11 risk 6): refusing to hold the clip in RAM;"
                    " use a shorter clip, raise the limit, or set source.loop=false"
                )
            aus.append(AccessUnit(data=data, pts=buf.pts, duration=last_duration))
    finally:
        pipeline.set_state(Gst.State.NULL)

    if not aus:
        raise RuntimeError(f"Extraction produced no access units for {path}")
    return ExtractedStream(
        aus=tuple(aus),
        caps=caps_str,
        loop_span=compute_loop_span(aus),
        width=width,
        height=height,
    )


class FileSource(SourceBin):
    """File variant: appsrc + parse + nvv4l2decoder(num-extra-surfaces=40).

    The feeder thread pushes AUs cyclically in decode order with
    ``buf.pts = schedule_pts(au.pts, n_loops, loop_span)`` and dts=NONE,
    incrementing ``n_loops`` at each wrap. Backpressure: appsrc block=true
    max-bytes=8MiB simply blocks the feeder (Sec 3.1 pacing). With
    loop=False the feeder pushes one pass then calls
    ``appsrc.end_of_stream()`` and exits — the EOS drains every branch and
    surfaces on the bus (Sec 5 EOS behavior); ds_node handles the ended
    transition.
    """

    is_live = False

    def __init__(self, uri: str, config: PipelineConfig) -> None:
        from gi.repository import Gst

        super().__init__(uri, config)
        self.loops = config.source_loop
        self._extracted = extract_access_units(uri, config.source_max_preload_mb)
        self.width = self._extracted.width
        self.height = self._extracted.height
        if not (self.width and self.height):
            from deepstream_yolo.model_cache import discover_size

            self.width, self.height = discover_size(_file_uri_to_path(uri).resolve().as_uri())

        self.bin = Gst.Bin.new("source")
        self._appsrc = _make("appsrc", "src")
        self._appsrc.set_property("is-live", True)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("block", True)
        self._appsrc.set_property("do-timestamp", False)
        self._appsrc.set_property("max-bytes", APPSRC_MAX_BYTES)
        self._appsrc.set_property("caps", Gst.Caps.from_string(self._extracted.caps))
        codec = self._extracted.caps.split(",", 1)[0].strip()
        parser = _PARSERS.get(codec)
        if parser is None:
            raise RuntimeError(f"Unsupported extracted caps for appsrc replay: {codec}")
        parse = _make(parser, "parse")
        dec = _make("nvv4l2decoder", "dec")
        dec.set_property("num-extra-surfaces", NUM_EXTRA_SURFACES)
        for elem in (self._appsrc, parse, dec):
            self.bin.add(elem)
        if not (self._appsrc.link(parse) and parse.link(dec)):
            raise RuntimeError("Failed to link appsrc ! parse ! nvv4l2decoder")
        ghost = Gst.GhostPad.new("src", dec.get_static_pad("src"))
        self.bin.add_pad(ghost)

        self._stop = threading.Event()
        self._feeder = threading.Thread(target=self._feeder_run, name="feeder", daemon=True)

    def start(self) -> None:
        self._feeder.start()

    def stop(self) -> None:
        from gi.repository import Gst

        self._stop.set()
        if not self._feeder.is_alive():
            return
        self._feeder.join(timeout=1.0)
        if self._feeder.is_alive():
            # Unblock a push stuck on the full appsrc queue: flushing appsrc
            # makes push-buffer return FLUSHING (the pipeline is being torn
            # down right after this anyway).
            self._appsrc.set_state(Gst.State.NULL)
            self._feeder.join()

    def _feeder_run(self) -> None:
        """Feeder thread body (thread name 'feeder', Sec 2). Owns n_loops."""
        from gi.repository import Gst

        aus = self._extracted.aus
        span = self._extracted.loop_span
        while not self._stop.is_set():
            for au in aus:
                if self._stop.is_set():
                    return
                buf = Gst.Buffer.new_wrapped(au.data)
                buf.pts = schedule_pts(au.pts, self._n_loops, span)
                buf.dts = Gst.CLOCK_TIME_NONE
                buf.duration = au.duration
                if self._appsrc.emit("push-buffer", buf) != Gst.FlowReturn.OK:
                    return
            if not self.loops:
                self._appsrc.emit("end-of-stream")
                return
            self._n_loops += 1


class RtspSource(SourceBin):
    """rtspsrc ntp-sync=true drop-on-latency=true
    (+ add-reference-timestamp-meta=true when GStreamer >= 1.22) ! depay !
    parse ! nvv4l2decoder (num-extra-surfaces=40). is_live=True; never
    loops; no feeder."""

    is_live = True
    loops = False

    def __init__(self, uri: str, config: PipelineConfig) -> None:
        from gi.repository import Gst

        from deepstream_yolo.model_cache import discover_size

        super().__init__(uri, config)
        self.width, self.height = discover_size(uri)

        self.bin = Gst.Bin.new("source")
        rtspsrc = _make("rtspsrc", "src")
        rtspsrc.set_property("location", uri)
        rtspsrc.set_property("ntp-sync", True)
        if rtspsrc.find_property("add-reference-timestamp-meta") is not None:
            rtspsrc.set_property("add-reference-timestamp-meta", True)
        else:
            # Sec 3.1 mandates the property but it only exists from GStreamer
            # 1.22; DS 7.1 ships 1.20.3. Sec 5 sanctions the fallback: the
            # ingest arrival-time registry supplies timestamps instead.
            print(
                "source: rtspsrc lacks add-reference-timestamp-meta"
                " (GStreamer < 1.22); timestamps fall back to the ingest registry",
                flush=True,
            )
        rtspsrc.set_property("drop-on-latency", True)
        dec = _make("nvv4l2decoder", "dec")
        dec.set_property("num-extra-surfaces", NUM_EXTRA_SURFACES)
        self.bin.add(rtspsrc)
        self.bin.add(dec)

        claim = threading.Lock()

        def on_pad_added(_src, pad) -> None:
            caps = pad.get_current_caps() or pad.query_caps(None)
            if caps.get_size() == 0:
                return
            structure = caps.get_structure(0)
            if structure.get_name() != "application/x-rtp":
                return
            if structure.get_string("media") not in (None, "video"):
                return
            encoding = (structure.get_string("encoding-name") or "").upper()
            depay_factory = _DEPAYS.get(encoding)
            if depay_factory is None:
                print(f"source: ignoring rtsp stream with encoding {encoding!r}", flush=True)
                return
            if not claim.acquire(blocking=False):
                return  # another video stream already claimed the decoder
            depay = _make(depay_factory, "depay")
            parse = _make("h265parse" if encoding == "H265" else "h264parse", "parse")
            self.bin.add(depay)
            self.bin.add(parse)
            if not (depay.link(parse) and parse.link(dec)):
                print("source: failed to link rtsp depay chain", flush=True)
                return
            depay.sync_state_with_parent()
            parse.sync_state_with_parent()
            if pad.link(depay.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                print("source: failed to link rtspsrc pad to depayloader", flush=True)

        rtspsrc.connect("pad-added", on_pad_added)
        ghost = Gst.GhostPad.new("src", dec.get_static_pad("src"))
        self.bin.add_pad(ghost)


def create_source(config: PipelineConfig) -> SourceBin:
    """Factory: build the SourceBin for ``config.source_uri`` (Sec 9 wiring).

    Uses ``select_variant`` then the matching class; discovers W x H via
    ``deepstream_yolo.model_cache.discover_size``. Called once from the main
    thread at startup, before the live pipeline is built around the bin.
    """
    uri = config.source_uri
    variant = select_variant(uri)
    if variant == "file":
        return FileSource(uri, config)
    if variant == "rtsp":
        return RtspSource(uri, config)
    return _registered_variants[variant](uri, config, None)
