"""Live pipeline builder (DESIGN.md Sec 3.1/3.2): trunk + Branch G + Branch P.

Builds ``Gst.Pipeline "live"``: source bin -> identity pace
(sync = not is_live) -> tee t_ingest (every branch on its own requested
src_%u pad behind its own leaky queue), with the grabber branch
(q_grab leaky=2 max-size-buffers=4 -> mux_grab batch-size=1
batched-push-timeout=40000 attach-sys-ts=false live-source={is_live}
sync-inputs=false -> conv_grab nvbuf-memory-type=3 -> caps_grab RGBA WxH ->
fakesink sync=false async=false enable-last-sample=false) and the preview
branch (q_preview leaky=2 max-size-buffers=1 -> conv_preview -> caps I420
preview.width x preview.height -> nvjpegenc quality=preview.quality ->
appsink emit-signals=true sync=false max-buffers=1 drop=true). Every element
name and property value in Sec 3.1 is load-bearing (empirically validated) —
implement them verbatim. Branch R is NOT built here; disk.py attaches it
dynamically to ``t_ingest`` via ``request_tee_pad`` and releases the pad
itself on the detach and source-EOS paths.

Gst imports are performed inside the builder functions so tests.py can import
sibling modules without GStreamer present.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from config import PipelineConfig
from source import SourceBin
from timestamps import TimestampRegistry, resolve

MUX_GRAB_PUSH_TIMEOUT_US = 40_000
NVBUF_MEM_CUDA_UNIFIED = 3
QUEUE_LEAKY_DOWNSTREAM = 2


@dataclass
class LiveParts:
    """Handles other modules wire probes/handlers to (style of
    deepstream_yolo.pipeline.PipelineParts — only what callers reach for).

    pipeline      -- the Gst.Pipeline "live"
    source        -- the SourceBin (start()/stop(), n_loops)
    pace          -- identity name=pace (ingest_stamp probe on its src pad)
    t_ingest      -- the tee; disk.py requests/releases recorder pads on it
    caps_grab     -- grab-probe attach point (its src pad, Sec 3.1)
    sink_preview  -- preview appsink; ros_io connects new-sample
    """

    pipeline: "object"
    source: SourceBin
    pace: "object"
    t_ingest: "object"
    caps_grab: "object"
    sink_preview: "object"


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


def request_tee_pad(t_ingest):
    """Request a new ``src_%u`` pad on ``t_ingest`` (recorder attach, Sec 7).

    Returns the Gst.Pad. Called from the grp_record service thread by
    disk.py; the tee hands out request pads safely while PLAYING.
    """
    pad = t_ingest.request_pad_simple("src_%u")
    if pad is None:
        raise RuntimeError("t_ingest refused a src_%u request pad")
    return pad


def build_live_pipeline(config: PipelineConfig, source: SourceBin) -> LiveParts:
    """Assemble the live graph around ``source`` with Sec 3.1's exact settings.

    Called once from the main thread at startup, after Gst.init. Creates
    elements, links trunk and both static branches (each on its own requested
    tee pad), and returns the handles. Does not set state (ds_node owns the
    state machine) and does not install probes.
    """
    from gi.repository import Gst

    width, height = source.width, source.height
    pipeline = Gst.Pipeline.new("live")

    pace = _make("identity", "pace")
    pace.set_property("sync", not source.is_live)

    t_ingest = _make("tee", "t_ingest")
    t_ingest.set_property("allow-not-linked", False)

    pipeline.add(source.bin)
    pipeline.add(pace)
    pipeline.add(t_ingest)
    _link(source.bin, pace)
    _link(pace, t_ingest)

    caps_grab = _build_grab_branch(pipeline, t_ingest, source, width, height)
    sink_preview = _build_preview_branch(pipeline, t_ingest, config)

    return LiveParts(
        pipeline=pipeline,
        source=source,
        pace=pace,
        t_ingest=t_ingest,
        caps_grab=caps_grab,
        sink_preview=sink_preview,
    )


def _build_grab_branch(pipeline, t_ingest, source: SourceBin,
                       width: int, height: int):
    """Branch G (Sec 3.1): q_grab -> mux_grab -> conv_grab -> caps_grab ->
    sink_grab. Returns caps_grab (the grab-probe attach point)."""
    from gi.repository import Gst

    q_grab = _make("queue", "q_grab")
    q_grab.set_property("leaky", QUEUE_LEAKY_DOWNSTREAM)
    q_grab.set_property("max-size-buffers", 4)
    q_grab.set_property("max-size-bytes", 0)
    q_grab.set_property("max-size-time", 0)

    mux_grab = _make("nvstreammux", "mux_grab")
    mux_grab.set_property("batch-size", 1)
    mux_grab.set_property("width", width)
    mux_grab.set_property("height", height)
    mux_grab.set_property("batched-push-timeout", MUX_GRAB_PUSH_TIMEOUT_US)
    mux_grab.set_property("attach-sys-ts", False)
    mux_grab.set_property("live-source", source.is_live)
    mux_grab.set_property("sync-inputs", False)

    conv_grab = _make("nvvideoconvert", "conv_grab")
    conv_grab.set_property("nvbuf-memory-type", NVBUF_MEM_CUDA_UNIFIED)

    caps_grab = _make("capsfilter", "caps_grab")
    caps_grab.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM),format=RGBA,width={width},height={height}"
        ),
    )

    sink_grab = _make("fakesink", "sink_grab")
    sink_grab.set_property("sync", False)
    sink_grab.set_property("async", False)
    sink_grab.set_property("enable-last-sample", False)

    for elem in (q_grab, mux_grab, conv_grab, caps_grab, sink_grab):
        pipeline.add(elem)

    tee_pad = request_tee_pad(t_ingest)
    if tee_pad.link(q_grab.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link t_ingest to q_grab")
    mux_sink = mux_grab.request_pad_simple("sink_0")
    if mux_sink is None:
        raise RuntimeError("mux_grab refused a sink_0 request pad")
    if q_grab.get_static_pad("src").link(mux_sink) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link q_grab to mux_grab")
    _link(mux_grab, conv_grab)
    _link(conv_grab, caps_grab)
    _link(caps_grab, sink_grab)
    return caps_grab


def _build_preview_branch(pipeline, t_ingest, config: PipelineConfig):
    """Branch P (Sec 3.1): q_preview -> conv_preview -> caps_preview ->
    enc_preview -> sink_preview. Returns the appsink."""
    from gi.repository import Gst

    q_preview = _make("queue", "q_preview")
    q_preview.set_property("leaky", QUEUE_LEAKY_DOWNSTREAM)
    q_preview.set_property("max-size-buffers", 1)

    conv_preview = _make("nvvideoconvert", "conv_preview")

    caps_preview = _make("capsfilter", "caps_preview")
    caps_preview.set_property(
        "caps",
        Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=I420,"
            f"width={config.preview_width},height={config.preview_height}"
        ),
    )

    enc_preview = _make("nvjpegenc", "enc_preview")
    enc_preview.set_property("quality", config.preview_quality)

    sink_preview = _make("appsink", "sink_preview")
    sink_preview.set_property("emit-signals", True)
    sink_preview.set_property("sync", False)
    sink_preview.set_property("max-buffers", 1)
    sink_preview.set_property("drop", True)

    for elem in (q_preview, conv_preview, caps_preview, enc_preview, sink_preview):
        pipeline.add(elem)

    tee_pad = request_tee_pad(t_ingest)
    if tee_pad.link(q_preview.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link t_ingest to q_preview")
    _link(q_preview, conv_preview)
    _link(conv_preview, caps_preview)
    _link(caps_preview, enc_preview)
    _link(enc_preview, sink_preview)
    return sink_preview


def install_ingest_stamp_probe(pace, registry: TimestampRegistry) -> int:
    """Install the ingest_stamp buffer probe on ``pace``'s SRC pad (Sec 5).

    The probe body is exactly ``registry.stamp(buffer.pts)`` and returns OK;
    it runs on the trunk streaming thread at paced delivery time. Returns the
    probe id. Called from the main thread during wiring.
    """
    from gi.repository import Gst

    def ingest_stamp(_pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            registry.stamp(buffer.pts)
        return Gst.PadProbeReturn.OK

    return pace.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, ingest_stamp)


def connect_preview(sink_preview, publish: Callable[[bytes, int], None],
                    registry: TimestampRegistry) -> None:
    """Connect the preview appsink's new-sample handler (Sec 3.1 Branch P).

    The handler (preview branch streaming thread) pulls the sample, maps the
    JPEG bytes, resolves the stamp from ``registry`` by buffer pts, and calls
    ``publish(jpeg_bytes, ntp_ns)`` — ros_io supplies ``publish`` (rclpy
    publishers are thread-safe, Sec 2); frames with no resolvable stamp are
    skipped. Returns Gst.FlowReturn.OK from the handler. Called from the main
    thread during wiring.
    """
    from gi.repository import Gst

    def on_new_sample(sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        ntp_ns = resolve(registry, buffer.pts)
        if ntp_ns is None:
            return Gst.FlowReturn.OK
        ok, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            jpeg = bytes(mapinfo.data)
        finally:
            buffer.unmap(mapinfo)
        publish(jpeg, ntp_ns)
        return Gst.FlowReturn.OK

    sink_preview.connect("new-sample", on_new_sample)
