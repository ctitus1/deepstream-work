from __future__ import annotations

import sys
from dataclasses import dataclass

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from .stream_source import StreamSource


@dataclass
class PipelineParts:
    pipeline: Gst.Pipeline
    streammux: Gst.Element
    pgie: Gst.Element
    assessment_queue: Gst.Element | None
    sgie: Gst.Element | None
    caps: Gst.Element
    osd: Gst.Element
    sink: Gst.Element


def element(factory: str, name: str):
    elem = Gst.ElementFactory.make(factory, name)
    if elem is None:
        raise RuntimeError(f"Missing GStreamer element: {factory}")
    return elem


def configure_latest_queue(queue) -> None:
    queue.set_property("max-size-buffers", 1)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    queue.set_property("leaky", 2)
    set_property_if_present(queue, "flush-on-eos", True)


def set_property_if_present(elem, name: str, value) -> None:
    if elem.find_property(name):
        elem.set_property(name, value)


def link_dynamic_pad(pad, sink) -> None:
    if sink.is_linked():
        return
    pad.link(sink)


def on_file_pad_added(_demux, pad, parsers):
    caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()

    if "video/x-h265" in caps:
        link_dynamic_pad(pad, parsers["h265"].get_static_pad("sink"))
    elif "video/x-h264" in caps:
        link_dynamic_pad(pad, parsers["h264"].get_static_pad("sink"))


def on_rtsp_pad_added(_source, pad, depayloaders):
    caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()
    caps_lower = caps.lower()

    if "encoding-name=(string)h265" in caps_lower or "encoding-name=h265" in caps_lower:
        link_dynamic_pad(pad, depayloaders["h265"].get_static_pad("sink"))
    elif "encoding-name=(string)h264" in caps_lower or "encoding-name=h264" in caps_lower:
        link_dynamic_pad(pad, depayloaders["h264"].get_static_pad("sink"))


def on_message(_bus, msg, loop):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"ERROR: {err}\nDEBUG: {dbg}", file=sys.stderr)
        loop.quit()
    elif msg.type == Gst.MessageType.EOS:
        loop.quit()
    return True


def build_pipeline(
    stream: StreamSource,
    src_w: int,
    src_h: int,
    config,
    assessment_config=None,
    rtsp_latency_ms: int = 0,
) -> PipelineParts:
    pipeline = Gst.Pipeline.new("yolo-parser")

    if stream.is_rtsp:
        source = element("rtspsrc", "source")
        h265_depay = element("rtph265depay", "h265-depay")
        h264_depay = element("rtph264depay", "h264-depay")
        source_elements = [source, h265_depay, h264_depay]
    else:
        source = element("filesrc", "source")
        demux = element("qtdemux", "demux")
        source_elements = [source, demux]

    h265_parser = element("h265parse", "h265-parser")
    h264_parser = element("h264parse", "h264-parser")
    decoder = element("nvv4l2decoder", "decoder")
    queue = element("queue", "queue")
    streammux = element("nvstreammux", "streammux")
    pgie = element("nvinfer", "pgie")
    assessment_queue = element("queue", "assessment-queue") if assessment_config else None
    sgie = element("nvinfer", "assessment") if assessment_config else None
    convert = element("nvvideoconvert", "convert")
    caps = element("capsfilter", "caps")
    osd = element("nvdsosd", "osd")
    sink = element("nveglglessink", "sink")

    if stream.is_rtsp:
        source.set_property("location", stream.uri)
        set_property_if_present(source, "latency", max(0, int(rtsp_latency_ms)))
        set_property_if_present(source, "drop-on-latency", True)
        set_property_if_present(source, "ntp-sync", True)
        set_property_if_present(source, "add-reference-timestamp-meta", True)
    else:
        if stream.path is None:
            raise ValueError(f"Unsupported stream URI for this pipeline: {stream.uri}")
        source.set_property("location", str(stream.path))

    streammux.set_property("batch-size", 1)
    streammux.set_property("width", src_w)
    streammux.set_property("height", src_h)
    streammux.set_property("batched-push-timeout", 0 if stream.is_rtsp else 40000)
    set_property_if_present(streammux, "attach-sys-ts", False)
    set_property_if_present(streammux, "live-source", bool(stream.is_rtsp))
    set_property_if_present(streammux, "sync-inputs", False)
    set_property_if_present(streammux, "cache-buffer", False)
    set_property_if_present(streammux, "cache-buffer-timeout", 0)
    set_property_if_present(decoder, "disable-dpb", bool(stream.is_rtsp))
    set_property_if_present(decoder, "low-latency-mode", bool(stream.is_rtsp))
    set_property_if_present(decoder, "qos", True)
    pgie.set_property("config-file-path", str(config))
    if sgie:
        sgie.set_property("config-file-path", str(assessment_config))
        sgie.set_property("process-mode", 2)
        sgie.set_property("output-tensor-meta", True)
    caps.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), format=RGBA, width={src_w}, height={src_h}"
        ),
    )
    osd.set_property("process-mode", 1)
    osd.set_property("display-bbox", 1)
    osd.set_property("display-text", 1)
    sink.set_property("sync", False)
    sink.set_property("qos", False)
    configure_latest_queue(queue)
    if assessment_queue:
        configure_latest_queue(assessment_queue)

    elements = source_elements + [h265_parser, h264_parser, decoder, queue, streammux, pgie]
    if assessment_queue and sgie:
        elements.extend((assessment_queue, sgie))
    elements.extend((convert, caps, osd, sink))

    for elem in elements:
        pipeline.add(elem)

    if stream.is_rtsp:
        source.connect("pad-added", on_rtsp_pad_added, {"h265": h265_depay, "h264": h264_depay})
        h265_depay.link(h265_parser)
        h264_depay.link(h264_parser)
    else:
        source.link(demux)
        demux.connect(
            "pad-added",
            on_file_pad_added,
            {"h265": h265_parser, "h264": h264_parser},
        )

    h265_parser.link(decoder)
    h264_parser.link(decoder)
    decoder.link(queue)
    queue.get_static_pad("src").link(streammux.request_pad_simple("sink_0"))
    streammux.link(pgie)
    if assessment_queue and sgie:
        pgie.link(assessment_queue)
        assessment_queue.link(sgie)
        sgie.link(convert)
    else:
        pgie.link(convert)
    convert.link(caps)
    caps.link(osd)
    osd.link(sink)

    return PipelineParts(
        pipeline=pipeline,
        streammux=streammux,
        pgie=pgie,
        assessment_queue=assessment_queue,
        sgie=sgie,
        caps=caps,
        osd=osd,
        sink=sink,
    )
