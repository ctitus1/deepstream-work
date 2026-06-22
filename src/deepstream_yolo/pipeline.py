from __future__ import annotations

import sys
from dataclasses import dataclass

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


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


def on_pad_added(_demux, pad, parsers):
    caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()

    if "video/x-h265" in caps:
        pad.link(parsers["h265"].get_static_pad("sink"))
    elif "video/x-h264" in caps:
        pad.link(parsers["h264"].get_static_pad("sink"))


def on_message(_bus, msg, loop):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"ERROR: {err}\nDEBUG: {dbg}", file=sys.stderr)
        loop.quit()
    elif msg.type == Gst.MessageType.EOS:
        loop.quit()
    return True


def build_pipeline(stream, src_w: int, src_h: int, config, assessment_config=None) -> PipelineParts:
    pipeline = Gst.Pipeline.new("yolo-parser")

    source = element("filesrc", "source")
    demux = element("qtdemux", "demux")
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

    source.set_property("location", str(stream))
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", src_w)
    streammux.set_property("height", src_h)
    streammux.set_property("batched-push-timeout", 40000)
    if streammux.find_property("attach-sys-ts"):
        streammux.set_property("attach-sys-ts", False)
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

    elements = [
        source,
        demux,
        h265_parser,
        h264_parser,
        decoder,
        queue,
        streammux,
        pgie,
    ]
    if assessment_queue and sgie:
        elements.extend((assessment_queue, sgie))
    elements.extend((convert, caps, osd, sink))

    for elem in elements:
        pipeline.add(elem)

    source.link(demux)
    demux.connect("pad-added", on_pad_added, {"h265": h265_parser, "h264": h264_parser})
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
