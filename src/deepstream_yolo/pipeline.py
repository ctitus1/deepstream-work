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
    detect_tee: Gst.Element | None
    detect_appsink: Gst.Element | None
    assessment_queue: Gst.Element | None
    sgie: Gst.Element | None
    assess_tee: Gst.Element | None
    assess_appsink: Gst.Element | None
    caps: Gst.Element
    osd: Gst.Element
    display_queue: Gst.Element | None
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


def link_tee_to_queue(tee, queue) -> None:
    src = tee.request_pad_simple("src_%u")
    sink = queue.get_static_pad("sink")
    if src is None or sink is None:
        raise RuntimeError(f"Failed to request tee pad for {tee.get_name()}")
    result = src.link(sink)
    if result != Gst.PadLinkReturn.OK:
        raise RuntimeError(f"Failed to link {tee.get_name()} to {queue.get_name()}: {result}")


def compressed_branch(
    pipeline,
    tee,
    name: str,
    width: int,
    height: int,
    jpeg_quality: int,
):
    queue = element("queue", f"{name}-queue")
    convert_encode = element("nvvideoconvert", f"{name}-encode-convert")
    caps_encode = element("capsfilter", f"{name}-encode-caps")
    encoder = element("nvjpegenc", f"{name}-jpeg")
    appsink = element("appsink", f"{name}-appsink")

    configure_latest_queue(queue)
    caps_encode.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), format=I420, width={width}, height={height}"
        ),
    )
    set_property_if_present(encoder, "quality", int(jpeg_quality))
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)
    appsink.set_property("max-buffers", 1)
    set_property_if_present(appsink, "drop", True)

    elements = [queue, convert_encode, caps_encode, encoder, appsink]
    for elem in elements:
        pipeline.add(elem)

    link_tee_to_queue(tee, queue)
    queue.link(convert_encode)
    convert_encode.link(caps_encode)
    caps_encode.link(encoder)
    encoder.link(appsink)
    return appsink


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
    display: bool = True,
    detect_output_size: tuple[int, int] | None = None,
    assess_output_size: tuple[int, int] | None = None,
    jpeg_quality: int = 85,
) -> PipelineParts:
    pipeline = Gst.Pipeline.new("yolo-parser")
    if assess_output_size and not assessment_config:
        raise ValueError("Assessment output requires an assessment config")

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
    detect_tee = element("tee", "detect-tee") if detect_output_size else None
    assessment_queue = element("queue", "assessment-queue") if assessment_config else None
    sgie = element("nvinfer", "assessment") if assessment_config else None
    assess_tee = element("tee", "assess-tee") if assess_output_size else None
    convert = element("nvvideoconvert", "convert")
    caps = element("capsfilter", "caps")
    osd = element("nvdsosd", "osd")
    display_queue = element("queue", "display-queue") if stream.is_rtsp else None
    sink = element("nveglglessink" if display else "fakesink", "sink")

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
    sink.set_property("sync", bool(stream.is_rtsp and display))
    set_property_if_present(sink, "qos", bool(stream.is_rtsp and display))
    configure_latest_queue(queue)
    if assessment_queue:
        configure_latest_queue(assessment_queue)
    if display_queue:
        configure_latest_queue(display_queue)

    elements = source_elements + [h265_parser, h264_parser, decoder, queue, streammux, pgie]
    if detect_tee:
        elements.append(detect_tee)
    if assessment_queue and sgie:
        elements.extend((assessment_queue, sgie))
    if assess_tee:
        elements.append(assess_tee)
    elements.extend((convert, caps, osd))
    if display_queue:
        elements.append(display_queue)
    elements.append(sink)

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
    detect_appsink = None
    assess_appsink = None
    if detect_tee and detect_output_size:
        pgie.link(detect_tee)
        detect_appsink = compressed_branch(
            pipeline,
            detect_tee,
            "detect-output",
            detect_output_size[0],
            detect_output_size[1],
            jpeg_quality,
        )
        if assessment_queue:
            link_tee_to_queue(detect_tee, assessment_queue)
        else:
            link_tee_to_queue(detect_tee, convert)
    elif assessment_queue:
        pgie.link(assessment_queue)
    else:
        pgie.link(convert)

    if assessment_queue and sgie:
        assessment_queue.link(sgie)
        if assess_tee and assess_output_size:
            sgie.link(assess_tee)
            assess_appsink = compressed_branch(
                pipeline,
                assess_tee,
                "assess-output",
                assess_output_size[0],
                assess_output_size[1],
                jpeg_quality,
            )
            link_tee_to_queue(assess_tee, convert)
        else:
            sgie.link(convert)

    convert.link(caps)
    caps.link(osd)
    if display_queue:
        osd.link(display_queue)
        display_queue.link(sink)
    else:
        osd.link(sink)

    return PipelineParts(
        pipeline=pipeline,
        streammux=streammux,
        pgie=pgie,
        detect_tee=detect_tee,
        detect_appsink=detect_appsink,
        assessment_queue=assessment_queue,
        sgie=sgie,
        assess_tee=assess_tee,
        assess_appsink=assess_appsink,
        caps=caps,
        osd=osd,
        display_queue=display_queue,
        sink=sink,
    )
