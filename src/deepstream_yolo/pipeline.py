"""Shared DeepStream pipeline builder.

Both ``parser_app.py`` and ``ros_source.py`` call ``build_pipeline()``. This
module owns the GStreamer element graph: input decode, ``nvstreammux``, PGIE
YOLO inference, optional SGIE assessment, optional compressed appsink branches,
OSD conversion, an optional mp4 recording branch, and display/fake sink output.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from .recording import select_encoder
from .stream_source import StreamSource


@dataclass
class PipelineParts:
    """Elements callers attach probes or signal handlers to.

    Tees and queues are deliberately absent: nothing outside this module reaches
    for them, and ``pipeline.add()`` already holds the C reference that keeps
    them alive. The three appsinks ARE needed here — ros_source.py connects
    ``new-sample`` to each of them, and they are the only frame source for the
    ROS topics.
    """

    pipeline: Gst.Pipeline
    streammux: Gst.Element
    raw_appsink: Gst.Element | None
    pgie: Gst.Element
    detect_appsink: Gst.Element | None
    sgie: Gst.Element | None
    assess_appsink: Gst.Element | None
    caps: Gst.Element
    osd: Gst.Element
    sink: Gst.Element
    record_sink: Gst.Element | None = None


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


def configure_record_queue(queue) -> None:
    """Buffer the record branch so encoder backpressure never stalls display.

    Unlike ``configure_latest_queue`` this keeps a real backlog (recordings want
    every frame) and must not flush on EOS or mp4mux loses the tail.
    """
    queue.set_property("max-size-buffers", 30)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    queue.set_property("leaky", 2)
    set_property_if_present(queue, "flush-on-eos", False)


def set_property_if_present(elem, name: str, value) -> None:
    if elem.find_property(name):
        elem.set_property(name, value)


def link_dynamic_pad(pad, sink) -> None:
    if sink.is_linked():
        return
    # Runs inside a pad-added callback, where raising would unwind into
    # GStreamer's C code, so report instead of throwing. Discarding this result
    # left the branch unlinked and the pipeline silently producing no frames.
    result = pad.link(sink)
    if result != Gst.PadLinkReturn.OK:
        owner = sink.get_parent_element()
        target = owner.get_name() if owner is not None else sink.get_name()
        print(
            f"ERROR: failed to link {pad.get_name()} to {target}: {result}",
            file=sys.stderr,
            flush=True,
        )


def link_tee_to_queue(tee, queue) -> None:
    src = tee.request_pad_simple("src_%u")
    sink = queue.get_static_pad("sink")
    if src is None or sink is None:
        raise RuntimeError(f"Failed to request tee pad for {tee.get_name()}")
    result = src.link(sink)
    if result != Gst.PadLinkReturn.OK:
        raise RuntimeError(f"Failed to link {tee.get_name()} to {queue.get_name()}: {result}")


def tee_queue_branch(pipeline, tee, name: str, sink) -> Gst.Element:
    queue = element("queue", f"{name}-queue")
    configure_latest_queue(queue)
    pipeline.add(queue)
    link_tee_to_queue(tee, queue)
    queue.link(sink)
    return queue


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


def recording_branch(pipeline, tee, record_path) -> Gst.Element:
    """Encode the burned-in OSD output straight to mp4, staying in NVMM when possible."""
    choice = select_encoder()
    queue = element("queue", "record-queue")
    convert = element("nvvideoconvert", "record-convert")
    caps = element("capsfilter", "record-caps")
    encoder = element(choice.factory, "record-encoder")
    parser = element("h264parse", "record-parser")
    muxer = element("mp4mux", "record-mux")
    sink = element("filesink", "record-sink")

    configure_record_queue(queue)
    caps.set_property("caps", Gst.Caps.from_string(choice.caps))
    for name, value in choice.properties.items():
        set_property_if_present(encoder, name, value)
    set_property_if_present(muxer, "faststart", True)
    sink.set_property("location", str(record_path))
    sink.set_property("sync", False)
    set_property_if_present(sink, "async", False)

    elements = [queue, convert, caps, encoder, parser, muxer, sink]
    for elem in elements:
        pipeline.add(elem)

    link_tee_to_queue(tee, queue)
    for upstream, downstream in zip(elements, elements[1:]):
        if not upstream.link(downstream):
            raise RuntimeError(
                f"Failed to link {upstream.get_name()} to {downstream.get_name()}"
            )
    return sink


def link_parser_to_decoder(parser, decoder) -> None:
    """Give the decoder's single static sink pad to the codec actually in use.

    ``nvv4l2decoder`` has one sink pad, so only one of the h264/h265 parser
    branches can hold it. Linking both up front leaves whichever one loses the
    race silently unlinked, so this runs from the pad-added callbacks instead,
    once the source has told us which codec it carries.
    """
    sink = decoder.get_static_pad("sink")
    if sink is None:
        raise RuntimeError(f"No sink pad on {decoder.get_name()}")

    peer = sink.get_peer()
    if peer is not None:
        holder = peer.get_parent_element()
        if holder is not None and holder.get_name() == parser.get_name():
            return
        held_by = holder.get_name() if holder is not None else peer.get_name()
        raise RuntimeError(
            f"{decoder.get_name()} sink already linked to {held_by}; "
            f"cannot also link {parser.get_name()}"
        )

    if not parser.link(decoder):
        raise RuntimeError(f"Failed to link {parser.get_name()} to {decoder.get_name()}")


def on_file_pad_added(_demux, pad, parsers, decoder):
    caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()

    if "video/x-h265" in caps:
        codec = "h265"
    elif "video/x-h264" in caps:
        codec = "h264"
    else:
        return

    parser = parsers[codec]
    link_parser_to_decoder(parser, decoder)
    link_dynamic_pad(pad, parser.get_static_pad("sink"))


def on_rtsp_pad_added(_source, pad, depayloaders, parsers=None, decoder=None):
    """Link the depayloader for the negotiated codec, and its parser if given.

    ``parsers``/``decoder`` are optional because callers that terminate each
    codec branch in its own sink have no shared decoder pad to contend for.
    """
    caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()
    caps_lower = caps.lower()

    if "encoding-name=(string)h265" in caps_lower:
        codec = "h265"
    elif "encoding-name=(string)h264" in caps_lower:
        codec = "h264"
    else:
        return

    if parsers is not None and decoder is not None:
        link_parser_to_decoder(parsers[codec], decoder)
    link_dynamic_pad(pad, depayloaders[codec].get_static_pad("sink"))


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
    raw_output_size: tuple[int, int] | None = None,
    detect_output_size: tuple[int, int] | None = None,
    assess_output_size: tuple[int, int] | None = None,
    jpeg_quality: int = 85,
    record_path=None,
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
    raw_tee = element("tee", "raw-input-tee") if raw_output_size else None
    pgie = element("nvinfer", "pgie")
    detect_tee = element("tee", "detect-tee") if detect_output_size else None
    assessment_queue = element("queue", "assessment-queue") if assessment_config else None
    sgie = element("nvinfer", "assessment") if assessment_config else None
    assess_tee = element("tee", "assess-tee") if assess_output_size else None
    convert = element("nvvideoconvert", "convert")
    caps = element("capsfilter", "caps")
    osd = element("nvdsosd", "osd")
    record_tee = element("tee", "record-tee") if record_path else None
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

    elements = source_elements + [h265_parser, h264_parser, decoder, queue, streammux]
    if raw_tee:
        elements.append(raw_tee)
    elements.append(pgie)
    if detect_tee:
        elements.append(detect_tee)
    if assessment_queue and sgie:
        elements.extend((assessment_queue, sgie))
    if assess_tee:
        elements.append(assess_tee)
    elements.extend((convert, caps, osd))
    if record_tee:
        elements.append(record_tee)
    if display_queue:
        elements.append(display_queue)
    elements.append(sink)

    for elem in elements:
        pipeline.add(elem)

    # The parser-to-decoder link is deferred to the pad-added callbacks: the
    # decoder has one sink pad, so only the codec the source actually carries
    # may claim it.
    parsers = {"h265": h265_parser, "h264": h264_parser}
    if stream.is_rtsp:
        source.connect(
            "pad-added",
            on_rtsp_pad_added,
            {"h265": h265_depay, "h264": h264_depay},
            parsers,
            decoder,
        )
        h265_depay.link(h265_parser)
        h264_depay.link(h264_parser)
    else:
        source.link(demux)
        demux.connect("pad-added", on_file_pad_added, parsers, decoder)

    decoder.link(queue)
    queue.get_static_pad("src").link(streammux.request_pad_simple("sink_0"))
    raw_appsink = None
    if raw_tee and raw_output_size:
        streammux.link(raw_tee)
        raw_appsink = compressed_branch(
            pipeline,
            raw_tee,
            "raw-output",
            raw_output_size[0],
            raw_output_size[1],
            jpeg_quality,
        )
        tee_queue_branch(pipeline, raw_tee, "raw-main", pgie)
    else:
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
    record_sink = None
    if record_tee:
        # Tee after the OSD so the recording captures the burned-in overlay.
        osd.link(record_tee)
        record_sink = recording_branch(pipeline, record_tee, record_path)
        if display_queue:
            link_tee_to_queue(record_tee, display_queue)
            display_queue.link(sink)
        else:
            tee_queue_branch(pipeline, record_tee, "display-main", sink)
    elif display_queue:
        osd.link(display_queue)
        display_queue.link(sink)
    else:
        osd.link(sink)

    return PipelineParts(
        pipeline=pipeline,
        streammux=streammux,
        raw_appsink=raw_appsink,
        pgie=pgie,
        detect_appsink=detect_appsink,
        sgie=sgie,
        assess_appsink=assess_appsink,
        caps=caps,
        osd=osd,
        sink=sink,
        record_sink=record_sink,
    )
