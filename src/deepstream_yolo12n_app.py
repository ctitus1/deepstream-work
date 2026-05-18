#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")

from gi.repository import GLib, Gst, GstPbutils

import pyds


PROJECT_DIR = Path("/home/user/deepstream-work")
STREAM_PATH = PROJECT_DIR / "streams/dtc-d4.ts"
INFER_CONFIG = PROJECT_DIR / "configs/config_infer_primary_yolo12n.txt"
MODEL_META = PROJECT_DIR / "models/yolo12n.meta.json"

PERSON_CLASS_ID = 0


def load_model_meta():
    if not MODEL_META.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_META}. Run: ./scripts/setup_yolo12n_model.sh 640"
        )

    meta = json.loads(MODEL_META.read_text())
    return int(meta["model_width"]), int(meta["model_height"])


def discover_video_size(path: Path):
    uri = path.resolve().as_uri()
    discoverer = GstPbutils.Discoverer.new(5 * Gst.SECOND)
    info = discoverer.discover_uri(uri)

    for stream in info.get_video_streams():
        width = stream.get_width()
        height = stream.get_height()
        if width and height:
            return int(width), int(height)

    raise RuntimeError(f"Could not discover video size for {path}")


def clamp(value, low, high):
    return max(low, min(high, value))


def transform_model_rect_to_frame(rect, frame_w, frame_h, model_w, model_h):
    scale = min(model_w / frame_w, model_h / frame_h)
    resized_w = frame_w * scale
    resized_h = frame_h * scale
    pad_x = (model_w - resized_w) / 2.0
    pad_y = (model_h - resized_h) / 2.0

    left = (rect.left - pad_x) / scale
    top = (rect.top - pad_y) / scale
    width = rect.width / scale
    height = rect.height / scale

    right = left + width
    bottom = top + height

    left = clamp(left, 0.0, frame_w - 1.0)
    top = clamp(top, 0.0, frame_h - 1.0)
    right = clamp(right, 0.0, frame_w - 1.0)
    bottom = clamp(bottom, 0.0, frame_h - 1.0)

    rect.left = float(left)
    rect.top = float(top)
    rect.width = float(max(0.0, right - left))
    rect.height = float(max(0.0, bottom - top))


def bbox_correction_probe(model_w, model_h):
    def _probe(_pad, info, _user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_w = int(getattr(frame_meta, "source_frame_width", 0) or 0)
            frame_h = int(getattr(frame_meta, "source_frame_height", 0) or 0)

            if frame_w <= 0 or frame_h <= 0:
                frame_w = int(getattr(frame_meta, "pipeline_width", 0) or model_w)
                frame_h = int(getattr(frame_meta, "pipeline_height", 0) or model_h)

            l_obj = frame_meta.obj_meta_list

            while l_obj is not None:
                next_obj = l_obj.next

                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if int(obj_meta.class_id) != PERSON_CLASS_ID:
                    try:
                        pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                    except Exception:
                        obj_meta.rect_params.border_width = 0
                        obj_meta.text_params.display_text = ""
                    l_obj = next_obj
                    continue

                rect = obj_meta.rect_params

                right = rect.left + rect.width
                bottom = rect.top + rect.height

                looks_like_model_space = (
                    frame_w > model_w
                    and frame_h > model_h
                    and right <= model_w * 1.05
                    and bottom <= model_h * 1.05
                )

                if looks_like_model_space:
                    transform_model_rect_to_frame(rect, frame_w, frame_h, model_w, model_h)

                rect.left = float(clamp(rect.left, 0.0, frame_w - 1.0))
                rect.top = float(clamp(rect.top, 0.0, frame_h - 1.0))
                rect.width = float(clamp(rect.width, 0.0, frame_w - rect.left))
                rect.height = float(clamp(rect.height, 0.0, frame_h - rect.top))
                rect.border_width = 3

                confidence = float(obj_meta.confidence)
                obj_meta.text_params.display_text = f"person {confidence:.2f}"

                l_obj = next_obj

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    return _probe


def make_element(factory, name):
    elem = Gst.ElementFactory.make(factory, name)
    if elem is None:
        raise RuntimeError(f"Could not create element: {factory} ({name})")
    return elem


def on_demux_pad_added(_demux, pad, parser):
    caps = pad.get_current_caps() or pad.query_caps(None)
    caps_str = caps.to_string()

    if "video/x-h265" not in caps_str:
        return

    sink_pad = parser.get_static_pad("sink")
    if sink_pad.is_linked():
        return

    result = pad.link(sink_pad)
    if result != Gst.PadLinkReturn.OK:
        raise RuntimeError(f"Failed to link demux to h265parse: {result}")


def on_bus_message(_bus, message, loop):
    if message.type == Gst.MessageType.EOS:
        print("EOS")
        loop.quit()

    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"ERROR: {err}", file=sys.stderr)
        print(f"DEBUG: {debug}", file=sys.stderr)
        loop.quit()

    return True


def main():
    Gst.init(None)

    if not STREAM_PATH.exists():
        raise FileNotFoundError(f"Missing input video: {STREAM_PATH}")

    if not INFER_CONFIG.exists():
        raise FileNotFoundError(f"Missing nvinfer config: {INFER_CONFIG}")

    model_w, model_h = load_model_meta()
    source_w, source_h = discover_video_size(STREAM_PATH)

    print(f"Input video: {source_w}x{source_h}")
    print(f"Model input: {model_w}x{model_h}")
    print(f"Display:     {source_w}x{source_h}")

    pipeline = Gst.Pipeline.new("deepstream-yolo12n-pipeline")

    source = make_element("filesrc", "file-source")
    demux = make_element("tsdemux", "ts-demux")
    parser = make_element("h265parse", "h265-parser")
    decoder = make_element("nvv4l2decoder", "nvv4l2-decoder")
    queue = make_element("queue", "decode-queue")
    streammux = make_element("nvstreammux", "stream-muxer")
    pgie = make_element("nvinfer", "primary-inference")
    nvvidconv = make_element("nvvideoconvert", "nv-video-converter")
    nvosd = make_element("nvdsosd", "nv-onscreendisplay")
    sink = make_element("nveglglessink", "display-sink")

    source.set_property("location", str(STREAM_PATH))

    streammux.set_property("batch-size", 1)
    streammux.set_property("width", source_w)
    streammux.set_property("height", source_h)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("live-source", 0)

    pgie.set_property("config-file-path", str(INFER_CONFIG))

    sink.set_property("sync", False)
    sink.set_property("qos", False)

    for elem in [
        source,
        demux,
        parser,
        decoder,
        queue,
        streammux,
        pgie,
        nvvidconv,
        nvosd,
        sink,
    ]:
        pipeline.add(elem)

    if not source.link(demux):
        raise RuntimeError("Failed to link filesrc -> tsdemux")

    demux.connect("pad-added", on_demux_pad_added, parser)

    if not parser.link(decoder):
        raise RuntimeError("Failed to link h265parse -> nvv4l2decoder")

    if not decoder.link(queue):
        raise RuntimeError("Failed to link nvv4l2decoder -> queue")

    mux_sink_pad = streammux.request_pad_simple("sink_0")
    if mux_sink_pad is None:
        mux_sink_pad = streammux.get_request_pad("sink_0")
    if mux_sink_pad is None:
        raise RuntimeError("Could not get nvstreammux sink_0 pad")

    queue_src_pad = queue.get_static_pad("src")
    if queue_src_pad.link(mux_sink_pad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Failed to link queue -> nvstreammux")

    if not streammux.link(pgie):
        raise RuntimeError("Failed to link nvstreammux -> nvinfer")

    if not pgie.link(nvvidconv):
        raise RuntimeError("Failed to link nvinfer -> nvvideoconvert")

    if not nvvidconv.link(nvosd):
        raise RuntimeError("Failed to link nvvideoconvert -> nvdsosd")

    if not nvosd.link(sink):
        raise RuntimeError("Failed to link nvdsosd -> sink")

    pgie_src_pad = pgie.get_static_pad("src")
    if pgie_src_pad is None:
        raise RuntimeError("Could not get nvinfer src pad")

    pgie_src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        bbox_correction_probe(model_w, model_h),
        None,
    )

    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted")

    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
