#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")

from gi.repository import GLib, Gst, GstPbutils

import pyds


PROJECT_DIR = Path("/home/user/deepstream-work")
DEFAULT_STREAM = PROJECT_DIR / "streams/dtc-d4.ts"
SETUP_SCRIPT = PROJECT_DIR / "scripts/setup_and_export_yolo.sh"

PERSON_CLASS_ID = 0
CACHE_POLICY = "parser_bbox_default_size_preferred_v4"


def model_stem_from_name(model: str) -> str:
    name = Path(model).name
    return name[:-3] if name.endswith(".pt") else name


def discover_video_size(path: Path) -> tuple[int, int]:
    uri = path.resolve().as_uri()
    discoverer = GstPbutils.Discoverer.new(5 * Gst.SECOND)
    info = discoverer.discover_uri(uri)

    for stream in info.get_video_streams():
        width = stream.get_width()
        height = stream.get_height()
        if width and height:
            return int(width), int(height)

    raise RuntimeError(f"Could not discover video size for {path}")


def get_onnx_input_size(onnx_path: Path) -> tuple[int, int]:
    python_bin = PROJECT_DIR / ".venv-yolo/bin/python"

    if not python_bin.exists():
        raise FileNotFoundError(
            f"Missing {python_bin}. The export script should create this venv."
        )

    code = f"""
import onnx
m = onnx.load({str(onnx_path)!r})
x = m.graph.input[0]
dims = [d.dim_value if d.dim_value else d.dim_param for d in x.type.tensor_type.shape.dim]
print(int(dims[3]), int(dims[2]))
"""

    result = subprocess.run(
        [str(python_bin), "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    w, h = result.stdout.strip().split()
    return int(w), int(h)


def cached_paths(model_stem: str, long_side: int, model_w: int, model_h: int) -> dict[str, Path]:
    tag = f"{model_stem}_{long_side}_{model_w}x{model_h}"

    return {
        "tag": tag,
        "onnx": PROJECT_DIR / "models" / f"{tag}.onnx",
        "meta": PROJECT_DIR / "models" / f"{tag}.meta.json",
        "engine": PROJECT_DIR / "models" / f"{tag}.onnx_b1_gpu0_fp16.engine",
        "infer_config": PROJECT_DIR / "configs" / f"config_infer_primary_{tag}.txt",
    }


def write_infer_config(paths: dict[str, Path]) -> None:
    paths["infer_config"].write_text(
        f"""[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0

onnx-file={paths["onnx"]}
model-engine-file={paths["engine"]}
labelfile-path=/home/user/deepstream-work/models/coco_labels.txt

batch-size=1
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0

parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=/home/user/deepstream-work/lib/libnvdsinfer_custom_impl_Yolo.so
output-blob-names=output

maintain-aspect-ratio=1
symmetric-padding=1

cluster-mode=2

[class-attrs-all]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
topk=300

[class-attrs-0]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
topk=300
"""
    )


def cache_is_valid(paths: dict[str, Path], model_w: int, model_h: int) -> bool:
    if not paths["onnx"].exists() or not paths["meta"].exists():
        return False

    actual_w, actual_h = get_onnx_input_size(paths["onnx"])
    if actual_w != model_w or actual_h != model_h:
        return False

    try:
        meta = json.loads(paths["meta"].read_text())
    except Exception:
        return False

    return (
        meta.get("cache_policy") == CACHE_POLICY
        and int(meta.get("model_width", -1)) == model_w
        and int(meta.get("model_height", -1)) == model_h
    )


def delete_stale_cache(model_stem: str, long_side: int) -> None:
    for onnx_path in sorted((PROJECT_DIR / "models").glob(f"{model_stem}_{long_side}_*.onnx")):
        stem = onnx_path.stem
        for stale in [
            onnx_path,
            PROJECT_DIR / "models" / f"{stem}.meta.json",
            PROJECT_DIR / "models" / f"{stem}.onnx_b1_gpu0_fp16.engine",
            PROJECT_DIR / "configs" / f"config_infer_primary_{stem}.txt",
        ]:
            stale.unlink(missing_ok=True)


def ensure_model_and_config(
    model: str,
    stream: Path,
    long_side: int,
    source_w: int,
    source_h: int,
) -> tuple[int, int, Path]:
    model_stem = model_stem_from_name(model)

    for onnx_path in sorted((PROJECT_DIR / "models").glob(f"{model_stem}_{long_side}_*.onnx")):
        actual_w, actual_h = get_onnx_input_size(onnx_path)
        paths = cached_paths(model_stem, long_side, actual_w, actual_h)

        if onnx_path == paths["onnx"] and cache_is_valid(paths, actual_w, actual_h):
            write_infer_config(paths)
            return actual_w, actual_h, paths["infer_config"]

    delete_stale_cache(model_stem, long_side)

    if not SETUP_SCRIPT.exists():
        raise FileNotFoundError(f"Missing export/setup script: {SETUP_SCRIPT}")

    print(f"No valid cached {model_stem} model for long_side={long_side}.")
    print("Exporting ONNX with setup script...")

    subprocess.run(
        [str(SETUP_SCRIPT), model, str(long_side), str(stream.relative_to(PROJECT_DIR))],
        cwd=str(PROJECT_DIR),
        check=True,
    )

    base_onnx = PROJECT_DIR / "models" / f"{model_stem}.onnx"
    if not base_onnx.exists():
        raise FileNotFoundError(f"Setup did not create {base_onnx}")

    actual_w, actual_h = get_onnx_input_size(base_onnx)
    paths = cached_paths(model_stem, long_side, actual_w, actual_h)

    shutil.copy2(base_onnx, paths["onnx"])

    paths["meta"].write_text(
        json.dumps(
            {
                "model": model_stem,
                "model_arg": model,
                "requested_long_side": long_side,
                "source_stream": str(stream.relative_to(PROJECT_DIR)),
                "source_width": source_w,
                "source_height": source_h,
                "model_width": actual_w,
                "model_height": actual_h,
                "onnx": str(paths["onnx"].relative_to(PROJECT_DIR)),
                "engine": str(paths["engine"].relative_to(PROJECT_DIR)),
                "labels": "models/coco_labels.txt",
                "cache_policy": CACHE_POLICY,
                "note": "DeepStream-Yolo parser path. Non-default custom input sizes are allowed but may have unreliable bbox metadata; default-size YOLO exports are preferred.",
            },
            indent=2,
        )
        + "\n"
    )

    write_infer_config(paths)

    print(f"Actual ONNX input: {actual_w}x{actual_h}")
    print(f"Cached ONNX:       {paths['onnx']}")
    print(f"Infer config:      {paths['infer_config']}")
    print(f"Engine path:       {paths['engine']}")
    print("Engine will be generated by nvinfer on first run if missing.")

    return actual_w, actual_h, paths["infer_config"]


def clamp(value, low, high):
    return max(low, min(high, value))


def box_tuple(rect):
    return float(rect.left), float(rect.top), float(rect.width), float(rect.height)


def box_is_usable(left, top, width, height, frame_w, frame_h):
    if width < 2 or height < 2:
        return False
    if left >= frame_w - 1 or top >= frame_h - 1:
        return False
    if left + width <= 1 or top + height <= 1:
        return False
    return True


def clamp_rect(rect, frame_w, frame_h):
    left = clamp(float(rect.left), 0.0, frame_w - 1.0)
    top = clamp(float(rect.top), 0.0, frame_h - 1.0)
    right = clamp(float(rect.left + rect.width), 0.0, frame_w - 1.0)
    bottom = clamp(float(rect.top + rect.height), 0.0, frame_h - 1.0)

    if right <= left:
        right = min(frame_w - 1.0, left + 2.0)
    if bottom <= top:
        bottom = min(frame_h - 1.0, top + 2.0)

    rect.left = float(left)
    rect.top = float(top)
    rect.width = float(max(0.0, right - left))
    rect.height = float(max(0.0, bottom - top))


def transform_model_rect_to_frame_values(left, top, width, height, frame_w, frame_h, model_w, model_h):
    scale = min(model_w / frame_w, model_h / frame_h)
    resized_w = frame_w * scale
    resized_h = frame_h * scale
    pad_x = (model_w - resized_w) / 2.0
    pad_y = (model_h - resized_h) / 2.0

    out_left = (left - pad_x) / scale
    out_top = (top - pad_y) / scale
    out_width = width / scale
    out_height = height / scale

    out_right = out_left + out_width
    out_bottom = out_top + out_height

    out_left = clamp(out_left, 0.0, frame_w - 1.0)
    out_top = clamp(out_top, 0.0, frame_h - 1.0)
    out_right = clamp(out_right, 0.0, frame_w - 1.0)
    out_bottom = clamp(out_bottom, 0.0, frame_h - 1.0)

    return out_left, out_top, max(0.0, out_right - out_left), max(0.0, out_bottom - out_top)


def apply_rect(rect, values):
    rect.left = float(values[0])
    rect.top = float(values[1])
    rect.width = float(values[2])
    rect.height = float(values[3])


def bbox_correction_probe(
    model_w,
    model_h,
    bbox_mode,
    anchor_box_to_label,
    debug_bboxes,
    debug_frames,
):
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

            frame_debug = debug_bboxes and frame_meta.frame_num < debug_frames
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
                orig = box_tuple(rect)

                right = orig[0] + orig[2]
                bottom = orig[1] + orig[3]

                looks_like_model_space = (
                    frame_w > model_w
                    and frame_h > model_h
                    and right <= model_w * 1.05
                    and bottom <= model_h * 1.05
                )

                corrected = transform_model_rect_to_frame_values(
                    orig[0],
                    orig[1],
                    orig[2],
                    orig[3],
                    frame_w,
                    frame_h,
                    model_w,
                    model_h,
                )

                label_x = float(getattr(obj_meta.text_params, "x_offset", 0) or 0)
                label_y = float(getattr(obj_meta.text_params, "y_offset", 0) or 0)

                def dist_to_label(box):
                    x, y, _w, _h = box
                    return abs(x - label_x) + abs(y - label_y)

                correction_used = "none"

                if bbox_mode == "letterbox" or (bbox_mode == "auto" and looks_like_model_space):
                    if box_is_usable(*corrected, frame_w, frame_h):
                        apply_rect(rect, corrected)
                        correction_used = "letterbox"
                    else:
                        apply_rect(rect, orig)
                        correction_used = "fallback-original"

                elif bbox_mode == "auto2":
                    orig_ok = box_is_usable(*orig, frame_w, frame_h)
                    corrected_ok = box_is_usable(*corrected, frame_w, frame_h)

                    if corrected_ok and not orig_ok:
                        apply_rect(rect, corrected)
                        correction_used = "auto2-letterbox-only-valid"
                    elif orig_ok and not corrected_ok:
                        apply_rect(rect, orig)
                        correction_used = "auto2-original-only-valid"
                    elif orig_ok and corrected_ok:
                        if dist_to_label(corrected) < dist_to_label(orig):
                            apply_rect(rect, corrected)
                            correction_used = "auto2-letterbox-nearer-label"
                        else:
                            apply_rect(rect, orig)
                            correction_used = "auto2-original-nearer-label"
                    else:
                        apply_rect(rect, orig)
                        correction_used = "auto2-fallback-original"

                elif bbox_mode == "off":
                    apply_rect(rect, orig)
                    correction_used = "off"

                elif bbox_mode == "auto":
                    apply_rect(rect, orig)
                    correction_used = "auto-original"

                clamp_rect(rect, frame_w, frame_h)

                # Do not use pre-existing text_params offsets to move the box.
                # Those offsets may be stale/uninitialized metadata. The rectangle
                # is the source of truth; the label is attached to it below.
                if anchor_box_to_label:
                    pass

                # NvOSD rectangles are axis-aligned. This script only writes
                # left/top/width/height, so it never intentionally creates rotation.
                rect.border_width = 3
                rect.has_bg_color = 0
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)

                confidence = float(obj_meta.confidence)
                obj_meta.text_params.display_text = f"person {confidence:.2f}"

                # Force label and box to stay attached.
                obj_meta.text_params.x_offset = int(rect.left)
                obj_meta.text_params.y_offset = max(0, int(rect.top) - 10)

                final = box_tuple(rect)

                if frame_debug:
                    print(
                        "BBOX_DEBUG "
                        f"frame={frame_meta.frame_num} "
                        f"conf={confidence:.3f} "
                        f"frame_size={frame_w}x{frame_h} "
                        f"model_size={model_w}x{model_h} "
                        f"mode={bbox_mode} "
                        f"anchor_label={anchor_box_to_label} "
                        f"looks_model={looks_like_model_space} "
                        f"correction={correction_used} "
                        f"label=({label_x:.1f},{label_y:.1f}) "
                        f"orig=({orig[0]:.1f},{orig[1]:.1f},{orig[2]:.1f},{orig[3]:.1f}) "
                        f"letterbox=({corrected[0]:.1f},{corrected[1]:.1f},{corrected[2]:.1f},{corrected[3]:.1f}) "
                        f"final=({final[0]:.1f},{final[1]:.1f},{final[2]:.1f},{final[3]:.1f})",
                        flush=True,
                    )

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
    argp = argparse.ArgumentParser(description="DeepStream-Yolo parser app")
    argp.add_argument("--model", default="yolo12x.pt", help="YOLO .pt model name/path")
    argp.add_argument("--long-side", type=int, default=640, help="Model long side")
    argp.add_argument("--stream", default=str(DEFAULT_STREAM), help="Input video/stream path")
    argp.add_argument(
        "--bbox-mode",
        choices=["auto", "auto2", "letterbox", "off"],
        default="letterbox",
        help="BBox correction mode. Default: letterbox",
    )
    argp.add_argument(
        "--anchor-box-to-label",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force bbox top-left to follow label position. Default: disabled",
    )
    argp.add_argument(
        "--debug-bboxes",
        action="store_true",
        help="Print bbox coordinate diagnostics for early frames",
    )
    argp.add_argument(
        "--debug-frames",
        type=int,
        default=10,
        help="Number of initial frames to print bbox diagnostics for",
    )
    args = argp.parse_args()

    Gst.init(None)

    stream_path = Path(args.stream)
    if not stream_path.is_absolute():
        stream_path = PROJECT_DIR / stream_path

    if not stream_path.exists():
        raise FileNotFoundError(f"Missing input video: {stream_path}")

    source_w, source_h = discover_video_size(stream_path)

    model_w, model_h, infer_config = ensure_model_and_config(
        model=args.model,
        stream=stream_path,
        long_side=args.long_side,
        source_w=source_w,
        source_h=source_h,
    )

    print(f"Model:        {args.model}")
    print(f"Input video:  {source_w}x{source_h}")
    print(f"Model input:  {model_w}x{model_h}")
    print(f"Display:      {source_w}x{source_h}")
    print(f"Infer config: {infer_config}")
    print(f"BBox mode:    {args.bbox_mode}")
    print(f"Anchor label: {args.anchor_box_to_label}")

    pipeline = Gst.Pipeline.new("deepstream-yolo-parser-pipeline")

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

    source.set_property("location", str(stream_path))

    streammux.set_property("batch-size", 1)
    streammux.set_property("width", source_w)
    streammux.set_property("height", source_h)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("live-source", 0)

    pgie.set_property("config-file-path", str(infer_config))

    # CPU OSD avoids GPU/overlay rendering artifacts when bbox coords are valid
    # but displayed boxes look visually skewed/slanted.
    try:
        nvosd.set_property("process-mode", 0)
    except TypeError:
        pass

    try:
        nvosd.set_property("display-bbox", 1)
        nvosd.set_property("display-text", 1)
    except TypeError:
        pass

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
        bbox_correction_probe(
            model_w=model_w,
            model_h=model_h,
            bbox_mode=args.bbox_mode,
            anchor_box_to_label=args.anchor_box_to_label,
            debug_bboxes=args.debug_bboxes,
            debug_frames=args.debug_frames,
        ),
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
