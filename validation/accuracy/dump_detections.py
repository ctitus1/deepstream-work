#!/usr/bin/env python3
"""Dump deployed DeepStream detections to COCO-format JSON.

Runs the shared pipeline (``deepstream_yolo.pipeline.build_pipeline``) headless
over a video or RTSP source and records every PGIE detection. The output feeds
``score_detections.py``. Assessment (SGIE) is off: this measures the detector
only.

The dump records the thresholds and geometry it ran under, because the deployed
nvinfer settings are not the ultralytics ``val`` defaults and the two are not
directly comparable.
"""

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import json
import sys
from pathlib import Path

import gi

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

from deepstream_yolo.assessment_runtime import frame_timestamp  # noqa: E402
from deepstream_yolo.detection_overlay import bbox_probe, get_detection_id  # noqa: E402
from deepstream_yolo.model_cache import discover_size, ensure_model  # noqa: E402
from deepstream_yolo.paths import (  # noqa: E402
    DEFAULT_STREAM,
    LABELS_PATH,
    LABELS_SOURCE_PATH,
    resolve_project_path,
)
from deepstream_yolo.pipeline import build_pipeline, on_message  # noqa: E402
from deepstream_yolo.stream_source import resolve_stream_source  # noqa: E402
from coco_categories import CATEGORY_MAP_HELP, CATEGORY_MAPS, describe_category_map, map_category_id  # noqa: E402

# ultralytics `val` defaults; the deployed nvinfer config uses much stricter values.
ULTRALYTICS_VAL_CONF = 0.001
ULTRALYTICS_VAL_IOU = 0.7


def read_infer_thresholds(config_path: Path) -> dict:
    """Pull the thresholds nvinfer actually ran with out of the generated config."""
    parser = configparser.ConfigParser(strict=False)
    parser.read(config_path)

    thresholds = {}
    for section in ("class-attrs-all", "property"):
        if not parser.has_section(section):
            continue
        for key in ("pre-cluster-threshold", "nms-iou-threshold", "topk"):
            if parser.has_option(section, key):
                thresholds[key.replace("-", "_")] = float(parser.get(section, key))
    return thresholds


def load_labels() -> list:
    for path in (LABELS_PATH, LABELS_SOURCE_PATH):
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return []


def load_image_id_map(path: Path) -> dict:
    """Frame index -> ground-truth image id, from a JSON list or object."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {index: value for index, value in enumerate(data)}
    if isinstance(data, dict):
        return {int(key): value for key, value in data.items()}
    raise ValueError(f"{path} must contain a JSON list or object mapping frame index to image id")


class ImageIdMapper:
    """Explicit image_id policy; pycocotools needs one consistent type."""

    def __init__(self, mode: str, id_type: str, offset: int, mapping: dict | None):
        self.mode = mode
        self.id_type = id_type
        self.offset = offset
        self.mapping = mapping or {}
        self.types = set()

    def image_id(self, frame_num: int):
        if self.mode == "map":
            if frame_num not in self.mapping:
                raise KeyError(
                    f"No image id for frame {frame_num} in --image-id-map; the map must cover every "
                    "decoded frame, or use --max-frames to stop at its length"
                )
            raw = self.mapping[frame_num]
        else:
            raw = int(frame_num) + self.offset

        value = self.coerce(raw)
        self.types.add(type(value).__name__)
        return value

    def coerce(self, raw):
        if self.id_type == "int":
            try:
                return int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"image id {raw!r} is not an int; pass --image-id-type str") from exc
        if self.id_type == "str":
            return str(raw)
        # auto reproduces the ultralytics rule: int(stem) when numeric, else the string.
        text = str(raw)
        return int(text) if text.isdigit() else text

    def check(self) -> None:
        if len(self.types) > 1:
            raise ValueError(
                f"Mixed image_id types {sorted(self.types)} in one dump. pycocotools matches ids by "
                "value AND type; re-run with --image-id-type int or --image-id-type str"
            )


class DetectionDump:
    """PGIE probe that accumulates COCO-style detection records."""

    def __init__(
        self,
        mapper: ImageIdMapper,
        category_map: str,
        class_ids: set | None,
        min_score: float,
        scale: tuple,
        image_size: tuple,
        max_frames: int,
        loop,
    ):
        self.mapper = mapper
        self.category_map = category_map
        self.class_ids = class_ids
        self.min_score = min_score
        self.scale_x, self.scale_y = scale
        self.image_width, self.image_height = image_size
        self.max_frames = max_frames
        self.loop = loop
        self.detections = []
        self.images = []
        self.seen_frames = set()
        self.stopping = False

    def probe(self, _pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            frame_num = int(frame_meta.frame_num)
            if self.max_frames and len(self.seen_frames) >= self.max_frames:
                self.stop()
                break
            if frame_num not in self.seen_frames:
                try:
                    self.record_frame(frame_meta, buffer, frame_num)
                except (KeyError, ValueError) as exc:
                    # Raising inside a pad probe just spams tracebacks; stop cleanly instead.
                    print(f"stopping: {exc}", file=sys.stderr, flush=True)
                    self.stop()
                    break
            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    def record_frame(self, frame_meta, buffer, frame_num: int) -> None:
        image_id = self.mapper.image_id(frame_num)
        self.seen_frames.add(frame_num)
        source, timestamp = frame_timestamp(frame_meta, buffer)
        self.images.append(
            {
                "id": image_id,
                "frame": frame_num,
                "width": self.image_width,
                "height": self.image_height,
                "timestamp_ns": timestamp,
                "timestamp_source": source,
            }
        )

        obj_list = frame_meta.obj_meta_list
        index = 0
        while obj_list:
            obj = pyds.NvDsObjectMeta.cast(obj_list.data)
            class_id = int(obj.class_id)
            score = float(obj.confidence)
            if (self.class_ids is None or class_id in self.class_ids) and score >= self.min_score:
                rect = obj.rect_params
                self.detections.append(
                    {
                        "image_id": image_id,
                        "category_id": map_category_id(class_id, self.category_map),
                        "bbox": [
                            round(float(rect.left) * self.scale_x, 2),
                            round(float(rect.top) * self.scale_y, 2),
                            round(float(rect.width) * self.scale_x, 2),
                            round(float(rect.height) * self.scale_y, 2),
                        ],
                        "score": round(score, 5),
                        "class_id": class_id,
                        "frame": frame_num,
                        "detection_id": get_detection_id(obj, index),
                    }
                )
            index += 1
            obj_list = obj_list.next

    def stop(self) -> None:
        if self.stopping:
            return
        self.stopping = True
        GLib.idle_add(self.loop.quit)


def parse_size(text: str) -> tuple:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected WxH, got {text!r}")
    return int(parts[0]), int(parts[1])


def parse_class_ids(text: str) -> set | None:
    if not text or text.lower() == "all":
        return None
    return {int(value) for value in text.replace(" ", "").split(",") if value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump DeepStream detections to COCO JSON.")
    parser.add_argument("--stream", default=str(DEFAULT_STREAM))
    parser.add_argument("--model", default="yolo12x-custom.pt")
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help=(
            "nvinfer pre-cluster-threshold for the generated config. 0.25 is the deployed value; "
            f"use {ULTRALYTICS_VAL_CONF} to make mAP comparable to an ultralytics baseline."
        ),
    )
    parser.add_argument("--output", default="outputs/validation/detections.json")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 runs until EOS.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Extra score floor applied after nvinfer.")
    parser.add_argument("--class-ids", default="all", help="Comma list of model class ids to keep, or 'all'.")
    parser.add_argument("--category-map", choices=CATEGORY_MAPS, default="coco91", help=CATEGORY_MAP_HELP)
    parser.add_argument("--image-id-mode", choices=("frame", "map"), default="frame")
    parser.add_argument("--image-id-offset", type=int, default=0, help="Added to the frame number in frame mode.")
    parser.add_argument("--image-id-map", default=None, help="JSON list/object of ground-truth image ids per frame.")
    parser.add_argument("--image-id-type", choices=("int", "str", "auto"), default="int")
    parser.add_argument(
        "--target-size",
        default=None,
        help="WxH of the ground-truth images; boxes are scaled from the source resolution.",
    )
    parser.add_argument("--rtsp-latency-ms", type=int, default=0)
    parser.add_argument("--overlay", action="store_true", help="Also run the display bbox probe.")
    parser.add_argument("--results-only", action="store_true", help="Write a bare COCO results array.")
    parser.add_argument("--indent", type=int, default=0, help="JSON indent; 0 writes compact output.")
    args = parser.parse_args()
    if args.image_id_mode == "map" and not args.image_id_map:
        parser.error("--image-id-mode map requires --image-id-map")
    return args


def warn_sharp_edges(args, thresholds: dict, model_size: tuple, source_size: tuple) -> list:
    """Surface the comparisons that silently make a healthy model look broken."""
    warnings = []
    pre_cluster = thresholds.get("pre_cluster_threshold")
    nms_iou = thresholds.get("nms_iou_threshold")

    if pre_cluster is not None and pre_cluster > ULTRALYTICS_VAL_CONF:
        warnings.append(
            f"threshold mismatch: nvinfer pre-cluster-threshold={pre_cluster} truncates the "
            f"precision-recall curve, while ultralytics val defaults to conf={ULTRALYTICS_VAL_CONF}. "
            "mAP from this dump is a deployment number, not a model-quality number; re-run with "
            f"--conf {ULTRALYTICS_VAL_CONF} for an apples-to-apples comparison."
        )
    if nms_iou is not None and abs(nms_iou - ULTRALYTICS_VAL_IOU) > 1e-6:
        warnings.append(
            f"nms mismatch: nvinfer nms-iou-threshold={nms_iou} vs ultralytics val iou={ULTRALYTICS_VAL_IOU}; "
            "a stricter NMS removes crowded true positives and lowers recall."
        )
    if model_size[0] != model_size[1]:
        warnings.append(
            f"non-square deploy resolution {model_size[0]}x{model_size[1]} vs the square letterbox used by "
            "offline eval; small-object recall and box regression differ between the two."
        )
    if args.category_map == "coco91":
        warnings.append(
            "category ids were mapped with coco91; ground truth must use 91-class COCO ids "
            "(--category-map identity for a custom dataset)."
        )
    if args.target_size and parse_size(args.target_size) != source_size:
        warnings.append(
            f"boxes rescaled from source {source_size[0]}x{source_size[1]} to "
            f"{args.target_size}; verify the ground truth is in that same space."
        )
    return warnings


def main() -> int:
    args = parse_args()
    stream = resolve_stream_source(args.stream)

    Gst.init(None)
    src_w, src_h = discover_size(stream.uri)
    model_w, model_h, config = ensure_model(args.model, stream, args.long_side, src_w, src_h, args.conf)
    thresholds = read_infer_thresholds(config)

    target_w, target_h = parse_size(args.target_size) if args.target_size else (src_w, src_h)
    scale = (target_w / src_w, target_h / src_h)
    warnings = warn_sharp_edges(args, thresholds, (model_w, model_h), (src_w, src_h))

    print(
        f"stream={stream.display} video={src_w}x{src_h} model={model_w}x{model_h} "
        f"conf={args.conf} config={config}",
        flush=True,
    )
    print(
        f"category_map={describe_category_map(args.category_map)} "
        f"image_id_mode={args.image_id_mode} image_id_type={args.image_id_type}",
        flush=True,
    )
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr, flush=True)

    mapping = load_image_id_map(Path(args.image_id_map)) if args.image_id_map else None
    mapper = ImageIdMapper(args.image_id_mode, args.image_id_type, args.image_id_offset, mapping)

    loop = GLib.MainLoop()
    parts = build_pipeline(
        stream,
        src_w,
        src_h,
        config,
        None,
        rtsp_latency_ms=args.rtsp_latency_ms,
        display=False,
    )
    dump = DetectionDump(
        mapper,
        args.category_map,
        parse_class_ids(args.class_ids),
        args.min_score,
        scale,
        (target_w, target_h),
        args.max_frames,
        loop,
    )
    pgie_src = parts.pgie.get_static_pad("src")
    if args.overlay:
        pgie_src.add_probe(Gst.PadProbeType.BUFFER, bbox_probe(args.conf), None)
    pgie_src.add_probe(Gst.PadProbeType.BUFFER, dump.probe, None)

    bus = parts.pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    parts.pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        parts.pipeline.set_state(Gst.State.NULL)

    mapper.check()
    if not dump.detections:
        print("No detections recorded; nothing to score.", file=sys.stderr)

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = args.indent or None

    if args.results_only:
        payload = dump.detections
    else:
        payload = {
            "info": {
                "generated": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
                "source": stream.display,
                "model": args.model,
                "infer_config": str(config),
                "requested_conf": args.conf,
                "thresholds": thresholds,
                "ultralytics_val_defaults": {"conf": ULTRALYTICS_VAL_CONF, "iou": ULTRALYTICS_VAL_IOU},
                "source_width": src_w,
                "source_height": src_h,
                "model_width": model_w,
                "model_height": model_h,
                "output_width": target_w,
                "output_height": target_h,
                "category_map": args.category_map,
                "image_id_mode": args.image_id_mode,
                "image_id_type": args.image_id_type,
                "class_ids": args.class_ids,
                "min_score": args.min_score,
                "frames": len(dump.seen_frames),
                "detections": len(dump.detections),
                "labels": load_labels(),
                "warnings": warnings,
            },
            "images": dump.images,
            "detections": dump.detections,
        }

    output.write_text(json.dumps(payload, indent=indent) + "\n")
    print(f"frames={len(dump.seen_frames)} detections={len(dump.detections)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
