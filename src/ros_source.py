#!/usr/bin/env python3
"""DeepStream frame source for ROS publishing.

This entrypoint prepares model configs, calls
``deepstream_yolo.pipeline.build_pipeline()`` with raw/detect/assess compressed
appsinks enabled, attaches metadata probes, and sends frame packets to
``ros_bridge.py`` over local TCP. The shared pipeline owns decode, inference,
tees, JPEG appsinks, and optional display; this file owns source timestamp
capture, bbox/assessment metadata construction, and ``FrameSocketSender``.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from deepstream_yolo.gst_warnings import (
    maybe_start_gst_scan_warning_filter,
    stop_gst_scan_warning_filter,
)

GST_SCAN_WARNING_FILTER = maybe_start_gst_scan_warning_filter(sys.argv)

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import GLib, Gst

import pyds

from deepstream_yolo.assessment_runtime import (
    AssessmentComputeTimes,
    AssessmentLogRow,
    AssessmentTiming,
    assessment_probe,
    compute_fps,
    format_timestamp,
    frame_timestamp,
    label_log_text,
)
from deepstream_yolo.detection_overlay import PERSON_CLASS_ID, bbox_probe, get_detection_id
from deepstream_yolo.frame_wire import send_frame
from deepstream_yolo.model_cache import discover_size, ensure_assessment_model, ensure_model
from deepstream_yolo.paths import DEFAULT_STREAM
from deepstream_yolo.pipeline import build_pipeline, on_message
from deepstream_yolo.stream_source import StreamSource, resolve_stream_source

DEFAULT_DETECT_ENDPOINT = "127.0.0.1:5610"
DEFAULT_ASSESS_ENDPOINT = "127.0.0.1:5611"
DEFAULT_IMAGE_ENDPOINT = "127.0.0.1:5609"
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 368
INT32_MAX = 2_147_483_647
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageSpace:
    source_width: int
    source_height: int
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        for name in ("source_width", "source_height", "image_width", "image_height"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def scale_bbox(self, bbox: BBox) -> list[float]:
        left, top, width, height = bbox
        scale_x = self.image_width / self.source_width
        scale_y = self.image_height / self.source_height
        return [
            left * scale_x,
            top * scale_y,
            width * scale_x,
            height * scale_y,
        ]

    def source_size(self) -> tuple[int, int]:
        return self.source_width, self.source_height

    def image_size(self) -> tuple[int, int]:
        return self.image_width, self.image_height


class RuntimeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        stop_gst_scan_warning_filter(GST_SCAN_WARNING_FILTER)
        super().error(message)


@dataclass(frozen=True)
class SourceTimestamp:
    frame_num: int
    source: str
    timestamp: int | None

    def formatted(self) -> str:
        return format_timestamp(self.source, self.timestamp)


class SourceTimestampStore:
    """Immutable source timestamp lookup keyed by DeepStream frame number."""

    def __init__(self, max_frames: int = 2048):
        self.max_frames = max_frames
        self.timestamps: OrderedDict[int, SourceTimestamp] = OrderedDict()

    def put(self, frame_num: int, source: str, timestamp: int | None) -> SourceTimestamp:
        key = int(frame_num)
        existing = self.timestamps.get(key)
        if existing is not None:
            return existing

        source_timestamp = SourceTimestamp(key, source, timestamp)
        self.timestamps[key] = source_timestamp
        self.timestamps.move_to_end(key)
        while len(self.timestamps) > self.max_frames:
            self.timestamps.popitem(last=False)
        return source_timestamp

    def resolve(
        self,
        frame_num: int,
        fallback_source: str,
        fallback_timestamp: int | None,
    ) -> SourceTimestamp:
        key = int(frame_num)
        source_timestamp = self.timestamps.get(key)
        if source_timestamp is not None:
            return source_timestamp
        return self.put(key, fallback_source, fallback_timestamp)


@dataclass(frozen=True)
class FrameLog:
    stage: str
    frame_num: int
    timestamp_source: str
    timestamp: int | None
    rows: list[str]
    timing_fields: tuple[str, ...] = ()
    objects: list[dict[str, Any]] = field(default_factory=list)
    source_size: tuple[int, int] | None = None
    image_size: tuple[int, int] | None = None

    def metadata(self) -> dict:
        # The ROS bridge treats source_timestamp_ns as the authoritative stamp.
        metadata = {
            "stage": self.stage,
            "frame": self.frame_num,
            "timestamp_ns": self.timestamp,
            "timestamp": format_timestamp(self.timestamp_source, self.timestamp),
            "timestamp_source": self.timestamp_source,
            "source_timestamp_ns": self.timestamp,
            "source_timestamp": format_timestamp(self.timestamp_source, self.timestamp),
            "source_timestamp_source": self.timestamp_source,
            "timestamp_is_source": True,
            "timing": list(self.timing_fields),
            "rows": self.rows,
            "objects": list(self.objects),
            "object_count": len(self.objects),
            "data_source_id": frame_data_source_id(self.frame_num),
            "log_text": self.format(),
        }
        if self.source_size:
            metadata["source_width"], metadata["source_height"] = self.source_size
        if self.image_size:
            metadata["image_width"], metadata["image_height"] = self.image_size
            metadata["bbox_coordinate_space"] = "image"
        return metadata

    def format(self) -> str:
        header = [
            self.stage,
            f"frame={self.frame_num}",
            f"timestamp={format_timestamp(self.timestamp_source, self.timestamp)}",
            f"timestamp_source={self.timestamp_source}",
            *self.timing_fields,
        ]
        if self.source_size:
            header.append(f"source={self.source_size[0]}x{self.source_size[1]}")
        if self.image_size:
            header.append(f"image={self.image_size[0]}x{self.image_size[1]}")
        return "\n".join([" ".join(header), *(f"  {row}" for row in self.rows)])


class FrameLogStore:
    """Small handoff buffer between pad probes and compressed appsinks."""

    def __init__(self, max_frames: int = 512):
        self.max_frames = max_frames
        self.logs: OrderedDict[int, FrameLog] = OrderedDict()
        self.latest: FrameLog | None = None

    def put(self, key: int | None, log: FrameLog) -> None:
        self.latest = log
        if key is None:
            return
        self.logs[key] = log
        self.logs.move_to_end(key)
        while len(self.logs) > self.max_frames:
            self.logs.popitem(last=False)

    def pop(self, key: int | None, allow_latest: bool = True) -> FrameLog | None:
        if key is not None and key in self.logs:
            return self.logs.pop(key)
        if allow_latest:
            return self.latest
        return None


def frame_data_source_id(frame_num: int) -> int:
    return int(frame_num) % INT32_MAX


class FrameSocketSender:
    def __init__(
        self,
        stage: str,
        endpoint: str,
        store: FrameLogStore,
        require_fresh_metadata: bool = False,
    ):
        self.stage = stage
        self.host, self.port = parse_endpoint(endpoint)
        self.store = store
        self.require_fresh_metadata = require_fresh_metadata
        self.sock: socket.socket | None = None
        self.next_connect_time = 0.0

    def on_sample(self, appsink):
        # Appsinks emit already-compressed JPEG buffers; attach the matching metadata and send.
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        if buffer is None:
            return Gst.FlowReturn.OK

        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            print(f"{self.stage} failed to map compressed frame", file=sys.stderr, flush=True)
            return Gst.FlowReturn.OK

        try:
            payload = bytes(mapped.data)
        finally:
            buffer.unmap(mapped)

        log = self.store.pop(
            buffer_key(buffer),
            allow_latest=not self.require_fresh_metadata,
        )
        if log is None and self.require_fresh_metadata:
            return Gst.FlowReturn.OK
        metadata = log.metadata() if log else {"stage": self.stage, "log_text": f"{self.stage} metadata=missing"}
        metadata["format"] = "jpeg"
        metadata["bytes"] = len(payload)

        sock = self.connect()
        if sock is None:
            return Gst.FlowReturn.OK

        try:
            send_frame(sock, metadata, payload)
        except OSError as exc:
            print(f"{self.stage} send failed: {exc}", file=sys.stderr, flush=True)
            self.close()

        return Gst.FlowReturn.OK

    def connect(self) -> socket.socket | None:
        if self.sock:
            return self.sock

        now = time.perf_counter()
        if now < self.next_connect_time:
            return None

        self.next_connect_time = now + 1.0
        try:
            sock = socket.create_connection((self.host, self.port), timeout=0.2)
        except OSError:
            return None

        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock
        print(f"{self.stage} connected endpoint={self.host}:{self.port}", flush=True)
        return sock

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, _, port = endpoint.rpartition(":")
    if not host or not port:
        raise ValueError(f"Expected endpoint HOST:PORT, got {endpoint!r}")
    return host, int(port)


def parse_args() -> argparse.Namespace:
    parser = RuntimeArgumentParser()
    parser.add_argument("--model", default="yolo12x-custom.pt")
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--stream", default=str(DEFAULT_STREAM))
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument(
        "--rtsp-latency-ms",
        type=int,
        default=0,
        help="RTSP jitterbuffer latency before old network packets are dropped; default 0.",
    )
    parser.add_argument("--assessment-model", default="models/injury.pt")
    parser.add_argument("--assessment-batch-size", type=int, default=8)
    parser.add_argument("--output-width", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--output-height", type=int, default=OUTPUT_HEIGHT)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--image-endpoint", default=DEFAULT_IMAGE_ENDPOINT)
    parser.add_argument("--detect-endpoint", default=DEFAULT_DETECT_ENDPOINT)
    parser.add_argument("--assess-endpoint", default=DEFAULT_ASSESS_ENDPOINT)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--show-gst-scan-warnings", action="store_true")
    return parser.parse_args()


def buffer_key(buffer) -> int | None:
    for value in (getattr(buffer, "pts", None), getattr(buffer, "dts", None)):
        if value is None:
            continue
        timestamp = int(value)
        if timestamp != Gst.CLOCK_TIME_NONE and timestamp >= 0:
            return timestamp
    return None


def timing_fields(compute_times: AssessmentComputeTimes | None) -> tuple[str, ...]:
    if compute_times is None:
        return ()

    fields = []
    detect_fps = compute_fps(compute_times.detect_ms)
    assess_fps = compute_fps(compute_times.assess_ms)
    if compute_times.detect_ms is not None:
        fields.append(f"detect_ms={compute_times.detect_ms:.2f}")
    if detect_fps is not None:
        fields.append(f"detect_fps={detect_fps:.2f}")
    if compute_times.assess_ms is not None:
        fields.append(f"assess_ms={compute_times.assess_ms:.2f}")
    if assess_fps is not None:
        fields.append(f"assess_fps={assess_fps:.2f}")
    return tuple(fields)


def bbox_values(rect) -> BBox:
    return (
        float(rect.left),
        float(rect.top),
        float(rect.width),
        float(rect.height),
    )


def serializable_predictions(predictions: dict[str, dict]) -> dict[str, dict]:
    serialized = {}
    for name, prediction in predictions.items():
        serialized[name] = {
            "class_id": int(prediction.get("class_id", -1)),
            "confidence": float(prediction.get("confidence", 0.0)),
            "probabilities": [
                float(value)
                for value in prediction.get("probabilities", [])
            ],
        }
    return serialized


def source_timestamp_probe(timestamps: SourceTimestampStore):
    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            # Capture the network/source timestamp before branch-local buffers can diverge.
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            timestamp_source, timestamp = frame_timestamp(frame_meta, buffer)
            timestamps.put(int(frame_meta.frame_num), timestamp_source, timestamp)
            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe


def image_metadata_probe(
    store: FrameLogStore,
    image_space: ImageSpace,
    timestamps: SourceTimestampStore,
):
    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        key = buffer_key(buffer)
        frame_list = batch_meta.frame_meta_list
        while frame_list:
            # Raw images carry no object rows but still need source-frame identity.
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            fallback_source, fallback_timestamp = frame_timestamp(frame_meta, buffer)
            frame_num = int(frame_meta.frame_num)
            source_timestamp = timestamps.resolve(frame_num, fallback_source, fallback_timestamp)
            store.put(
                key,
                FrameLog(
                    "IMAGE",
                    frame_num,
                    source_timestamp.source,
                    source_timestamp.timestamp,
                    [],
                    (),
                    [],
                    image_space.source_size(),
                    image_space.image_size(),
                ),
            )
            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe


def detect_metadata_probe(
    store: FrameLogStore,
    timing: AssessmentTiming,
    image_space: ImageSpace,
    timestamps: SourceTimestampStore,
):
    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        key = buffer_key(buffer)
        frame_list = batch_meta.frame_meta_list
        while frame_list:
            # Detection metadata is emitted in the compressed image coordinate space.
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            fallback_source, fallback_timestamp = frame_timestamp(frame_meta, buffer)
            frame_num = int(frame_meta.frame_num)
            source_timestamp = timestamps.resolve(frame_num, fallback_source, fallback_timestamp)
            rows = []
            objects = []
            person_index = 0

            obj_list = frame_meta.obj_meta_list
            while obj_list:
                obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                if obj.class_id == PERSON_CLASS_ID:
                    object_index = len(objects)
                    fallback_id = person_index
                    person_index += 1
                    object_id = get_detection_id(obj, fallback_id)
                    rect = obj.rect_params
                    source_bbox = bbox_values(rect)
                    bbox = image_space.scale_bbox(source_bbox)
                    confidence = float(obj.confidence)
                    rows.append(
                        "object="
                        f"{object_id} "
                        f"bbox={bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f} "
                        f"person_conf={confidence:.2f}"
                    )
                    objects.append(
                        {
                            "object_index": object_index,
                            "object_id": int(object_id),
                            "bbox": bbox,
                            "source_bbox": list(source_bbox),
                            "class_name": "person",
                            "confidence": confidence,
                        }
                    )
                obj_list = obj_list.next

            detect_ms = timing.detect_compute_ms(frame_num)
            fields = []
            detect_fps = compute_fps(detect_ms)
            if detect_ms is not None:
                fields.append(f"detect_ms={detect_ms:.2f}")
            if detect_fps is not None:
                fields.append(f"detect_fps={detect_fps:.2f}")
            store.put(
                key,
                FrameLog(
                    "DETECT",
                    frame_num,
                    source_timestamp.source,
                    source_timestamp.timestamp,
                    rows,
                    tuple(fields),
                    objects,
                    image_space.source_size(),
                    image_space.image_size(),
                ),
            )
            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe


def assessment_frame_sink(
    store: FrameLogStore,
    image_space: ImageSpace,
    timestamps: SourceTimestampStore,
):
    def _sink(
        buffer,
        frame_num: int,
        timestamp_source: str,
        timestamp: int | None,
        rows: list[AssessmentLogRow],
        compute_times: AssessmentComputeTimes | None,
    ) -> None:
        log_rows = []
        objects = []
        # Assessment rows inherit the immutable timestamp captured before inference.
        source_timestamp = timestamps.resolve(frame_num, timestamp_source, timestamp)
        for row in rows:
            object_index = len(objects)
            source_bbox = tuple(float(value) for value in row.bbox)
            bbox = image_space.scale_bbox(source_bbox)
            left, top, width, height = bbox
            log_rows.append(
                "object="
                f"{row.object_id} "
                f"bbox={left:.0f},{top:.0f},{width:.0f},{height:.0f} "
                f"{label_log_text(row.lines)}"
            )
            objects.append(
                {
                    "object_index": object_index,
                    "object_id": int(row.object_id),
                    "bbox": bbox,
                    "source_bbox": list(source_bbox),
                    "class_name": "person",
                    "labels": list(row.lines),
                    "predictions": serializable_predictions(row.predictions),
                }
            )
        store.put(
            buffer_key(buffer),
            FrameLog(
                "ASSESS",
                frame_num,
                source_timestamp.source,
                source_timestamp.timestamp,
                log_rows,
                timing_fields(compute_times),
                objects,
                image_space.source_size(),
                image_space.image_size(),
            ),
        )

    return _sink


def print_runtime_info(
    stream: StreamSource,
    src_w: int,
    src_h: int,
    model_w: int,
    model_h: int,
    conf: float,
    config,
    assessment_meta: dict,
    assessment_config,
    args: argparse.Namespace,
) -> None:
    print(
        f"stream={stream.display} "
        f"video={src_w}x{src_h} "
        f"model={model_w}x{model_h} "
        f"conf={conf} "
        f"config={config}"
    )
    print(
        "assessment="
        f"{assessment_meta.get('architecture', 'injury model')} "
        f"batch={args.assessment_batch_size} "
        f"config={assessment_config}"
    )
    print(
        "frame_outputs="
        f"image={args.image_endpoint} "
        f"detect={args.detect_endpoint} "
        f"assess={args.assess_endpoint} "
        f"size={args.output_width}x{args.output_height} "
        f"jpeg_quality={args.jpeg_quality}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    stream = resolve_stream_source(args.stream)

    # Prepare source geometry and model configs before assembling the GStreamer graph.
    try:
        Gst.init(None)
        src_w, src_h = discover_size(stream.uri)
    finally:
        stop_gst_scan_warning_filter(GST_SCAN_WARNING_FILTER)

    model_w, model_h, config = ensure_model(args.model, stream, args.long_side, src_w, src_h, args.conf)
    assessment_meta, assessment_config = ensure_assessment_model(args.assessment_model, args.assessment_batch_size)
    print_runtime_info(stream, src_w, src_h, model_w, model_h, args.conf, config, assessment_meta, assessment_config, args)
    output_size = (args.output_width, args.output_height)
    image_space = ImageSpace(src_w, src_h, *output_size)

    image_store = FrameLogStore()
    detect_store = FrameLogStore()
    assess_store = FrameLogStore()
    source_timestamps = SourceTimestampStore()
    # Each sender owns one compressed branch and one TCP endpoint.
    image_sender = FrameSocketSender(
        "IMAGE",
        args.image_endpoint,
        image_store,
        require_fresh_metadata=True,
    )
    detect_sender = FrameSocketSender(
        "DETECT",
        args.detect_endpoint,
        detect_store,
        require_fresh_metadata=True,
    )
    assess_sender = FrameSocketSender(
        "ASSESS",
        args.assess_endpoint,
        assess_store,
        require_fresh_metadata=True,
    )

    parts = build_pipeline(
        stream,
        src_w,
        src_h,
        config,
        assessment_config,
        rtsp_latency_ms=args.rtsp_latency_ms,
        display=args.display,
        raw_output_size=output_size,
        detect_output_size=output_size,
        assess_output_size=output_size,
        jpeg_quality=args.jpeg_quality,
    )
    if not parts.raw_appsink or not parts.detect_appsink or not parts.assess_appsink:
        raise RuntimeError("Frame source pipeline did not create all appsinks")

    # Probes produce metadata; appsinks produce JPEG payloads; FrameLogStore joins them.
    timing = AssessmentTiming()
    streammux_src = parts.streammux.get_static_pad("src")
    streammux_src.add_probe(Gst.PadProbeType.BUFFER, source_timestamp_probe(source_timestamps), None)
    streammux_src.add_probe(Gst.PadProbeType.BUFFER, timing.mark_start, None)
    streammux_src.add_probe(
        Gst.PadProbeType.BUFFER,
        image_metadata_probe(image_store, image_space, source_timestamps),
        None,
    )
    parts.pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, timing.mark_detect_done, None)
    parts.pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, bbox_probe(args.conf), None)
    parts.pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        detect_metadata_probe(detect_store, timing, image_space, source_timestamps),
        None,
    )
    parts.sgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        assessment_probe(
            None,
            timing=timing,
            frame_sink=assessment_frame_sink(assess_store, image_space, source_timestamps),
        ),
        None,
    )
    parts.raw_appsink.connect("new-sample", image_sender.on_sample)
    parts.detect_appsink.connect("new-sample", detect_sender.on_sample)
    parts.assess_appsink.connect("new-sample", assess_sender.on_sample)

    loop = GLib.MainLoop()
    bus = parts.pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    parts.pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        parts.pipeline.set_state(Gst.State.NULL)
        image_sender.close()
        detect_sender.close()
        assess_sender.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
