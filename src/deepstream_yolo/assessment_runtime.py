import ctypes
import datetime as dt
import math
import struct
import time
from dataclasses import dataclass

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds

from .configs import INJURY_CLASS_COUNTS, INJURY_HEADS
from .detection_overlay import PERSON_CLASS_ID, get_detection_id

ASSESSMENT_GIE_ID = 2
NTP_TO_UNIX_SECONDS = 2_208_988_800
NS_PER_SEC = 1_000_000_000
ASSESSMENT_LABELS = {
    "severe_hemorrhage": {0: "hem-", 1: "hem+", 2: "hem?"},
    "respiratory_distress": {0: "resp-", 1: "resp+", 2: "resp?"},
    "trauma_head": {0: "head-", 1: "head+", 3: "head?"},
    "trauma_torso": {0: "torso-", 1: "torso+", 3: "torso?"},
    "trauma_upper_ext": {0: "upper-", 1: "upper+", 2: "upper_amp", 4: "upper?"},
    "trauma_lower_ext": {0: "lower-", 1: "lower+", 2: "lower_amp", 4: "lower?"},
    "alertness_ocular": {0: "eyes_open", 1: "eyes_closed", 2: "eyes_nt", 3: "eyes?"},
    "person_type": {0: "manikin", 1: "human", 2: "type?"},
}
ASSESSMENT_DISPLAY_ROWS = (
    ("person_type", "severe_hemorrhage", "respiratory_distress"),
    ("trauma_head", "trauma_torso"),
    ("trauma_upper_ext", "trauma_lower_ext", "alertness_ocular"),
)


def dims_num_elements(dims) -> int:
    count = int(getattr(dims, "numElements", 0) or 0)
    if count > 0:
        return count

    count = 1
    for idx in range(int(dims.numDims)):
        count *= int(dims.d[idx])
    return count


def ptr_value(buffer) -> int:
    try:
        return int(pyds.get_ptr(buffer))
    except TypeError:
        return int(buffer)


def half_to_float(value: int) -> float:
    return struct.unpack("<e", struct.pack("<H", int(value)))[0]


def tensor_values(layer, tensor_meta, index: int) -> list[float]:
    try:
        layer.buffer = tensor_meta.out_buf_ptrs_host[index]
    except Exception:
        pass

    count = dims_num_elements(layer.inferDims)
    if count <= 0:
        return []

    address = ptr_value(layer.buffer)
    if not address:
        return []

    if layer.dataType == pyds.NvDsInferDataType.FLOAT:
        ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_float))
        return [float(ptr[i]) for i in range(count)]
    if layer.dataType == pyds.NvDsInferDataType.HALF:
        ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_uint16))
        return [half_to_float(ptr[i]) for i in range(count)]
    if layer.dataType == pyds.NvDsInferDataType.INT32:
        ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_int32))
        return [float(ptr[i]) for i in range(count)]
    if layer.dataType == pyds.NvDsInferDataType.INT8:
        ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_int8))
        return [float(ptr[i]) for i in range(count)]

    return []


def layer_name(layer, fallback: str) -> str:
    name = getattr(layer, "layerName", "") or ""
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return str(name) or fallback


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def parse_assessment_tensor_meta(tensor_meta) -> dict[str, dict]:
    predictions = {}

    for index in range(int(tensor_meta.num_output_layers)):
        fallback_name = INJURY_HEADS[index] if index < len(INJURY_HEADS) else f"output_{index}"
        layer = pyds.get_nvds_LayerInfo(tensor_meta, index)
        head_name = layer_name(layer, fallback_name)
        if head_name not in INJURY_CLASS_COUNTS:
            head_name = fallback_name
        if head_name not in INJURY_CLASS_COUNTS:
            continue

        logits = tensor_values(layer, tensor_meta, index)
        expected = INJURY_CLASS_COUNTS[head_name]
        logits = logits[:expected]
        probs = softmax(logits)
        if not probs:
            continue

        class_id = max(range(len(probs)), key=lambda idx: probs[idx])
        predictions[head_name] = {
            "class_id": class_id,
            "confidence": probs[class_id],
            "probabilities": probs,
        }

    return predictions


def injury_label(name: str, predictions: dict[str, dict]) -> str:
    pred = predictions.get(name)
    if not pred:
        return ""

    class_id = int(pred["class_id"])
    return ASSESSMENT_LABELS.get(name, {}).get(class_id, f"{name}=class{class_id}")


def label_text(box_id: int, yolo_label: str, predictions: dict[str, dict]) -> list[str]:
    lines = [f"{yolo_label} {box_id} injuries:"]

    for row in ASSESSMENT_DISPLAY_ROWS:
        labels = [injury_label(name, predictions) for name in row]
        labels = [label for label in labels if label]
        if labels:
            lines.append("  ".join(labels))

    return lines


def label_log_text(lines: list[str]) -> str:
    return " | ".join(lines)


def valid_clock_time(value, allow_zero: bool) -> int | None:
    if value is None:
        return None

    timestamp = int(value)
    if timestamp == Gst.CLOCK_TIME_NONE or timestamp < 0:
        return None
    if timestamp == 0 and not allow_zero:
        return None
    return timestamp


def network_ns_to_unix_ns(timestamp: int) -> int:
    ntp_offset_ns = NTP_TO_UNIX_SECONDS * NS_PER_SEC
    if timestamp > ntp_offset_ns:
        return timestamp - ntp_offset_ns
    return timestamp


def unix_ns_to_utc(timestamp: int) -> str:
    try:
        return (
            dt.datetime.fromtimestamp(timestamp / NS_PER_SEC, tz=dt.timezone.utc)
            .strftime("%H:%M:%S.%f")[:-3]
            + "Z"
        )
    except (OverflowError, OSError, ValueError):
        return f"{timestamp / NS_PER_SEC:.3f}s"


def seconds(timestamp: int) -> str:
    return f"{timestamp / NS_PER_SEC:.3f}s"


def reference_timestamp(buffer) -> int | None:
    getter = getattr(buffer, "get_reference_timestamp_meta", None)
    if getter is None:
        return None

    try:
        ref_meta = getter(None)
    except Exception:
        return None

    if ref_meta is None:
        return None

    timestamp = valid_clock_time(getattr(ref_meta, "timestamp", None), allow_zero=False)
    if timestamp is None:
        return None
    return network_ns_to_unix_ns(timestamp)


def frame_timestamp(frame_meta, buffer) -> tuple[str, int | None]:
    timestamp = valid_clock_time(getattr(frame_meta, "ntp_timestamp", None), allow_zero=False)
    if timestamp is not None:
        return "ntp", network_ns_to_unix_ns(timestamp)

    timestamp = reference_timestamp(buffer)
    if timestamp is not None:
        return "ref", timestamp

    timestamp = valid_clock_time(getattr(frame_meta, "buf_pts", None), allow_zero=True)
    if timestamp is not None:
        return "buf_pts", timestamp

    timestamp = valid_clock_time(getattr(buffer, "pts", None), allow_zero=True)
    if timestamp is not None:
        return "pts", timestamp

    return "none", None


def format_timestamp(source: str, timestamp: int | None) -> str:
    if timestamp is None:
        return "NONE"
    if source in {"ntp", "ref"}:
        return unix_ns_to_utc(timestamp)
    return seconds(timestamp)


def set_assessment_text(obj, lines: list[str]) -> None:
    rect = obj.rect_params
    obj.text_params.display_text = "\n".join(lines)
    obj.text_params.x_offset = max(0, round(rect.left))
    obj.text_params.y_offset = max(0, round(rect.top + rect.height + 8))
    obj.text_params.font_params.font_name = "Serif"
    obj.text_params.font_params.font_size = 12
    obj.text_params.font_params.font_color.set(0.2, 0.9, 1.0, 1.0)
    obj.text_params.set_bg_clr = 1
    obj.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)


def clear_assessment_text(obj) -> None:
    obj.text_params.display_text = ""
    obj.text_params.set_bg_clr = 0


@dataclass(frozen=True)
class AssessmentLogRow:
    object_id: int
    bbox: tuple[float, float, float, float]
    lines: list[str]
    predictions: dict[str, dict]


@dataclass(frozen=True)
class AssessmentComputeTimes:
    detect_ms: float | None
    assess_ms: float | None


def compute_fps(milliseconds: float | None) -> float | None:
    if milliseconds is None or milliseconds <= 0:
        return None
    return 1000.0 / milliseconds


class AssessmentTiming:
    def __init__(self, max_frames: int = 2048):
        self.max_frames = max_frames
        self.start_times: dict[int, float] = {}
        self.detect_done_times: dict[int, float] = {}

    def mark_start(self, _pad, info, _data):
        return self._mark(info, self.start_times)

    def mark_detect_done(self, _pad, info, _data):
        return self._mark(info, self.detect_done_times)

    def _mark(self, info, times: dict[int, float]):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        now = time.perf_counter()
        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            times[int(frame_meta.frame_num)] = now
            frame_list = frame_list.next

        self._trim(times)

        return Gst.PadProbeReturn.OK

    def _trim(self, times: dict[int, float]) -> None:
        while len(times) > self.max_frames:
            times.pop(next(iter(times)))

    def detect_compute_ms(self, frame_num: int) -> float | None:
        start_time = self.start_times.get(frame_num)
        detect_done_time = self.detect_done_times.get(frame_num)
        if start_time is None or detect_done_time is None:
            return None
        return max(0.0, (detect_done_time - start_time) * 1000.0)

    def pop_compute_times(self, frame_num: int, now: float) -> AssessmentComputeTimes:
        start_time = self.start_times.pop(frame_num, None)
        detect_done_time = self.detect_done_times.pop(frame_num, None)

        detect_ms = None
        if start_time is not None and detect_done_time is not None:
            detect_ms = max(0.0, (detect_done_time - start_time) * 1000.0)

        assess_ms = None
        if detect_done_time is not None:
            assess_ms = max(0.0, (now - detect_done_time) * 1000.0)

        return AssessmentComputeTimes(detect_ms=detect_ms, assess_ms=assess_ms)


class AssessmentReporter:
    def __init__(self, interval: float):
        self.interval = interval
        self.last_log_time = 0.0
        self.logging_frame = None

    def should_log(self, frame_num: int, now: float) -> bool:
        if self.interval < 0:
            return False

        if self.interval == 0:
            return True

        if self.logging_frame == frame_num:
            return True

        if now - self.last_log_time < self.interval:
            return False

        self.last_log_time = now
        self.logging_frame = frame_num
        return True

    def log_frame(
        self,
        frame_num: int,
        timestamp_source: str,
        timestamp: int | None,
        rows: list[AssessmentLogRow],
        compute_times: AssessmentComputeTimes | None,
    ) -> None:
        if not rows:
            return

        now = time.perf_counter()
        if not self.should_log(frame_num, now):
            return

        fields = [
            "ASSESS",
            f"frame={frame_num}",
            f"timestamp={format_timestamp(timestamp_source, timestamp)}",
            f"timestamp_source={timestamp_source}",
        ]
        if compute_times:
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

        log_lines = [" ".join(fields)]
        for row in rows:
            left, top, width, height = row.bbox
            log_lines.append(
                "  "
                f"object={row.object_id} "
                f"bbox={left:.0f},{top:.0f},{width:.0f},{height:.0f} "
                f"{label_log_text(row.lines)}"
            )

        print("\n".join(log_lines), flush=True)


def assessment_probe(
    reporter: AssessmentReporter | None,
    timing: AssessmentTiming | None = None,
    show_assessed_only: bool = False,
    frame_sink=None,
):
    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        buffer_has_assessment = False
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            obj_list = frame_meta.obj_meta_list
            person_index = 0
            timestamp_source, timestamp = frame_timestamp(frame_meta, buffer)
            frame_num = int(frame_meta.frame_num)
            frame_rows = []

            while obj_list:
                obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                clear_assessment_text(obj)
                user_meta_list = obj.obj_user_meta_list
                fallback_id = person_index
                if obj.class_id == PERSON_CLASS_ID:
                    person_index += 1
                box_id = get_detection_id(obj, fallback_id)

                while user_meta_list:
                    user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                    if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        if int(tensor_meta.unique_id) == ASSESSMENT_GIE_ID:
                            predictions = parse_assessment_tensor_meta(tensor_meta)
                            if predictions:
                                buffer_has_assessment = True
                                lines = label_text(
                                    box_id,
                                    "person",
                                    predictions,
                                )
                                set_assessment_text(obj, lines)
                                rect = obj.rect_params
                                frame_rows.append(
                                    AssessmentLogRow(
                                        object_id=box_id,
                                        bbox=(rect.left, rect.top, rect.width, rect.height),
                                        lines=lines,
                                        predictions=predictions,
                                    )
                                )

                    user_meta_list = user_meta_list.next

                obj_list = obj_list.next

            if frame_rows:
                now = time.perf_counter()
                compute_times = timing.pop_compute_times(frame_num, now) if timing else None
                if reporter:
                    reporter.log_frame(
                        frame_num,
                        timestamp_source,
                        timestamp,
                        frame_rows,
                        compute_times,
                    )
                if frame_sink:
                    frame_sink(
                        buffer,
                        frame_num,
                        timestamp_source,
                        timestamp,
                        frame_rows,
                        compute_times,
                    )

            frame_list = frame_list.next

        if show_assessed_only and not buffer_has_assessment:
            return Gst.PadProbeReturn.DROP

        return Gst.PadProbeReturn.OK

    return _probe
