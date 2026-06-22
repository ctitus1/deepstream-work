import ctypes
import datetime as dt
import math
import struct
import time

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


class AssessmentReporter:
    def __init__(self, interval: float):
        self.interval = max(0.0, interval)
        self.last_log_time = 0.0
        self.logging_frame = None

    def should_log(self, frame_num: int) -> bool:
        if self.interval == 0:
            return False

        if self.logging_frame == frame_num:
            return True

        now = time.perf_counter()
        if now - self.last_log_time < self.interval:
            return False

        self.last_log_time = now
        self.logging_frame = frame_num
        return True

    def maybe_log(
        self,
        frame_num: int,
        object_id: int,
        obj,
        timestamp_source: str,
        timestamp: int | None,
        lines: list[str],
    ) -> None:
        if not self.should_log(frame_num):
            return

        rect = obj.rect_params
        print(
            "ASSESS "
            f"frame={frame_num} "
            f"timestamp={format_timestamp(timestamp_source, timestamp)} "
            f"timestamp_source={timestamp_source} "
            f"object={object_id} "
            f"bbox={rect.left:.0f},{rect.top:.0f},{rect.width:.0f},{rect.height:.0f} "
            f"{label_log_text(lines)}",
            flush=True,
        )


def assessment_probe(reporter: AssessmentReporter):
    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            obj_list = frame_meta.obj_meta_list
            person_index = 0
            timestamp_source, timestamp = frame_timestamp(frame_meta, buffer)

            while obj_list:
                obj = pyds.NvDsObjectMeta.cast(obj_list.data)
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
                                lines = label_text(
                                    box_id,
                                    "person",
                                    predictions,
                                )
                                set_assessment_text(obj, lines)
                                reporter.maybe_log(
                                    int(frame_meta.frame_num),
                                    box_id,
                                    obj,
                                    timestamp_source,
                                    timestamp,
                                    lines,
                                )

                    user_meta_list = user_meta_list.next

                obj_list = obj_list.next

            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe
