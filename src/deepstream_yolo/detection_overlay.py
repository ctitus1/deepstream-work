"""Detection overlay probe helpers.

``parser_app.py`` and ``ros_source.py`` attach ``bbox_probe()`` at the PGIE
output. The probe assigns stable per-frame person ids, clears the default
DeepStream label, and draws the colored person-confidence box used by the
display pipeline and downstream metadata.
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds

PERSON_CLASS_ID = 0
DETECTION_ID_MISC_INDEX = 0
DETECTION_ID_OFFSET = 1


def set_detection_id(obj, detection_id: int) -> None:
    try:
        obj.misc_obj_info[DETECTION_ID_MISC_INDEX] = int(detection_id) + DETECTION_ID_OFFSET
    except Exception:
        pass


def get_detection_id(obj, fallback: int) -> int:
    try:
        stored_id = int(obj.misc_obj_info[DETECTION_ID_MISC_INDEX])
    except Exception:
        return int(fallback)

    if stored_id >= DETECTION_ID_OFFSET:
        return stored_id - DETECTION_ID_OFFSET
    return int(fallback)


def conf_color(confidence: float, lower_conf: float):
    mid_conf = (lower_conf + 0.8) / 2.0

    if confidence <= lower_conf:
        return 1.0, 0.0, 0.0, 1.0

    if confidence < mid_conf:
        t = (confidence - lower_conf) / max(mid_conf - lower_conf, 1e-6)
        return 1.0, t, 0.0, 1.0

    if confidence < 0.8:
        t = (confidence - mid_conf) / max(0.8 - mid_conf, 1e-6)
        return 1.0 - t, 1.0, 0.0, 1.0

    return 0.0, 1.0, 0.0, 1.0


def add_line_box(batch_meta, frame_meta, left, top, width, height, label, color) -> None:
    x1, y1 = round(left), round(top)
    x2, y2 = round(left + width), round(top + height)

    frame_h = int(getattr(frame_meta, "source_frame_height", 0) or 1080)
    font_size = max(1, round(frame_h * 0.001))
    line_width = 3

    meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    meta.num_lines = 4
    meta.num_labels = 1

    for line, coords in zip(
        meta.line_params,
        (
            (x1, y1, x2, y1),
            (x2, y1, x2, y2),
            (x2, y2, x1, y2),
            (x1, y2, x1, y1),
        ),
    ):
        line.x1, line.y1, line.x2, line.y2 = coords
        line.line_width = line_width
        line.line_color.set(*color)

    text = meta.text_params[0]
    text.display_text = label
    text.x_offset = max(0, x1 - line_width // 2)
    text_height = max(18, 2 * font_size + 8)
    text.y_offset = max(0, y1 - text_height)
    text.font_params.font_name = "Serif"
    text.font_params.font_size = font_size
    text.font_params.font_color.set(*color)
    text.set_bg_clr = 1
    text.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

    pyds.nvds_add_display_meta_to_frame(frame_meta, meta)


def bbox_probe(conf: float):
    def _probe(_pad, info, _data):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list

        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            obj_list = frame_meta.obj_meta_list
            person_index = 0

            while obj_list:
                obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                rect = obj.rect_params
                rect.border_width = 0
                obj.text_params.display_text = ""

                if obj.class_id == PERSON_CLASS_ID:
                    set_detection_id(obj, person_index)
                    person_index += 1
                    add_line_box(
                        batch_meta,
                        frame_meta,
                        rect.left,
                        rect.top,
                        rect.width,
                        rect.height,
                        f"person {obj.confidence:.2f}",
                        conf_color(float(obj.confidence), conf),
                    )

                obj_list = obj_list.next

            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe
