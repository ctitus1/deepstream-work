"""COCO category-id mapping shared by the dump and score tools.

Dependency-free on purpose: ``dump_detections.py`` runs under the DeepStream
interpreter and ``score_detections.py`` runs under ``.venv-yolo``.

Contiguous model class ids (0..N-1) are NOT COCO category ids. Ultralytics
silently applies ``coco80_to_coco91_class()`` when it decides a dataset "is
COCO", which is where most category-id mismatches come from. Here the mapping
is an explicit, named choice.
"""

from __future__ import annotations

# ultralytics coco80_to_coco91_class(): contiguous index -> 91-class COCO id.
COCO80_TO_COCO91 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)

CATEGORY_MAPS = ("coco91", "identity", "offset1")
CATEGORY_MAP_HELP = (
    "coco91: contiguous 0..79 -> 91-class COCO ids (ultralytics default for COCO); "
    "identity: model class id unchanged; "
    "offset1: model class id + 1 (contiguous 1-based category ids)"
)


def map_category_id(class_id: int, category_map: str) -> int:
    if category_map == "identity":
        return int(class_id)
    if category_map == "offset1":
        return int(class_id) + 1
    if category_map == "coco91":
        if not 0 <= int(class_id) < len(COCO80_TO_COCO91):
            raise ValueError(
                f"class_id {class_id} is outside the 80-class COCO range; "
                "use --category-map identity for a non-COCO model"
            )
        return COCO80_TO_COCO91[int(class_id)]
    raise ValueError(f"Unknown category map {category_map!r}; expected one of {', '.join(CATEGORY_MAPS)}")


def describe_category_map(category_map: str) -> str:
    if category_map == "coco91":
        return "coco91 (class 0 -> category_id 1 'person', class 79 -> 90)"
    if category_map == "offset1":
        return "offset1 (class 0 -> category_id 1)"
    return "identity (class 0 -> category_id 0)"


def remap_category_ids(detections: list, source_map: str, target_map: str) -> list:
    """Re-map already-dumped category ids without re-running the pipeline."""
    if source_map == target_map:
        return detections

    inverse = {}
    if source_map == "coco91":
        inverse = {value: index for index, value in enumerate(COCO80_TO_COCO91)}

    remapped = []
    for detection in detections:
        category_id = int(detection["category_id"])
        if source_map == "identity":
            class_id = category_id
        elif source_map == "offset1":
            class_id = category_id - 1
        else:
            if category_id not in inverse:
                raise ValueError(f"category_id {category_id} is not a 91-class COCO id")
            class_id = inverse[category_id]
        entry = dict(detection)
        entry["category_id"] = map_category_id(class_id, target_map)
        remapped.append(entry)
    return remapped
