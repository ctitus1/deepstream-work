#!/usr/bin/env python3
"""Score dumped DeepStream detections against COCO ground truth.

Reads the JSON written by ``dump_detections.py`` and reports mAP50, mAP50-95,
and precision/recall at a stated operating point. ``pycocotools`` lives in
``.venv-yolo``, so it is imported inside the scoring function and this file
never imports gi/pyds - run it with the export interpreter.

The report always prints the thresholds the detections were produced under: a
deployed 0.25 confidence floor truncates the precision-recall curve, so the
resulting mAP is not comparable to an ultralytics baseline run at conf=0.001.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepstream_yolo.paths import YOLO_PYTHON, resolve_project_path  # noqa: E402
from coco_categories import CATEGORY_MAPS, remap_category_ids  # noqa: E402

ULTRALYTICS_VAL_CONF = 0.001
ULTRALYTICS_VAL_IOU = 0.7


def import_pycocotools():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "Missing pycocotools. Run this scorer with the export environment:\n"
            f"  {YOLO_PYTHON} validation/accuracy/score_detections.py ...\n"
            f"and install it there once:\n  {YOLO_PYTHON} -m pip install pycocotools"
        ) from exc
    return COCO, COCOeval


def load_dump(path: Path) -> tuple:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and "detections" in data:
        return data["detections"], data.get("info", {})
    raise ValueError(f"{path} is neither a COCO results array nor a dump_detections.py file")


def id_type_name(value) -> str:
    return type(value).__name__


def check_alignment(detections: list, coco_gt, categories: list) -> list:
    """image_id typing and category-id mismatches are the two silent zero-mAP causes."""
    problems = []
    if not detections:
        problems.append("dump contains no detections")
        return problems

    gt_image_ids = set(coco_gt.getImgIds())
    dt_image_ids = {detection["image_id"] for detection in detections}
    gt_types = {id_type_name(value) for value in gt_image_ids}
    dt_types = {id_type_name(value) for value in dt_image_ids}

    if len(dt_types) > 1:
        problems.append(f"mixed image_id types in the dump: {sorted(dt_types)}")
    if gt_types and dt_types and not (gt_types & dt_types):
        problems.append(
            f"image_id type mismatch: ground truth uses {sorted(gt_types)}, dump uses {sorted(dt_types)}. "
            "pycocotools matches on value AND type; re-dump with --image-id-type "
            f"{'int' if 'int' in gt_types else 'str'}"
        )

    overlap = dt_image_ids & gt_image_ids
    if not overlap:
        sample_dt = sorted(str(value) for value in dt_image_ids)[:3]
        sample_gt = sorted(str(value) for value in gt_image_ids)[:3]
        problems.append(
            f"no image_id overlap: dump ids {sample_dt}... vs ground-truth ids {sample_gt}.... "
            "Use --image-id-mode map with --image-id-map, or --image-id-offset, when dumping"
        )

    gt_categories = set(coco_gt.getCatIds())
    dt_categories = {detection["category_id"] for detection in detections}
    unknown = sorted(dt_categories - gt_categories)
    if unknown:
        problems.append(
            f"category_ids {unknown[:8]} are absent from the ground truth. Contiguous model class ids "
            "are not COCO category ids; re-dump or re-map with --category-map coco91/identity"
        )
    missing = sorted(set(categories) - dt_categories)
    if missing:
        problems.append(f"no detections for scored categories {missing[:8]} (AP for those will be 0)")
    return problems


def precision_recall(coco_eval, iou_threshold: float, score_threshold: float) -> dict:
    """Greedy TP/FP/FN at one IoU and one score floor, from COCOeval match state."""
    import numpy as np

    params = coco_eval.params
    iou_index = int(np.argmin(np.abs(np.asarray(params.iouThrs) - iou_threshold)))
    num_area = len(params.areaRng)
    num_img = len(params.imgIds)

    true_positives = 0
    false_positives = 0
    ground_truths = 0

    for cat_index in range(len(params.catIds)):
        for img_index in range(num_img):
            # evalImgs is ordered catId -> areaRng -> imgId; area index 0 is 'all'.
            entry = coco_eval.evalImgs[cat_index * num_area * num_img + img_index]
            if entry is None:
                continue

            gt_ignore = np.asarray(entry["gtIgnore"]).reshape(-1)
            ground_truths += int(np.count_nonzero(gt_ignore == 0))

            scores = np.asarray(entry["dtScores"]).reshape(-1)
            if scores.size == 0:
                continue
            matches = np.asarray(entry["dtMatches"])[iou_index]
            ignored = np.asarray(entry["dtIgnore"])[iou_index]
            keep = (scores >= score_threshold) & (ignored == 0)
            true_positives += int(np.count_nonzero(keep & (matches > 0)))
            false_positives += int(np.count_nonzero(keep & (matches == 0)))

    detected = true_positives + false_positives
    false_negatives = max(0, ground_truths - true_positives)
    return {
        "iou": iou_threshold,
        "score_threshold": score_threshold,
        "precision": true_positives / detected if detected else 0.0,
        "recall": true_positives / ground_truths if ground_truths else 0.0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ground_truths": ground_truths,
    }


def resolve_categories(coco_gt, selection: str, info: dict) -> list:
    if selection not in {"auto", "all"}:
        return [int(value) for value in selection.replace(" ", "").split(",") if value]
    if selection == "all":
        return sorted(coco_gt.getCatIds())

    class_filter = str(info.get("class_ids", "all"))
    if class_filter and class_filter != "all":
        from coco_categories import map_category_id

        category_map = info.get("category_map", "coco91")
        return sorted(
            map_category_id(int(value), category_map)
            for value in class_filter.replace(" ", "").split(",")
            if value
        )
    return sorted(coco_gt.getCatIds())


def threshold_report(info: dict, score_threshold: float) -> list:
    thresholds = info.get("thresholds", {})
    pre_cluster = thresholds.get("pre_cluster_threshold")
    nms_iou = thresholds.get("nms_iou_threshold")
    lines = [
        "thresholds  "
        f"deploy pre-cluster={pre_cluster} nms-iou={nms_iou} topk={thresholds.get('topk')} | "
        f"eval score floor={score_threshold} | "
        f"ultralytics val conf={ULTRALYTICS_VAL_CONF} iou={ULTRALYTICS_VAL_IOU}"
    ]
    if pre_cluster is not None and pre_cluster > ULTRALYTICS_VAL_CONF:
        lines.append(
            f"            NOTE: detections were truncated at score>={pre_cluster} by nvinfer, so the "
            "precision-recall curve has no low-confidence tail. mAP here is a deployment number and "
            f"will read far below a PyTorch baseline run at conf={ULTRALYTICS_VAL_CONF}."
        )
    if nms_iou is not None and abs(nms_iou - ULTRALYTICS_VAL_IOU) > 1e-6:
        lines.append(
            f"            NOTE: deployed NMS IoU {nms_iou} differs from the ultralytics val default "
            f"{ULTRALYTICS_VAL_IOU}; crowded scenes lose recall first."
        )
    model_w = info.get("model_width")
    model_h = info.get("model_height")
    if model_w and model_h and model_w != model_h:
        lines.append(
            f"            NOTE: deployed at {model_w}x{model_h} (non-square) while offline eval "
            "letterboxes to a square; expect a small systematic gap."
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score dumped detections against COCO ground truth.")
    parser.add_argument("--detections", required=True, help="JSON from dump_detections.py or a COCO results array.")
    parser.add_argument("--ground-truth", required=True, help="COCO-format ground-truth annotations JSON.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Operating point for precision/recall; 0 uses every dumped detection.",
    )
    parser.add_argument("--pr-iou", type=float, default=0.5, help="IoU for the precision/recall operating point.")
    parser.add_argument("--categories", default="auto", help="auto, all, or a comma list of COCO category ids.")
    parser.add_argument(
        "--images",
        choices=("overlap", "all"),
        default="overlap",
        help="overlap scores only ground-truth images the dump covered; all counts uncovered images as missed.",
    )
    parser.add_argument("--category-map", choices=CATEGORY_MAPS, default=None, help="Re-map dumped category ids.")
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--json", default=None, help="Also write the metrics to this JSON path.")
    parser.add_argument("--quiet", action="store_true", help="Skip the pycocotools summarize() table.")
    return parser.parse_args()


def score(args: argparse.Namespace) -> dict:
    COCO, COCOeval = import_pycocotools()

    detections, info = load_dump(resolve_project_path(args.detections))
    if args.category_map:
        detections = remap_category_ids(detections, info.get("category_map", "identity"), args.category_map)
        info = dict(info, category_map=args.category_map)

    coco_gt = COCO(str(resolve_project_path(args.ground_truth)))
    categories = resolve_categories(coco_gt, args.categories, info)
    problems = check_alignment(detections, coco_gt, categories)
    for problem in problems:
        print(f"WARNING: {problem}", file=sys.stderr, flush=True)
    if not detections:
        raise SystemExit(1)

    gt_image_ids = set(coco_gt.getImgIds())
    if args.images == "overlap":
        image_ids = gt_image_ids & {detection["image_id"] for detection in detections}
    else:
        image_ids = gt_image_ids

    coco_dt = coco_gt.loadRes(list(detections))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = categories
    coco_eval.params.imgIds = sorted(image_ids)
    coco_eval.params.maxDets = [1, 10, args.max_dets]

    coco_eval.evaluate()
    coco_eval.accumulate()
    if not args.quiet:
        coco_eval.summarize()

    stats = [float(value) for value in coco_eval.stats]
    operating_point = precision_recall(coco_eval, args.pr_iou, args.score_threshold)
    return {
        "detections_file": str(resolve_project_path(args.detections)),
        "ground_truth_file": str(resolve_project_path(args.ground_truth)),
        "categories": categories,
        "image_selection": args.images,
        "images_scored": len(coco_eval.params.imgIds),
        "images_in_ground_truth": len(gt_image_ids),
        "max_dets": args.max_dets,
        "detections_scored": len(detections),
        "map50_95": stats[0],
        "map50": stats[1],
        "map75": stats[2],
        "map_small": stats[3],
        "map_medium": stats[4],
        "map_large": stats[5],
        "recall_max_dets": stats[8],
        "operating_point": operating_point,
        "dump_info": info,
        "warnings": problems,
    }


def print_report(metrics: dict) -> None:
    info = metrics["dump_info"]
    point = metrics["operating_point"]
    print("")
    print(f"detections  {metrics['detections_file']}")
    print(
        f"            n={metrics['detections_scored']} "
        f"images={metrics['images_scored']}/{metrics['images_in_ground_truth']} "
        f"({metrics['image_selection']}) categories={len(metrics['categories'])}"
    )
    if info:
        print(
            f"geometry    source={info.get('source_width')}x{info.get('source_height')} "
            f"model={info.get('model_width')}x{info.get('model_height')} "
            f"output={info.get('output_width')}x{info.get('output_height')} "
            f"category_map={info.get('category_map')}"
        )
    for line in threshold_report(info, point["score_threshold"]):
        print(line)
    print(
        f"map         mAP50-95={metrics['map50_95']:.4f} mAP50={metrics['map50']:.4f} "
        f"mAP75={metrics['map75']:.4f}"
    )
    print(
        f"            small={metrics['map_small']:.4f} medium={metrics['map_medium']:.4f} "
        f"large={metrics['map_large']:.4f} AR@{metrics['max_dets']}={metrics['recall_max_dets']:.4f}"
    )
    print(
        f"operating   IoU={point['iou']:g} score>={point['score_threshold']:g} "
        f"precision={point['precision']:.4f} recall={point['recall']:.4f} "
        f"TP={point['true_positives']} FP={point['false_positives']} FN={point['false_negatives']} "
        f"GT={point['ground_truths']}"
    )


def main() -> int:
    args = parse_args()
    metrics = score(args)
    print_report(metrics)

    if args.json:
        output = resolve_project_path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"metrics     {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
