#!/usr/bin/env python3
"""Score a motion run against the detector's own boxes.

The question being asked is "did the motion detector put a box on the person
who is actually moving, and only on them". So:

  * A motion box is a HIT when its centroid falls inside a mover detection box
    for that frame. Centroid-in-box rather than IoU on purpose: an optical-flow
    blob legitimately covers the space a target swept through, not the target's
    silhouette, so its box is often larger or offset. Demanding IoU would
    punish a detector for correctly reporting a trail. Landing on the target is
    the thing that matters.
  * A motion box that hits nothing is a FALSE POSITIVE.
  * A frame that contains a mover but gets no hit is a MISS.

Reported separately, because the two failures mean different things:

  * ``on_stationary`` -- boxes landing on one of the three people known to be
    standing still. These are the worst kind of false positive: the approach is
    not merely noisy, it is calling a stationary person a mover.
  * ``on_background`` -- boxes landing on nothing at all.

Usage:
    python3 eval/score.py eval/runs/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "eval"))

from tracks import mover_boxes_by_frame, resolve  # noqa: E402


def inside(box: dict, point: tuple[float, float]) -> bool:
    return (
        box["left"] <= point[0] <= box["left"] + box["width"]
        and box["top"] <= point[1] <= box["top"] + box["height"]
    )


def stationary_boxes_by_frame(data: dict, stationary) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for track in stationary:
        for index, box in zip(track.frames, track.boxes):
            out.setdefault(index, []).append(box)
    return out


def score(run: dict, movers: dict, statics: dict) -> dict:
    hits = 0
    on_stationary = 0
    on_background = 0
    total_boxes = 0
    frames_with_mover = 0
    frames_hit = 0
    covered_movers = 0
    total_movers = 0

    for frame in run["frames"]:
        index = frame["index"]
        mover = movers.get(index, [])
        static = statics.get(index, [])
        boxes = frame["boxes"]
        total_boxes += len(boxes)

        if mover:
            frames_with_mover += 1
            total_movers += len(mover)

        frame_hit = False
        hit_movers = set()
        for box in boxes:
            point = (
                box["left"] + box["width"] / 2.0,
                box["top"] + box["height"] / 2.0,
            )
            matched = [i for i, m in enumerate(mover) if inside(m, point)]
            if matched:
                hits += 1
                frame_hit = True
                hit_movers.update(matched)
            elif any(inside(s, point) for s in static):
                on_stationary += 1
            else:
                on_background += 1

        covered_movers += len(hit_movers)
        if frame_hit:
            frames_hit += 1

    precision = hits / total_boxes if total_boxes else 0.0
    recall = frames_hit / frames_with_mover if frames_with_mover else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "approach": run.get("approach", run.get("module", "?")),
        "frames": len(run["frames"]),
        "boxes": total_boxes,
        "boxes_per_frame": total_boxes / max(len(run["frames"]), 1),
        "hits": hits,
        "on_stationary": on_stationary,
        "on_background": on_background,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mover_coverage": covered_movers / total_movers if total_movers else 0.0,
        "frames_with_mover": frames_with_mover,
        "ms_median": run.get("ms_per_frame_median", 0.0),
        "ms_p95": run.get("ms_per_frame_p95", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--detections", default=str(PROJECT_DIR / "eval" / "detections.json"))
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    data, _tracks, _movers, stationary = resolve(args.detections)
    movers = mover_boxes_by_frame(data, stationary)
    statics = stationary_boxes_by_frame(data, stationary)

    rows = []
    for path in args.runs:
        run = json.loads(Path(path).read_text())
        rows.append(score(run, movers, statics))

    rows.sort(key=lambda r: -r["f1"])

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(
        f"{'approach':<24} {'F1':>6} {'prec':>6} {'recall':>7} {'cover':>6} "
        f"{'box/f':>6} {'onstat':>7} {'onbg':>7} {'ms':>6}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r['approach'][:24]:<24} {r['f1']:>6.3f} {r['precision']:>6.3f} "
            f"{r['recall']:>7.3f} {r['mover_coverage']:>6.3f} {r['boxes_per_frame']:>6.2f} "
            f"{r['on_stationary']:>7} {r['on_background']:>7} {r['ms_median']:>6.1f}"
        )
    print(
        f"\nground truth: {len(movers)} frames contain a mover, "
        f"{len(statics)} contain stationary people"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
