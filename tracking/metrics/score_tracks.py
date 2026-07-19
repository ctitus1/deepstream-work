#!/usr/bin/env python3
"""PLACEHOLDER MOT metrics CLI.

Scores tracker output against ground truth using the MOT Challenge metrics
(MOTA, MOTP, IDF1, HOTA). Nothing is implemented: ``load_mot_txt()`` and
``score_tracks()`` raise ``NotImplementedError``. The argument parsing and the
file format below are the real contract, so the CLI shape does not have to be
redesigned when the scoring is written.

Neither input file exists yet. Ground truth has to be hand-labelled, and
predictions need an exporter probe on the ``nvtracker`` src pad -- see
``tracking/README.md``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from deepstream_yolo.paths import resolve_project_path  # noqa: E402

TRACKING_DIR = PROJECT_DIR / "tracking"

# MOT Challenge txt: one comma-separated row per object per frame, no header.
# frame and id are 1-based; bb_left/bb_top are the top-left corner in pixels.
# In detection/prediction files conf is the score and x,y,z are unused (-1).
# In 2D ground truth files conf is the 0/1 "consider this box" flag, x,y,z are -1.
MOT_COLUMNS = (
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
)

# Example row: 1,3,912.5,484.0,97.0,109.0,0.87,-1,-1,-1
MOT_UNUSED = -1

SUPPORTED_METRICS = ("mota", "motp", "idf1", "hota")
BACKENDS = ("motmetrics", "trackeval")


@dataclass(frozen=True)
class TrackBox:
    """One row of a MOT Challenge txt file."""

    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: float


def load_mot_txt(path: Path) -> list[TrackBox]:
    """Parse a MOT Challenge txt file into TrackBox rows.

    TODO: split on commas, coerce the first two columns to int and the next five
    to float, ignore the trailing x/y/z, and skip blank lines. Reject rows whose
    column count is not len(MOT_COLUMNS) rather than padding them -- a truncated
    export should fail loudly.
    """
    raise NotImplementedError(f"MOT txt parsing is not implemented (would read {path})")


def score_tracks(
    gt: list[TrackBox],
    pred: list[TrackBox],
    metrics: list[str],
    backend: str,
    iou_threshold: float,
) -> dict[str, float]:
    """Score predicted tracks against ground truth.

    TODO: implement against one of the two intended backends, both of which live
    in .venv-yolo and must be imported inside this function, never at module
    scope -- this file may end up imported by the DeepStream app, which runs on
    the system interpreter.

    py-motmetrics (https://github.com/cheind/py-motmetrics)
        Lighter, pure Python. Covers MOTA, MOTP, IDF1, ID switches and
        fragmentation. Usage: build a ``motmetrics.MOTAccumulator``, call
        ``update()`` per frame with gt ids, hypothesis ids and an IOU distance
        matrix from ``motmetrics.distances.iou_matrix(..., max_iou=1 - iou_threshold)``,
        then ``motmetrics.metrics.create().compute(acc, metrics=[...])``.
        Does NOT implement HOTA.

    TrackEval (https://github.com/JonathonLuiten/TrackEval)
        The MOT Challenge reference evaluator and the only source of a canonical
        HOTA. Heavier: expects an on-disk directory layout (seqmaps, per-sequence
        gt/ and trackers/ folders) rather than in-memory lists, so wiring it up
        means materialising that layout in a temp dir.

    Suggested split: motmetrics for the fast MOTA/MOTP/IDF1 loop during tuning,
    TrackEval when HOTA is requested or a publishable number is needed.
    """
    raise NotImplementedError(
        f"Scoring is not implemented (backend={backend}, metrics={metrics}, "
        f"iou_threshold={iou_threshold}, {len(gt)} gt rows, {len(pred)} pred rows)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", required=True, help="Ground-truth MOT Challenge txt")
    parser.add_argument("--pred", required=True, help="Predicted tracks MOT Challenge txt")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(SUPPORTED_METRICS),
        choices=SUPPORTED_METRICS,
    )
    parser.add_argument("--backend", default="motmetrics", choices=BACKENDS)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--json", help="Write results to this path instead of stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = resolve_project_path(args.gt)
    pred_path = resolve_project_path(args.pred)

    for path in (gt_path, pred_path):
        if not path.exists():
            raise SystemExit(f"Missing MOT txt: {path}")

    gt = load_mot_txt(gt_path)
    pred = load_mot_txt(pred_path)
    results = score_tracks(gt, pred, args.metrics, args.backend, args.iou_threshold)

    # TODO: write results to args.json when set, otherwise print an aligned table.
    print(results)


if __name__ == "__main__":
    main()
