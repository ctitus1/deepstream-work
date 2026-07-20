#!/usr/bin/env python3
"""Render a 2x2 comparison video from saved run outputs.

Draws each approach's boxes over the source video and tiles the four side by
side. Reads the runs that are already on disk rather than re-running anything,
so this costs one decode.

Writes raw BGR frames to stdout for ffmpeg to encode -- keeping the pixels
lossless until the single encode at the end, which is what stops the boxes and
labels turning to mush. Everything else goes to stderr so stdout stays clean.

    docker run --rm -i --user "$(id -u):$(id -g)" --entrypoint python3 \
      -v "$PWD":/w -w /w deepstream-work:7.1 \
      eval/make_comparison_video.py --start 2160 --frames 150 2>/dev/null \
      | ffmpeg -f rawvideo -pix_fmt bgr24 -s 1920x1080 -r 30 -i - \
               -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p out.mp4

Two details of that command are load-bearing, and getting either wrong
corrupts the output in a way that looks like a rendering bug rather than a
plumbing one:

* ``--entrypoint python3`` bypasses the image's entrypoint, which prints a
  747-byte CUDA banner **to stdout**. Those bytes land at the head of the raw
  stream and shift every frame by 249 pixels, so each tile wraps its right
  edge onto its left. 747 is not divisible by 3, so the channel order rotates
  too and the video comes out with its reds and blues swapped. One stray
  banner, two symptoms that look unrelated.
* ``docker run``, not ``docker compose run``: compose writes its own status
  lines to stdout and does the same thing.

Neither needs the GPU -- this reads saved JSON and decodes video on the CPU --
so plain ``docker run`` with no device flags is enough.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

# (run file, label). Order is the grid: top-left, top-right, bottom-left, bottom-right.
PANELS = [
    ("klt_homography.json", "klt-homography   F1 0.937"),
    ("gradient_diff.json", "gradient-diff    F1 0.925"),
    ("bgsub_compensated.json", "bgsub-compensated  F1 0.920"),
    ("baseline.json", "baseline (shipped)  F1 0.422"),
]

TILE_W, TILE_H = 960, 540
BOX_COLOR = (170, 170, 170)
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="streams/lorton-d4-rgb.mp4")
    parser.add_argument("--start", type=int, default=3000, help="first frame index")
    parser.add_argument("--frames", type=int, default=540, help="how many frames (30fps)")
    parser.add_argument("--runs", default="eval/runs")
    return parser.parse_args()


def load(runs_dir: Path):
    panels = []
    for name, label in PANELS:
        path = runs_dir / name
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            raise SystemExit(2)
        data = json.loads(path.read_text())
        by_index = {f["index"]: f["boxes"] for f in data["frames"]}
        panels.append((by_index, label))
    return panels


def draw(tile: np.ndarray, boxes, label: str, scale_x: float, scale_y: float) -> None:
    for box in boxes:
        x0 = int(round(box["left"] * scale_x))
        y0 = int(round(box["top"] * scale_y))
        x1 = int(round((box["left"] + box["width"]) * scale_x))
        y1 = int(round((box["top"] + box["height"]) * scale_y))
        cv2.rectangle(tile, (x0, y0), (x1, y1), BOX_COLOR, 3)

    # Label bar across the top. Double the size that was legible uncompressed --
    # these tiles are a quarter of the frame and get re-encoded once more.
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(tile, (0, 0), (text_w + 28, text_h + 26), LABEL_BG, -1)
    cv2.putText(tile, label, (14, text_h + 12), font, font_scale, LABEL_FG, thickness, cv2.LINE_AA)

    count = f"{len(boxes)} box" + ("" if len(boxes) == 1 else "es")
    (cw, ch), _ = cv2.getTextSize(count, font, 0.9, 2)
    cv2.rectangle(tile, (0, TILE_H - ch - 22), (cw + 24, TILE_H), LABEL_BG, -1)
    cv2.putText(tile, count, (12, TILE_H - 10), font, 0.9, LABEL_FG, 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    panels = load(PROJECT_DIR / args.runs)

    capture = cv2.VideoCapture(str(PROJECT_DIR / args.video))
    if not capture.isOpened():
        print("cannot open video", file=sys.stderr)
        return 2

    src_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    src_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale_x, scale_y = TILE_W / src_w, TILE_H / src_h
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    out = sys.stdout.buffer
    for offset in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            break
        index = args.start + offset
        small = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

        tiles = []
        for by_index, label in panels:
            tile = small.copy()
            draw(tile, by_index.get(index, []), label, scale_x, scale_y)
            tiles.append(tile)

        grid = np.vstack(
            [np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])]
        )
        out.write(grid.tobytes())
        if offset % 120 == 0:
            print(f"  frame {index}", file=sys.stderr, flush=True)

    capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
