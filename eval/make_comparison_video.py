#!/usr/bin/env python3
"""Render a 2x2 comparison video from saved run outputs.

Draws each approach's boxes over the source video and tiles the four side by
side. Reads the runs that are already on disk rather than re-running anything,
so this costs one decode.

Writes an MJPEG stream to stdout for ffmpeg to encode. Not raw frames: docker's
stdout proxy moves about 55 MB/s, and raw 1920x1080 is 6.2 MB a frame, so five
seconds of output is 933 MB and measured 17 seconds of pure piping against 1.6
to decode, resize and tile the same frames. The pipe was costing more than all
the work. JPEG at quality 97 is visually indistinguishable here and roughly
thirty times smaller, and the host still does one proper x264 encode at the end.

Encoding in-container would avoid the pipe entirely, but this image has no
usable encoder: DeepStream deletes the ffmpeg shared libraries (nothing on the
filesystem provides libavcodec.so.58), OpenCV is built without GStreamer, and
its bundled ffmpeg has no x264 -- cv2.VideoWriter falls through to a V4L2
device that does not exist.

Everything except the JPEG bytes goes to stderr so stdout stays clean.

    docker run --rm -i --user "$(id -u):$(id -g)" --entrypoint python3 \
      -v "$PWD":/w -w /w deepstream-work:7.1 \
      eval/make_comparison_video.py --video V --runs A.json,B.json 2>/dev/null \
      | ffmpeg -f image2pipe -vcodec mjpeg -r 30 -i - \
               -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p out.mp4

Two details of that command are load-bearing, and getting either wrong
corrupts the output in a way that looks like a rendering bug rather than a
plumbing one:

* ``--entrypoint python3`` bypasses the image's entrypoint, which prints a
  747-byte CUDA banner **to stdout**. When this wrote raw frames those bytes
  shifted every frame by 249 pixels and rotated the colour channels, giving a
  wrapped, red-and-blue-swapped video. MJPEG is self-framing so the damage is
  now a corrupt first frame rather than a corrupt everything, but the banner
  still has no business in the stream.
* ``docker run``, not ``docker compose run``: compose writes its own status
  lines to stdout and does the same thing.

Neither needs the GPU -- this reads saved JSON and decodes video on the CPU --
so plain ``docker run`` with no device flags is enough.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

TILE_W, TILE_H = 960, 540
BOX_COLOR = (170, 170, 170)
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--runs", required=True, help="comma-separated run JSON paths")
    parser.add_argument(
        "--labels",
        default="",
        help="comma-separated panel labels; defaults to each run's approach name",
    )
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--frames", type=int, default=0, help="0 = to end of video")
    parser.add_argument("--quality", type=int, default=97, help="JPEG quality on the pipe")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="threads used for JPEG encoding; cv2 releases the GIL so these do run",
    )
    parser.add_argument("--tile-width", type=int, default=TILE_W)
    parser.add_argument("--tile-height", type=int, default=TILE_H)
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="print the output WIDTHxHEIGHT and exit, so the caller can tell ffmpeg",
    )
    return parser.parse_args()


def grid_shape(count: int) -> tuple[int, int]:
    """Rows and columns for ``count`` panels: two columns unless there are two."""
    if count <= 1:
        return 1, 1
    if count == 2:
        return 1, 2
    return (count + 1) // 2, 2


def load(run_paths: list[str], labels: list[str]):
    panels = []
    for i, raw in enumerate(run_paths):
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if not path.is_file():
            print(f"missing run: {path}", file=sys.stderr)
            raise SystemExit(2)
        data = json.loads(path.read_text())
        label = labels[i] if i < len(labels) and labels[i] else data.get("approach", path.stem)
        by_index = {f["index"]: f["boxes"] for f in data["frames"]}
        panels.append((by_index, label))
    return panels


def draw(tile, boxes, label: str, scale_x: float, scale_y: float) -> None:
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



def main() -> int:
    args = parse_args()
    run_paths = [r for r in args.runs.split(",") if r]
    labels = args.labels.split("|") if args.labels else []
    rows, cols = grid_shape(len(run_paths))
    tile_w, tile_h = args.tile_width, args.tile_height

    if args.geometry:
        # The caller has to tell ffmpeg the raw frame size before a byte is
        # written, and only this script knows the grid layout.
        print(f"{cols * tile_w}x{rows * tile_h}")
        return 0

    panels = load(run_paths, labels)

    video = Path(args.video)
    if not video.is_absolute():
        video = PROJECT_DIR / video
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print("cannot open video", file=sys.stderr)
        return 2

    src_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    src_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale_x, scale_y = tile_w / src_w, tile_h / src_h
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    count = args.frames if args.frames > 0 else max(total - args.start, 0)
    if args.start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    # A blank tile pads the grid when the panel count is odd, so the frame size
    # stays constant -- ffmpeg is told one size up front and every frame must
    # match it exactly.
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    out = sys.stdout.buffer
    params = [cv2.IMWRITE_JPEG_QUALITY, int(args.quality)]

    def encode(grid):
        ok, buf = cv2.imencode(".jpg", grid, params)
        return buf.tobytes() if ok else b""

    # JPEG encoding is the remaining per-frame cost, and cv2.imencode releases
    # the GIL, so a small pool genuinely overlaps it with decode and tiling.
    # Results are written in submission order -- a video is an ordered thing and
    # completion order is not.
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    pending: list = []

    def drain(limit: int) -> None:
        while len(pending) > limit:
            out.write(pending.pop(0).result())

    for offset in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        index = args.start + offset
        small = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        tiles = []
        for by_index, label in panels:
            tile = small.copy()
            draw(tile, by_index.get(index, []), label, scale_x, scale_y)
            tiles.append(tile)
        while len(tiles) < rows * cols:
            tiles.append(blank)

        grid = np.vstack(
            [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
        )
        pending.append(pool.submit(encode, grid))
        # Bounded so the queue cannot grow into a copy of the whole video.
        drain(args.workers * 2)
        if offset % 120 == 0:
            print(f"  frame {index}", file=sys.stderr, flush=True)

    drain(0)
    pool.shutdown()
    capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
