"""Cost of ``age_min`` on a target entering from the panning edge.

The real footage cannot answer this: its mover is already well inside the frame
whenever it is visible, and age_min 1 and age_min 15 score identically on it,
to three decimals, at every distance from the border. So the failure mode is
provoked directly here.

Camera pans right at a fixed rate. Every frame, a strip of scene that has never
been modelled appears at the right border; those blocks are reinitialised with
age 0 and cannot fire until age_min frames have passed. A target that walks in
through that strip is therefore invisible for a predictable distance, and the
prediction is that the blind band is roughly ``age_min * pan_speed`` pixels
wide -- independent of the target.

Run: docker compose run --rm -T deepstream-dev python3 edge_entry_experiment.py
"""

import sys

sys.path.insert(0, "src")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from deepstream_yolo.approaches.fastmcd import Approach  # noqa: E402

W, H = 960, 540
SRC_W, SRC_H = 3840, 2160
WARMUP = 90          # frames before the target appears: model fully mature
TARGET_W, TARGET_H = 26, 56


class Ctx:
    src_w, src_h, width, height = SRC_W, SRC_H, W, H

    def __init__(self, i, rgba):
        self.frame_index, self.pts, self.flow, self.rgba = i, 0, None, rgba


def make_world(seed=1):
    rng = np.random.default_rng(seed)
    fine = cv2.GaussianBlur((rng.random((1400, 3000)) * 255).astype(np.uint8), (0, 0), 1.4)
    coarse = cv2.resize((rng.random((70, 150)) * 255).astype(np.uint8), (3000, 1400))
    return cv2.addWeighted(fine, 0.55, coarse, 0.45, 0)


def run(world, age_min, pan, speed=3.0, frames=200):
    """Target walks in through the right border at ``speed`` px/frame.

    It must have motion of its own relative to the scene, or it is background
    by definition and the question is meaningless. ``speed`` is its velocity in
    the frame; the camera pans right at ``pan`` underneath it.
    """
    a = Approach({"age_min": age_min})
    for i in range(frames):
        ox = 200 + i * pan
        view = cv2.warpAffine(world, np.float32([[1, 0, -ox], [0, 1, -100]]), (W, H))
        depth = None
        if i >= WARMUP:
            depth = speed * (i - WARMUP)       # px travelled in from the border
            tx = int(W - TARGET_W - depth)
            ty = 240
            if tx < 0:
                break
            cv2.rectangle(view, (tx, ty), (tx + TARGET_W, ty + TARGET_H), 25, -1)
        rgba = np.dstack([view, view, view, np.full_like(view, 255)])
        boxes = a.process(Ctx(i, rgba))
        if depth is not None:
            cx = (tx + TARGET_W / 2) * (SRC_W / W)
            cy = (ty + TARGET_H / 2) * (SRC_H / H)
            if any(
                b["left"] <= cx <= b["left"] + b["width"]
                and b["top"] <= cy <= b["top"] + b["height"]
                for b in boxes
            ):
                return i - WARMUP, int(depth)
    return None, None


world = make_world()
print(f"{'pan px/f':>9} {'age_min':>8} {'frames blind':>13} {'blind band px':>14} {'predicted':>10}")
for pan in (1.0, 2.0, 4.0):
    for age_min in (1, 5, 15, 25):
        lat, depth = run(world, age_min, pan)
        lat_s = "never" if lat is None else str(lat)
        depth_s = "-" if depth is None else f"{depth}"
        print(f"{pan:>9.1f} {age_min:>8} {lat_s:>13} {depth_s:>14} {age_min * pan:>10.0f}")
