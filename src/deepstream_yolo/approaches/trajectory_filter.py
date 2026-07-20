"""Track-before-detect: deliberately loose proposals, confirmed by trajectory.

The baseline tunes for per-frame precision and pays recall 0.300 for it. But
its false positives and its true positive do not fail the same way: on this
footage the FPs are sparse and transient -- a blob appears for a frame or two
somewhere new -- while the walker is dense and persistent, in the same place,
moving the same direction, frame after frame. The baseline throws that
distinction away by deciding afresh every frame.

So the tuning philosophy is inverted here:

  1. **Loosen the front end.** Same flow analysis as the baseline (copied, not
     imported, so this branch cannot perturb the shipping detector), but the
     seed threshold is dropped to the ordinary threshold, the adaptive term is
     relaxed, the area floor is lowered and no cap is applied. Five to ten
     noisy proposals per frame is the intended operating point. Recall at the
     proposal stage is all that matters; precision becomes the tracker's job.

  2. **Track them.** IoU + centroid association, one constant-velocity Kalman
     filter per track, coasting through gaps.

  3. **Confirm on the trajectory, not the frame.** A track emits nothing until
     it has been seen ``n_init`` times *and* its path looks like a walk. That
     is the actual filter, and the reason it is worth doing: a false positive
     has to be re-found in the same place, moving the same way, ``n_init``
     times running, so its survival probability falls roughly as p^M while the
     walker's falls only linearly.

The confirmation tests are three, and all three are physical rather than tuned:

  * **Net displacement.** ``|p_end - p_start|`` must exceed ``d_min``. Rejects a
    blob that flickers in place without going anywhere -- which is what a
    misregistered edge or a wind-shaken bush does.
  * **Straightness.** ``|p_end - p_start| / sum|dp|`` must exceed 0.6. Over a
    third of a second a person walks near-straight; noise wanders. (The
    baseline computes this same ratio *spatially*, across neighbouring flow
    vectors, as its coherence term. Applying it *temporally* along a track is a
    strictly better use of the statistic: it has the whole history to work
    with rather than a 3x3 window.)
  * **Velocity plausibility.** A person walks 1-2 m/s, which on this footage is
    ~0.35-2.5 cells/frame measured off the detector's own boxes. The gate is
    set wide around that. It is a prior about people, not a knob.

**Ego-motion compensation is not optional here.** The camera shakes, so a box's
image-space motion is target motion plus camera motion, and at this altitude
the camera term is the larger of the two. Feeding that to a constant-velocity
filter makes it fight the shake, and every displacement and straightness test
downstream measures the wrong thing. The affine background fit the proposal
stage already runs is evaluated at each track's own position and accumulated,
so association and all three gates happen in a scene-stabilised frame.

The known limit, stated up front: this suppresses *random* false positives, not
*systematic* ones. A structure that misregisters the same way every frame is
persistent and straight by construction and will confirm. See the ``img_disp``
/ ``stab_disp`` keys on every emitted box, which exist to measure exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.ndimage import label as _scipy_label
except Exception:  # pragma: no cover - scipy is installed in the image
    _scipy_label = None

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None

NAME = "trajectory-filter"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Every knob, with the baseline's value noted where it differs.

    Units: ``*_frac`` speeds are a fraction of the branch frame width per frame
    (as in the baseline, so they mean the same thing at any resolution).
    Distances named ``*_cells`` are in flow cells -- one cell is ``grid_size``
    branch pixels, 16 source pixels here.
    """

    # --- proposal stage, copied from MotionConfig then loosened -----------
    border: int = 20
    decay: float = 0.85
    min_speed_frac: float = 0.0022
    # Baseline 6.0. The adaptive floor, not the fixed one, is usually what
    # binds; leaving it at 6 sigma makes lowering min_speed_frac a no-op.
    noise_scale: float = 3.0
    # Baseline 0.0040. Dropped to min_speed_frac: hysteresis exists to protect
    # precision, and precision is no longer this stage's job.
    seed_speed_frac: float = 0.0022
    max_speed: float = 18.0
    min_coherence: float = 0.40  # baseline 0.50
    # Baseline 0.005 (~95 cells). A mover's bounding box is only ~430 cells and
    # its *moving* cells far fewer, so the baseline floor is already cutting
    # into real targets.
    min_area_frac: float = 0.0010
    max_area_frac: float = 0.25
    dilate: int = 2
    merge_gap_frac: float = 0.02
    # No max_boxes. Capping output is not how this is meant to score.

    # --- association -------------------------------------------------------
    # Gate on centroid distance, in cells. The walker's own 99th percentile
    # step is ~9.6 cells/frame, so 12 admits every real step.
    gate_cells: float = 12.0
    iou_weight: float = 1.0
    dist_weight: float = 1.0
    feature_weight: float = 0.0  # >0 turns on the appearance descriptor
    feature_bins: int = 6

    # --- track lifecycle ---------------------------------------------------
    n_init: int = 4
    max_age: int = 15  # 0.5 s at 30 fps
    # Frames a confirmed track may be emitted for while coasting. Separate from
    # max_age: keeping a track alive through an occlusion is cheap, but drawing
    # a box where nothing was measured is a precision risk that grows with
    # every frame of extrapolation.
    max_coast_emit: int = 6

    # --- confirmation gates ------------------------------------------------
    # 8 branch px = 2 cells = 32 source px.
    d_min_cells: float = 2.0
    straightness: float = 0.60
    # 1-2 m/s here is ~0.35-2.5 cells/frame off the detector's own boxes.
    # Gated wide: this rejects the impossible, it does not select the typical.
    v_min_cells: float = 0.12
    v_max_cells: float = 7.0
    # Trajectory window the gates are measured over, in frames.
    window: int = 20
    # Re-test the gates every frame on the trailing window, not just at
    # confirmation. A track that confirms and then stops dead is no longer
    # evidence of a mover.
    revalidate: bool = True

    # --- Kalman ------------------------------------------------------------
    q_pos: float = 0.25
    q_size: float = 1.0
    q_vel: float = 0.04
    r_pos: float = 2.0
    r_size: float = 9.0


# ---------------------------------------------------------------------------
# Proposal stage -- copied from deepstream_yolo.motion, then loosened
# ---------------------------------------------------------------------------


def _background_flow(flow: np.ndarray, iterations: int = 3):
    """Robust affine fit of the camera's own contribution to the flow field.

    Copied from ``motion.py`` with one addition: the fitted coefficients come
    back alongside the field, because ego-motion compensation needs to evaluate
    the model at an arbitrary point (a track's position) rather than only at
    the cells it was fitted on.

    A single median translation cannot describe a moving camera -- pan, roll and
    altitude change all make the background flow vary across the frame, and
    subtracting one number leaves a gradient that reads as motion at whichever
    edge is furthest from centre. Six parameters cover all three. The fit is
    iterated with outlier rejection because the targets are in the data too,
    and a plain least-squares fit would partly explain away the thing being
    looked for.
    """
    rows, cols = flow.shape[:2]
    ys, xs = np.mgrid[0:rows, 0:cols]
    nx = (xs / max(cols - 1, 1)) * 2.0 - 1.0
    ny = (ys / max(rows - 1, 1)) * 2.0 - 1.0
    basis = np.stack(
        [nx.ravel(), ny.ravel(), np.ones(nx.size, dtype=np.float32)], axis=1
    ).astype(np.float32)

    fx = flow[..., 0].ravel()
    fy = flow[..., 1].ravel()
    keep = np.ones(fx.size, dtype=bool)
    coef_x = coef_y = None

    for _ in range(iterations):
        if int(keep.sum()) < 16:
            break
        coef_x = np.linalg.lstsq(basis[keep], fx[keep], rcond=None)[0]
        coef_y = np.linalg.lstsq(basis[keep], fy[keep], rcond=None)[0]
        residual = np.hypot(fx - basis @ coef_x, fy - basis @ coef_y)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        limit = median + 3.0 * 1.4826 * max(mad, 1e-3) + 0.25
        tightened = residual <= limit
        if int(tightened.sum()) < 16:
            break
        keep = tightened

    if coef_x is None or coef_y is None:
        return np.zeros_like(flow), None, None

    field_ = np.stack(
        [(basis @ coef_x).reshape(rows, cols), (basis @ coef_y).reshape(rows, cols)],
        axis=-1,
    )
    return field_, coef_x, coef_y


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2r+1) square window; copied from ``motion.py``."""
    out = values.copy()
    height, width = values.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            src_y = slice(max(0, -dy), height - max(0, dy))
            dst_y = slice(max(0, dy), height - max(0, -dy))
            src_x = slice(max(0, -dx), width - max(0, dx))
            dst_x = slice(max(0, dx), width - max(0, -dx))
            out[dst_y, dst_x] += values[src_y, src_x]
    return out


def _coherence(residual: np.ndarray, radius: int = 1) -> np.ndarray:
    """Local agreement on a direction, in [0, 1]; copied from ``motion.py``.

    ``|sum of vectors| / sum of |vectors|``. This is what tells a target apart
    from a featureless patch where flow has nothing to lock onto and returns
    large arbitrary vectors.
    """
    sum_x = _box_sum(residual[..., 0].copy(), radius)
    sum_y = _box_sum(residual[..., 1].copy(), radius)
    vector_sum = np.hypot(sum_x, sum_y)
    magnitude_sum = _box_sum(np.hypot(residual[..., 0], residual[..., 1]), radius)
    return vector_sum / np.maximum(magnitude_sum, 1e-6)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation by OR-ing shifted copies; copied from ``motion.py``."""
    if radius <= 0:
        return mask
    height, width = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            src_y = slice(max(0, -dy), height - max(0, dy))
            dst_y = slice(max(0, dy), height - max(0, -dy))
            src_x = slice(max(0, -dx), width - max(0, dx))
            dst_x = slice(max(0, dx), width - max(0, -dx))
            out[dst_y, dst_x] |= mask[src_y, src_x]
    return out


_STRUCT8 = np.ones((3, 3), dtype=bool)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components as flat cell indices.

    ``motion.py`` hand-rolls union-find to avoid a scipy dependency in the
    shipping path. Here scipy is already a hard requirement (the assignment
    solver), and loosening the thresholds multiplies the number of set cells,
    so the vectorised labeller is what keeps this inside the frame budget.
    """
    if _scipy_label is None:  # pragma: no cover
        return []
    labels, count = _scipy_label(mask, structure=_STRUCT8)
    if count == 0:
        return []
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    ordered = flat[order]
    # Everything before the first nonzero label is background.
    start = int(np.searchsorted(ordered, 1))
    bounds = np.searchsorted(ordered, np.arange(1, count + 1), side="right")
    out = []
    prev = start
    for end in bounds:
        if end > prev:
            out.append(order[prev:end])
        prev = int(end)
    return out


def _merge_close(boxes: list[dict], gap: float, max_cells: float) -> list[dict]:
    """Merge boxes closer than ``gap``, repeatedly; copied from ``motion.py``.

    A person is not one rigid blob: limbs move at their own velocity and break
    off into their own components even after dilation. Merging by proximity
    puts them back with the torso.
    """
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out: list[dict] = []
        while boxes:
            head = boxes.pop()
            joined = True
            while joined:
                joined = False
                for i, other in enumerate(boxes):
                    gap_x = max(other["x0"] - head["x1"], head["x0"] - other["x1"])
                    gap_y = max(other["y0"] - head["y1"], head["y0"] - other["y1"])
                    if gap_x > gap or gap_y > gap:
                        continue
                    span_x = max(head["x1"], other["x1"]) - min(head["x0"], other["x0"])
                    span_y = max(head["y1"], other["y1"]) - min(head["y0"], other["y0"])
                    if span_x * span_y <= max_cells:
                        head = {
                            "x0": min(head["x0"], other["x0"]),
                            "y0": min(head["y0"], other["y0"]),
                            "x1": max(head["x1"], other["x1"]),
                            "y1": max(head["y1"], other["y1"]),
                            "energy": max(head["energy"], other["energy"]),
                            "cells": head["cells"] + other["cells"],
                        }
                        boxes.pop(i)
                        joined = True
                        merged = True
                        break
            out.append(head)
        boxes = out
    return boxes


# ---------------------------------------------------------------------------
# Appearance descriptor -- the chromaticity histogram from motion.py, adapted
# to take cell coordinates instead of a MotionBox
# ---------------------------------------------------------------------------


def _appearance_feature(rgba, x0c, y0c, x1c, y1c, grid_size, bins):
    """RG chromaticity histogram over a cell-coordinate patch.

    Chromaticity rather than RGB so a target keeps its descriptor walking from
    sun into shade, which is the entire job of a re-id feature. Blue is dropped
    because the three chromaticities sum to one.
    """
    height, width = rgba.shape[:2]
    x0 = max(0, min(width - 1, int(x0c * grid_size)))
    y0 = max(0, min(height - 1, int(y0c * grid_size)))
    x1 = max(x0 + 1, min(width, int(x1c * grid_size)))
    y1 = max(y0 + 1, min(height, int(y1c * grid_size)))

    patch = rgba[y0:y1, x0:x1, :3].astype(np.float32)
    if patch.size == 0:
        return None

    total = patch.sum(axis=2, keepdims=True)
    valid = total[..., 0] > 12.0
    if not bool(valid.any()):
        return None

    chroma = (patch / np.maximum(total, 1e-3))[valid]
    bins = max(2, bins)
    idx_r = np.clip((chroma[:, 0] * bins).astype(np.int32), 0, bins - 1)
    idx_g = np.clip((chroma[:, 1] * bins).astype(np.int32), 0, bins - 1)
    hist = np.bincount(idx_r * bins + idx_g, minlength=bins * bins).astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm <= 0.0:
        return None
    return hist / norm


def _feature_distance(a, b) -> float:
    """Cosine distance in [0, 2]; both inputs are L2-normalised already."""
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return 1.0 - float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Kalman track
# ---------------------------------------------------------------------------


@dataclass
class Track:
    """One constant-velocity track, in scene-stabilised cell coordinates.

    ``x`` is ``[cx, cy, w, h, vx, vy]``. Position and velocity live in the
    stabilised frame -- image position minus the accumulated camera motion in
    ``cam`` -- so ``vx, vy`` is the target's own velocity and the confirmation
    gates measure the target's own path.
    """

    track_id: int
    x: np.ndarray
    P: np.ndarray
    cam: np.ndarray  # cumulative camera displacement since birth, in cells
    hits: int = 1
    age: int = 0
    misses: int = 0
    born: int = 0
    confirmed: bool = False
    feature: np.ndarray | None = None
    # Stabilised centres, one per frame the track has existed.
    history: list = field(default_factory=list)
    # Image-space centres over the same frames, kept only so a surviving false
    # positive can be classified as static-in-image-space after the fact.
    img_history: list = field(default_factory=list)

    def predict(self, cfg: Config) -> None:
        self.x[0] += self.x[4]
        self.x[1] += self.x[5]
        # F P F^T for a constant-velocity model, written out: only the
        # position/velocity blocks couple.
        P = self.P
        for i, v in ((0, 4), (1, 5)):
            P[i, i] += 2.0 * P[i, v] + P[v, v]
            P[i, v] += P[v, v]
            P[v, i] = P[i, v]
        P[0, 0] += cfg.q_pos
        P[1, 1] += cfg.q_pos
        P[2, 2] += cfg.q_size
        P[3, 3] += cfg.q_size
        P[4, 4] += cfg.q_vel
        P[5, 5] += cfg.q_vel
        self.age += 1

    def update(self, z: np.ndarray, cfg: Config) -> None:
        """Scalar-sequential Kalman update over the four measured components.

        R is diagonal and H selects the first four states, so updating one
        component at a time is algebraically identical to the matrix form and
        avoids a 4x4 inverse per track per frame.
        """
        for i in range(4):
            r = cfg.r_pos if i < 2 else cfg.r_size
            s = self.P[i, i] + r
            if s <= 0.0:
                continue
            K = self.P[:, i] / s
            self.x += K * (z[i] - self.x[i])
            self.P -= np.outer(K, self.P[i, :])
        self.hits += 1
        self.misses = 0


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(union, 1e-6)


# ---------------------------------------------------------------------------
# Approach
# ---------------------------------------------------------------------------


class Approach:
    needs_flow = True
    needs_pixels = False

    def __init__(self, cfg: dict):
        base = Config()
        for k, v in (cfg or {}).items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v) if not isinstance(v, bool) else v)
        self.cfg = base
        # run_motion.py constructs the Approach before it reads needs_pixels,
        # so the appearance term can be switched on from --cfg without a second
        # module. Mapping the surface is not free, so it stays off by default.
        if base.feature_weight > 0.0:
            Approach.needs_pixels = True
        else:
            Approach.needs_pixels = False

        self.sustained: np.ndarray | None = None
        self.tracks: list[Track] = []
        self._next_id = 1
        self._frame = 0

    # -- proposal stage ---------------------------------------------------

    def _proposals(self, flow, ctx):
        """Loose motion proposals, in cropped-cell coordinates.

        Returns ``(proposals, coef_x, coef_y, offset, shape)``. The affine
        coefficients come back with them because ego-motion compensation needs
        the same fit.
        """
        cfg = self.cfg
        offset = cfg.border
        if offset > 0 and flow.shape[0] > 2 * offset and flow.shape[1] > 2 * offset:
            flow = flow[offset:-offset, offset:-offset]
        else:
            offset = 0

        bg, coef_x, coef_y = _background_flow(flow)
        residual = flow - bg

        if self.sustained is None or self.sustained.shape != residual.shape:
            self.sustained = residual.astype(np.float32, copy=True)
        else:
            self.sustained *= cfg.decay
            self.sustained += (1.0 - cfg.decay) * residual
        sustained = self.sustained
        speed = np.hypot(sustained[..., 0], sustained[..., 1])

        median = float(np.median(speed))
        mad = float(np.median(np.abs(speed - median)))
        threshold = max(
            cfg.min_speed_frac * ctx.width,
            median + cfg.noise_scale * 1.4826 * mad,
        )

        plausible = (speed <= cfg.max_speed) & (
            _coherence(sustained) >= cfg.min_coherence
        )
        moving = (speed > threshold) & plausible
        seed = (speed > max(cfg.seed_speed_frac * ctx.width, threshold)) & plausible
        mask = _dilate(moving, cfg.dilate)

        rows, cols = mask.shape
        total_cells = float(rows * cols)
        max_cells = cfg.max_area_frac * total_cells
        min_cells = cfg.min_area_frac * total_cells

        candidates: list[dict] = []
        for members in _components(mask):
            if len(members) < min_cells:
                continue
            ys, xs = np.divmod(members, cols)
            if not bool(seed[ys, xs].any()):
                continue
            x0, y0 = float(xs.min()), float(ys.min())
            x1, y1 = float(xs.max() + 1), float(ys.max() + 1)
            if (x1 - x0) * (y1 - y0) > max_cells or len(members) > max_cells:
                continue
            candidates.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "energy": float(speed[ys, xs].mean()),
                    "cells": int(len(members)),
                }
            )

        gap = cfg.merge_gap_frac * max(rows, cols)
        candidates = _merge_close(candidates, gap, max_cells)
        return candidates, coef_x, coef_y, offset, (rows, cols)

    # -- ego motion -------------------------------------------------------

    @staticmethod
    def _camera_flow(coef_x, coef_y, x, y, rows, cols, grid_size):
        """Background flow at cell ``(x, y)``, converted to cells/frame.

        The fit is in branch pixels per frame over normalised coordinates, so
        both a unit change and a coordinate change are needed. Evaluating the
        model per track rather than taking one global translation is what makes
        this correct under camera roll or altitude change, where the background
        moves in opposite directions at opposite edges of the frame.
        """
        if coef_x is None or coef_y is None:
            return 0.0, 0.0
        nx = (x / max(cols - 1, 1)) * 2.0 - 1.0
        ny = (y / max(rows - 1, 1)) * 2.0 - 1.0
        fx = coef_x[0] * nx + coef_x[1] * ny + coef_x[2]
        fy = coef_y[0] * nx + coef_y[1] * ny + coef_y[2]
        return float(fx) / grid_size, float(fy) / grid_size

    # -- confirmation gates ------------------------------------------------

    def _trajectory_ok(self, track: Track) -> tuple[bool, float, float, float]:
        """Does the stabilised path look like somebody walking?

        Three tests over the trailing window, and none of them is a threshold
        fitted to this clip: something has to have gone somewhere (``d_min``),
        it has to have gone there directly rather than milled about
        (straightness), and it has to have done so at a speed a person can walk.
        """
        cfg = self.cfg
        pts = track.history[-cfg.window :]
        if len(pts) < 2:
            return False, 0.0, 0.0, 0.0

        arr = np.asarray(pts, dtype=np.float32)
        steps = np.diff(arr, axis=0)
        path = float(np.hypot(steps[:, 0], steps[:, 1]).sum())
        net_v = arr[-1] - arr[0]
        net = float(np.hypot(net_v[0], net_v[1]))
        straight = net / path if path > 1e-6 else 0.0
        speed = net / max(len(arr) - 1, 1)

        ok = (
            net >= cfg.d_min_cells
            and straight >= cfg.straightness
            and cfg.v_min_cells <= speed <= cfg.v_max_cells
        )
        return ok, net, straight, speed

    # -- main --------------------------------------------------------------

    def process(self, ctx) -> list[dict]:
        cfg = self.cfg
        self._frame += 1

        if ctx.flow is None:
            for t in self.tracks:
                t.misses += 1
            self.tracks = [t for t in self.tracks if t.misses <= cfg.max_age]
            return []

        props, coef_x, coef_y, offset, (rows, cols) = self._proposals(ctx.flow, ctx)
        grid = float(ctx.grid_size)

        # Measurements: [cx, cy, w, h] in cropped-cell coordinates.
        meas = []
        for p in props:
            meas.append(
                np.array(
                    [
                        (p["x0"] + p["x1"]) / 2.0,
                        (p["y0"] + p["y1"]) / 2.0,
                        p["x1"] - p["x0"],
                        p["y1"] - p["y0"],
                    ],
                    dtype=np.float32,
                )
            )

        feats = [None] * len(meas)
        if cfg.feature_weight > 0.0 and getattr(ctx, "rgba", None) is not None:
            for i, p in enumerate(props):
                feats[i] = _appearance_feature(
                    ctx.rgba,
                    p["x0"] + offset,
                    p["y0"] + offset,
                    p["x1"] + offset,
                    p["y1"] + offset,
                    ctx.grid_size,
                    cfg.feature_bins,
                )

        # --- ego motion, then predict, both before any association ---------
        for t in self.tracks:
            img_x = t.x[0] + t.cam[0]
            img_y = t.x[1] + t.cam[1]
            dx, dy = self._camera_flow(coef_x, coef_y, img_x, img_y, rows, cols, grid)
            t.cam[0] += dx
            t.cam[1] += dy
            t.predict(cfg)

        # --- association ---------------------------------------------------
        matches, unmatched = self._associate(meas, feats)

        for ti, mi in matches:
            t = self.tracks[ti]
            z = meas[mi].copy()
            z[0] -= t.cam[0]
            z[1] -= t.cam[1]
            t.update(z, cfg)
            if feats[mi] is not None:
                t.feature = (
                    feats[mi]
                    if t.feature is None
                    else 0.9 * t.feature + 0.1 * feats[mi]
                )
                n = float(np.linalg.norm(t.feature))
                if n > 0:
                    t.feature = t.feature / n

        matched_tracks = {ti for ti, _ in matches}
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.misses += 1
            t.history.append((float(t.x[0]), float(t.x[1])))
            t.img_history.append((float(t.x[0] + t.cam[0]), float(t.x[1] + t.cam[1])))
            if len(t.history) > cfg.window * 3:
                del t.history[: len(t.history) - cfg.window * 3]
                del t.img_history[: len(t.img_history) - cfg.window * 3]

        for mi in unmatched:
            self.tracks.append(self._spawn(meas[mi], feats[mi]))

        self.tracks = [t for t in self.tracks if t.misses <= cfg.max_age]

        # --- confirmation and emission -------------------------------------
        out: list[dict] = []
        for t in self.tracks:
            ok, net, straight, speed = self._trajectory_ok(t)
            if not t.confirmed:
                if t.hits >= cfg.n_init and ok:
                    t.confirmed = True
                else:
                    continue
            elif cfg.revalidate and not ok:
                continue
            if t.misses > cfg.max_coast_emit:
                continue

            # float() everywhere below, not just for tidiness: the state is
            # float32 and json.dumps refuses numpy scalars, which fails only at
            # the very end of a whole-video run.
            cx = float(t.x[0] + t.cam[0]) + offset
            cy = float(t.x[1] + t.cam[1]) + offset
            w = max(float(t.x[2]), 1.0)
            h = max(float(t.x[3]), 1.0)

            img = np.asarray(t.img_history[-cfg.window :], dtype=np.float32)
            img_disp = (
                float(np.hypot(*(img[-1] - img[0]))) if len(img) >= 2 else 0.0
            )

            out.append(
                {
                    "left": float((cx - w / 2.0) * ctx.scale_x),
                    "top": float((cy - h / 2.0) * ctx.scale_y),
                    "width": float(w * ctx.scale_x),
                    "height": float(h * ctx.scale_y),
                    "track_id": int(t.track_id),
                    "hits": int(t.hits),
                    "misses": int(t.misses),
                    # Diagnostics for the systematic-false-positive question:
                    # a structure that misregisters identically every frame is
                    # straight and persistent in the stabilised frame but does
                    # not move in the image.
                    "stab_disp": round(float(net), 2),
                    "img_disp": round(float(img_disp), 2),
                    "straight": round(float(straight), 3),
                    "speed": round(float(speed), 3),
                }
            )
        return out

    # -- helpers -----------------------------------------------------------

    def _spawn(self, z: np.ndarray, feature) -> Track:
        x = np.zeros(6, dtype=np.float32)
        x[:4] = z
        P = np.diag(np.array([4.0, 4.0, 16.0, 16.0, 1.0, 1.0], dtype=np.float32))
        t = Track(
            track_id=self._next_id,
            x=x,
            P=P,
            cam=np.zeros(2, dtype=np.float32),
            born=self._frame,
            feature=feature,
        )
        t.history.append((float(x[0]), float(x[1])))
        t.img_history.append((float(x[0]), float(x[1])))
        self._next_id += 1
        return t

    def _associate(self, meas, feats):
        """Assign proposals to tracks by IoU + centroid distance (+ appearance).

        The gate is what does the work; the solver only settles ties. A
        proposal outside every track's gate becomes a new track, which is how
        the loose front end is allowed to be loose: a spurious proposal costs a
        tentative track that is never confirmed, not a box.
        """
        cfg = self.cfg
        n_t, n_m = len(self.tracks), len(meas)
        if n_t == 0 or n_m == 0:
            return [], list(range(n_m))

        cost = np.full((n_t, n_m), 1e6, dtype=np.float32)
        for i, t in enumerate(self.tracks):
            tx = t.x[0] + t.cam[0]
            ty = t.x[1] + t.cam[1]
            tw, th = max(float(t.x[2]), 1.0), max(float(t.x[3]), 1.0)
            tbox = (tx - tw / 2, ty - th / 2, tx + tw / 2, ty + th / 2)
            for j, z in enumerate(meas):
                d = float(np.hypot(z[0] - tx, z[1] - ty))
                if d > cfg.gate_cells:
                    continue
                mbox = (
                    z[0] - z[2] / 2,
                    z[1] - z[3] / 2,
                    z[0] + z[2] / 2,
                    z[1] + z[3] / 2,
                )
                c = cfg.dist_weight * (d / cfg.gate_cells)
                c += cfg.iou_weight * (1.0 - _iou(tbox, mbox))
                if cfg.feature_weight > 0.0:
                    c += cfg.feature_weight * _feature_distance(t.feature, feats[j])
                cost[i, j] = c

        matches = []
        if _lsa is not None:
            rows_, cols_ = _lsa(cost)
            for i, j in zip(rows_, cols_):
                if cost[i, j] < 1e5:
                    matches.append((int(i), int(j)))
        else:  # pragma: no cover - greedy fallback
            used_t, used_m = set(), set()
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            for i, j in order:
                if cost[i, j] >= 1e5:
                    break
                if i in used_t or j in used_m:
                    continue
                used_t.add(int(i))
                used_m.add(int(j))
                matches.append((int(i), int(j)))

        taken = {j for _, j in matches}
        return matches, [j for j in range(n_m) if j not in taken]
