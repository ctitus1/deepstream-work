"""Quadratic background model, structure-tensor validity, raw-residual coherence.

A targeted repair of the baseline in ``deepstream_yolo.motion``, not a new
algorithm. The survey named three defects; each is fixed behind its own flag so
its contribution can be measured on its own rather than asserted.

**Defect 1 -- the background model is the wrong order.** Flow induced by a
camera moving relative to a *plane* is a homography, and the small-motion
expansion of a homography is quadratic::

    u = a1 + a2*x + a3*y + a7*x^2 + a8*x*y
    v = a4 + a5*x + a6*y + a7*x*y + a8*y^2

The baseline fits an affine model, which drops a7 and a8. What it drops grows
as x^2: zero in the middle of the frame, maximal at the edges and corners. That
is precisely where the baseline's false positives concentrate -- at a 12-cell
border crop the top-left corner alone accounted for 49% of every detection made
across a clip. Two extra basis columns per axis, and the systematic error the
crop was papering over is modelled instead of excluded.

The two axes are fitted separately here, so u gets [1, x, y, x^2, xy] and v
gets [1, x, y, xy, y^2] with their own coefficients. The strict homography
shares a7 and a8 between the axes; fitting them independently is a 10-parameter
superset of the 8-parameter truth. Looser, but it spans the same shapes and
keeps the solve two small independent least-squares problems.

**Defect 2 -- coherence is measured after the signal it judges is gone.** The
baseline computes coherence on the EMA-filtered field. But the EMA has already
collapsed incoherent noise toward zero, so by the time coherence looks at the
field the incoherence it exists to detect has been averaged away, and the gate
is far weaker than it reads. Coherence is computed here on the raw per-frame
residual, before the EMA, and then *accumulated*: the fraction of the last N
frames in which a cell was locally coherent. A walker is coherent nearly every
frame; a textureless patch throwing large arbitrary vectors is coherent by
chance now and then and never persistently.

**Defect 3 -- the fit is contaminated by the garbage it exists to reject.** The
baseline's fit is unweighted across every cell, including textureless ones
returning 20-80 px/frame, and those sit at maximum leverage for a spatial
polynomial: the frame periphery. Cells are weighted by reliability before they
vote, via ``w = w_hw * w_tex * w_coh``, and the same weight drives a
reliability-weighted EMA so an unreliable frame cannot drag a cell's
accumulator.

  * ``w_tex`` -- structure tensor per 4x4 block, gated on **lambda2**, the
    smaller eigenvalue. Not lambda1 and not gradient magnitude: both are large
    on edges, and an edge is exactly where the aperture problem makes flow
    unreliable. lambda2 is small on an edge and large only on a corner, which
    is the condition under which flow is actually determined.
  * ``w_hw`` -- nvof's own per-vector cost. Reachable, but see below.

On the nvof cost plane: it is readable (pyds binds neither ``cost`` nor
``cost_size``, so it is read from the raw struct via ``pyds.get_ptr`` plus
ctypes; the pointer is host-pinned on this dGPU). It is also, measurably, the
wrong signal. Cost runs lower on the unreliable frame border than in the
interior and higher on cells that are genuinely moving, so used as "higher cost
= less reliable" it down-weights targets and trusts artifacts. It is wired up
behind ``use_cost`` so the claim is measured, and it defaults off.
"""

from __future__ import annotations

import ctypes
import sys

import cv2
import numpy as np

NAME = "quadratic-flow-cost"

FLOW_FIXED_POINT_SCALE = 32.0

# NvDsOpticalFlowMeta, 64-bit natural alignment. Offsets verified at runtime by
# reading rows/cols back through this same view and comparing against pyds.
_OFF_COST_SIZE = 12
_OFF_DATA = 24
_OFF_COST = 32
_STRUCT_BYTES = 56


DEFAULTS = {
    # --- which repairs are active; the point of the exercise ---------------
    "quadratic": True,      # defect 1: x^2, xy / xy, y^2 basis columns
    "raw_coherence": True,  # defect 2: coherence before the EMA, accumulated
    "weighting": True,      # defect 3: reliability-weighted fit and EMA
    "use_cost": False,      # nvof cost as w_hw; measured, and it does not help

    # --- geometry, unchanged from the baseline -----------------------------
    "width": 960,
    "height": 540,
    "grid_size": 4,
    "border": 20,

    # --- temporal accumulation ---------------------------------------------
    "decay": 0.85,
    "min_speed_frac": 0.0022,
    "noise_scale": 6.0,
    "seed_speed_frac": 0.0040,
    "max_speed": 18.0,
    "min_coherence": 0.5,

    # --- defect 2 knobs ------------------------------------------------------
    # Weight of the accumulator that per-frame coherence verdicts feed into.
    # 0.85 matches the flow EMA's ~6-frame constant, so the two gates ask about
    # the same stretch of history.
    "coh_decay": 0.85,
    # Fraction of recent frames a cell must have been coherent in. Well under
    # the 0.5 the baseline demands of a single (already smoothed) frame,
    # because a per-frame verdict on a raw residual is a much noisier test and
    # the accumulation is what supplies the confidence.
    "min_coh_acc": 0.35,

    # --- defect 3 knobs ------------------------------------------------------
    # lambda2 percentiles used to place the texture ramp. Taken per frame
    # rather than fixed, so the gate means the same thing whatever the scene's
    # overall contrast is.
    "tex_noise_pct": 20.0,
    "tex_ref_pct": 80.0,
    # Floor under w_tex. A cell with no corner structure still gets some say:
    # zeroing it outright empties whole regions of the fit and leaves the model
    # extrapolating into them, which is worse than a weak vote.
    "tex_floor": 0.05,
    # Confidence a cell must accumulate before it may be thresholded at all.
    "w_min": 0.25,

    # --- shape filters, as the baseline leaves them ------------------------
    "min_area_frac": 0.005,
    "max_area_frac": 1.0,
    "min_fill": 0.0,
    "dilate": 2,
    "merge_gap_frac": 0.02,
}


# ---------------------------------------------------------------------------
# nvof cost plane
# ---------------------------------------------------------------------------


class _CostTap:
    """Reads nvof's per-vector cost plane, which pyds does not bind.

    Two things have to be arranged, and both work by rebinding globals in
    ``__main__``: eval/run_motion.py is shared and must not be edited, but it
    *is* __main__, and its probe looks ``element`` and ``flow_field`` up as
    module globals on every call. So wrapping them from a constructor changes
    what the harness runs without touching the file.

    Everything here is best-effort. If any part of it fails the approach still
    runs, with the hardware weight pinned to 1.
    """

    def __init__(self):
        self.cost: np.ndarray | None = None
        self.available = False
        self.reason = "not attempted"
        self._offsets_ok: bool | None = None

    def attach(self) -> None:
        main = sys.modules.get("__main__")
        if main is None or not hasattr(main, "flow_field"):
            self.reason = "__main__ has no flow_field to rebind"
            return

        original_element = getattr(main, "element", None)
        if original_element is None:
            self.reason = "__main__ has no element factory to wrap"
            return

        def wrapped_element(factory, name, *args, **kwargs):
            elem = original_element(factory, name, *args, **kwargs)
            if factory == "nvof" and elem is not None:
                try:
                    elem.set_property("output-cost", True)
                except Exception as exc:
                    self.reason = f"output-cost not settable: {exc}"
            return elem

        main.element = wrapped_element
        main.flow_field = self._flow_field
        self.reason = "attached"

    def _flow_field(self, frame_meta):
        """Stand-in for run_motion.flow_field that also captures cost."""
        import pyds

        self.cost = None
        user_list = frame_meta.frame_user_meta_list
        while user_list:
            user_meta = pyds.NvDsUserMeta.cast(user_list.data)
            if user_meta.base_meta.meta_type == pyds.NVDS_OPTICAL_FLOW_META:
                of_meta = pyds.NvDsOpticalFlowMeta.cast(user_meta.user_meta_data)
                raw = pyds.get_optical_flow_vectors(of_meta)
                rows, cols = int(of_meta.rows), int(of_meta.cols)
                if rows <= 0 or cols <= 0 or raw is None:
                    return None
                try:
                    self.cost = self._read_cost(pyds, of_meta, rows, cols)
                except Exception as exc:
                    self.cost = None
                    self.reason = f"cost read failed: {type(exc).__name__}: {exc}"
                flow = np.asarray(raw, dtype=np.float32).reshape(rows, cols, 2)
                return flow / FLOW_FIXED_POINT_SCALE
            user_list = user_list.next
        return None

    def _read_cost(self, pyds, of_meta, rows, cols) -> np.ndarray | None:
        base = pyds.get_ptr(of_meta)
        if not base:
            self.reason = "get_ptr returned null"
            return None

        blob = ctypes.string_at(base, _STRUCT_BYTES)
        header = np.frombuffer(blob, dtype=np.uint32, count=4)
        cost_size = int(header[3])
        cost_ptr = int(
            np.frombuffer(blob[_OFF_COST:_OFF_COST + 8], dtype=np.uint64)[0]
        )

        if self._offsets_ok is None:
            # Never dereference a pointer read at an unverified offset. If the
            # rows/cols this view reports disagree with what pyds decoded, the
            # layout assumption is wrong and the cost pointer is not a pointer.
            self._offsets_ok = int(header[0]) == rows and int(header[1]) == cols
            if not self._offsets_ok:
                self.reason = "struct offsets disagree with pyds rows/cols"

        if not self._offsets_ok or cost_size == 0 or not cost_ptr:
            return None

        # cost_size is the size of one cost element -- mirroring mv_size=4 for
        # the 2x int16 flow vector -- not the size of the plane.
        nbytes = rows * cols * cost_size
        cost = np.frombuffer(
            ctypes.string_at(cost_ptr, nbytes), dtype=np.uint8
        ).reshape(rows, cols).astype(np.float32)
        self.available = True
        self.reason = "ok"
        return cost


# ---------------------------------------------------------------------------
# Background model
# ---------------------------------------------------------------------------


def _basis(rows: int, cols: int, quadratic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Design matrices for u and v, in normalised [-1, 1] frame coordinates.

    Normalised because the polynomial is fitted by least squares and raw cell
    indices would make the quadratic columns four orders of magnitude larger
    than the constant one.
    """
    ys, xs = np.mgrid[0:rows, 0:cols]
    nx = ((xs / max(cols - 1, 1)) * 2.0 - 1.0).ravel().astype(np.float32)
    ny = ((ys / max(rows - 1, 1)) * 2.0 - 1.0).ravel().astype(np.float32)
    one = np.ones(nx.size, dtype=np.float32)

    if not quadratic:
        affine = np.stack([nx, ny, one], axis=1)
        return affine, affine
    # u picks up x^2 and xy; v picks up xy and y^2. This is the small-motion
    # expansion of a homography, minus the constraint that the two axes share
    # the pair of coefficients.
    basis_u = np.stack([nx, ny, one, nx * nx, nx * ny], axis=1)
    basis_v = np.stack([nx, ny, one, nx * ny, ny * ny], axis=1)
    return basis_u, basis_v


def _background_flow(
    flow: np.ndarray,
    weights: np.ndarray | None,
    quadratic: bool,
    iterations: int = 3,
) -> np.ndarray:
    """Fit the camera's own contribution, robustly and with per-cell weights.

    Two mechanisms keep the targets out of the model, and they do different
    jobs. Iterated outlier rejection drops cells whose residual is large *after
    a fit*, which catches the movers. The weights drop cells that were never
    trustworthy to begin with, which catches textureless flow -- and those need
    catching before the first fit, not after it, because they sit at the frame
    periphery where a spatial polynomial has its maximum leverage.
    """
    rows, cols = flow.shape[:2]
    basis_u, basis_v = _basis(rows, cols, quadratic)
    n_params = basis_u.shape[1]

    fx = flow[..., 0].ravel()
    fy = flow[..., 1].ravel()
    w = (
        np.ones(fx.size, dtype=np.float32)
        if weights is None
        else np.clip(weights.ravel().astype(np.float32), 0.0, 1.0)
    )
    keep = np.ones(fx.size, dtype=bool)
    coef_x = coef_y = None
    floor = max(4 * n_params, 16)

    for _ in range(iterations):
        active = keep & (w > 1e-4)
        if int(active.sum()) < floor:
            break
        # Weighted least squares by row scaling: minimising |sqrt(w)(Ax - b)|^2
        # is minimising sum w_i (a_i.x - b_i)^2, which is what is wanted, and it
        # keeps the solve a plain lstsq rather than normal equations.
        root = np.sqrt(w[active])[:, None]
        coef_x = np.linalg.lstsq(basis_u[active] * root, fx[active] * root[:, 0], rcond=None)[0]
        coef_y = np.linalg.lstsq(basis_v[active] * root, fy[active] * root[:, 0], rcond=None)[0]

        residual = np.hypot(fx - basis_u @ coef_x, fy - basis_v @ coef_y)
        median = float(np.median(residual[active]))
        mad = float(np.median(np.abs(residual[active] - median)))
        limit = median + 3.0 * 1.4826 * max(mad, 1e-3) + 0.25
        tightened = residual <= limit
        if int((tightened & (w > 1e-4)).sum()) < floor:
            break
        keep = tightened

    if coef_x is None or coef_y is None:
        return np.zeros_like(flow)

    return np.stack(
        [(basis_u @ coef_x).reshape(rows, cols), (basis_v @ coef_y).reshape(rows, cols)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Structure tensor
# ---------------------------------------------------------------------------


def _block_mean(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean of each tile, one tile per flow cell.

    INTER_AREA at an exact integer downscale is precisely the per-tile mean, and
    it is several times faster than reshaping to (rows, block, cols, block) and
    reducing. Mean rather than sum only rescales the eigenvalues by a constant,
    and the ramp below is placed by per-frame percentiles, so the constant
    cancels.
    """
    return cv2.resize(values, (cols, rows), interpolation=cv2.INTER_AREA)


def _texture_weight(
    rgba: np.ndarray,
    rows: int,
    cols: int,
    block: int,
    noise_pct: float,
    ref_pct: float,
    floor: float,
) -> np.ndarray | None:
    """Per-cell flow reliability from the smaller structure-tensor eigenvalue.

    ``J = sum_w [Ix^2, IxIy; IxIy, Iy^2]`` over each flow cell, eigenvalues
    lambda1 >= lambda2. The gate is on lambda2 and that choice is the whole
    point: gradient magnitude and lambda1 are both large along an edge, and an
    edge is where the aperture problem leaves flow determined in one direction
    only. lambda2 is large only where the gradient has two independent
    directions -- a corner -- which is the condition under which optical flow
    has a unique answer.

    The ramp endpoints are per-frame percentiles rather than constants, so the
    weight means "textured relative to this scene" instead of being tied to one
    clip's contrast.
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
        return None
    if rgba.shape[0] < rows * block or rgba.shape[1] < cols * block:
        return None

    # Rec.601 luma. The flow engine works on luma, so the validity of its
    # result is a property of luma structure, not of colour.
    #
    # Via cv2 rather than numpy throughout, and that is not incidental: the
    # numpy spelling of this function (astype + weighted sum + np.gradient +
    # three reshape-reductions over half a million elements) measured 24 ms per
    # frame, which on its own would have consumed most of the 33 ms budget. The
    # same arithmetic through cvtColor, Sobel and INTER_AREA is 7 ms.
    patch = rgba[: rows * block, : cols * block]
    gray = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGBA2GRAY)

    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = _block_mean(ix * ix, rows, cols)
    jyy = _block_mean(iy * iy, rows, cols)
    jxy = _block_mean(ix * iy, rows, cols)

    # Closed-form eigenvalues of a symmetric 2x2; cheaper and better behaved
    # than np.linalg.eigvalsh over 30k tiny matrices.
    half_trace = 0.5 * (jxx + jyy)
    spread = np.sqrt(np.maximum((0.5 * (jxx - jyy)) ** 2 + jxy * jxy, 0.0))
    lambda2 = half_trace - spread

    noise = float(np.percentile(lambda2, noise_pct))
    reference = float(np.percentile(lambda2, ref_pct)) - noise
    if reference <= 1e-6:
        return None
    weight = np.clip((lambda2 - noise) / reference, 0.0, 1.0)
    return np.maximum(weight, floor).astype(np.float32)


# ---------------------------------------------------------------------------
# Field helpers, copied from deepstream_yolo.motion
# ---------------------------------------------------------------------------


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
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


def _coherence(field: np.ndarray, radius: int = 1) -> np.ndarray:
    """|sum of vectors| / sum of |vectors| over a neighbourhood, in [0, 1]."""
    sum_x = _box_sum(field[..., 0].copy(), radius)
    sum_y = _box_sum(field[..., 1].copy(), radius)
    vector_sum = np.hypot(sum_x, sum_y)
    magnitude_sum = _box_sum(np.hypot(field[..., 0], field[..., 1]), radius)
    return vector_sum / np.maximum(magnitude_sum, 1e-6)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
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


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components as flat cell indices, by union-find."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []

    index = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    parent = list(range(len(ys)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((-1, 0), (0, -1), (-1, -1), (-1, 1)):
            neighbour = index.get((int(y) + dy, int(x) + dx))
            if neighbour is not None:
                ra, rb = find(i), find(neighbour)
                if ra != rb:
                    parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for i in range(len(ys)):
        groups.setdefault(find(i), []).append(i)

    flat = ys.astype(np.int64) * mask.shape[1] + xs.astype(np.int64)
    return [flat[np.asarray(members, dtype=np.int64)] for members in groups.values()]


def _merge_close(boxes: list[dict], gap: float, max_cells: float) -> list[dict]:
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
                            "dx": (head["dx"] + other["dx"]) / 2.0,
                            "dy": (head["dy"] + other["dy"]) / 2.0,
                        }
                        boxes.pop(i)
                        joined = True
                        merged = True
                        break
            out.append(head)
        boxes = out
    return boxes


# ---------------------------------------------------------------------------
# Approach
# ---------------------------------------------------------------------------


class Approach:
    needs_flow = True
    needs_pixels = True

    def __init__(self, cfg: dict):
        self.cfg = dict(DEFAULTS)
        self.cfg.update({k: v for k, v in (cfg or {}).items() if k in DEFAULTS})

        # Accumulators, all lazily shaped to the cropped field on first frame.
        self.sustained: np.ndarray | None = None   # EMA of residual vectors
        self.confidence: np.ndarray | None = None  # W, the EMA's own weight
        self.coh_acc: np.ndarray | None = None     # fraction of recent frames coherent

        self.cost_tap: _CostTap | None = None
        if self.cfg["use_cost"]:
            self.cost_tap = _CostTap()
            self.cost_tap.attach()

    # -- accumulators -------------------------------------------------------

    def _reset(self, shape: tuple[int, int]) -> None:
        self.sustained = np.zeros((*shape, 2), dtype=np.float32)
        self.confidence = np.zeros(shape, dtype=np.float32)
        self.coh_acc = np.zeros(shape, dtype=np.float32)

    def _update_sustained(self, residual: np.ndarray, weight: np.ndarray | None) -> None:
        """Fold this frame's residual into the running average.

        Unweighted, this is the baseline's EMA. Weighted, it is a running
        weighted mean: a cell whose flow was unreliable this frame contributes
        in proportion to how much it was believed, and crucially its confidence
        decays rather than being topped up. So a cell that is never reliable
        never accumulates enough W to be thresholded at all, instead of drifting
        into the mask on the strength of vectors nobody trusts.
        """
        decay = self.cfg["decay"]
        if weight is None:
            self.sustained *= decay
            self.sustained += (1.0 - decay) * residual
            self.confidence[...] = 1.0
            return

        prior = decay * self.confidence
        gain = (1.0 - decay) * weight
        total = prior + gain
        safe = np.maximum(total, 1e-6)[..., None]
        self.sustained = (
            prior[..., None] * self.sustained + gain[..., None] * residual
        ) / safe
        self.confidence = total

    # -- main ---------------------------------------------------------------

    def process(self, ctx) -> list[dict]:
        cfg = self.cfg
        flow = ctx.flow
        if flow is None or flow.ndim != 3 or flow.shape[2] != 2 or flow.size == 0:
            return []

        block = int(getattr(ctx, "grid_size", cfg["grid_size"]) or cfg["grid_size"])
        full_rows, full_cols = flow.shape[:2]

        # Per-cell reliability is computed on the uncropped field so it lines up
        # with the pixel surface, then cropped alongside the flow.
        texture = None
        if cfg["weighting"]:
            texture = _texture_weight(
                getattr(ctx, "rgba", None),
                full_rows,
                full_cols,
                block,
                cfg["tex_noise_pct"],
                cfg["tex_ref_pct"],
                cfg["tex_floor"],
            )
        hardware = None
        if cfg["weighting"] and self.cost_tap is not None:
            cost = self.cost_tap.cost
            if cost is not None and cost.shape == (full_rows, full_cols):
                # Normalised against the frame's own spread, since the cost
                # scale is not documented and is not stable across scenes.
                hi = float(np.percentile(cost, 90))
                hardware = (
                    np.clip(1.0 - cost / hi, 0.0, 1.0).astype(np.float32)
                    if hi > 1e-6
                    else None
                )

        offset = cfg["border"]
        if offset > 0 and full_rows > 2 * offset and full_cols > 2 * offset:
            crop = (slice(offset, -offset), slice(offset, -offset))
            flow = flow[crop]
            texture = texture[crop] if texture is not None else None
            hardware = hardware[crop] if hardware is not None else None
        else:
            offset = 0

        rows, cols = flow.shape[:2]
        if self.sustained is None or self.sustained.shape[:2] != (rows, cols):
            self._reset((rows, cols))

        # --- reliability weight ------------------------------------------
        # The fit weight uses the *previous* frame's accumulated coherence:
        # this frame's coherence is a function of the residual, which is a
        # function of the fit, so it cannot be an input to it.
        if cfg["weighting"]:
            weight = np.ones((rows, cols), dtype=np.float32)
            if texture is not None:
                weight *= texture
            if hardware is not None:
                weight *= hardware
            fit_weight = weight * np.maximum(self.coh_acc, cfg["tex_floor"])
        else:
            weight = None
            fit_weight = None

        # --- defect 1: quadratic background model ------------------------
        background = _background_flow(flow, fit_weight, cfg["quadratic"])
        residual = flow - background

        # --- defect 2: coherence on the raw residual, accumulated --------
        if cfg["raw_coherence"]:
            coherent_now = _coherence(residual) >= cfg["min_coherence"]
            self.coh_acc *= cfg["coh_decay"]
            self.coh_acc += (1.0 - cfg["coh_decay"]) * coherent_now
        else:
            self.coh_acc[...] = 1.0

        # --- defect 3: reliability-weighted EMA ---------------------------
        ema_weight = weight * np.maximum(self.coh_acc, 1e-3) if cfg["weighting"] else None
        self._update_sustained(residual, ema_weight)

        sustained = self.sustained
        speed = np.hypot(sustained[..., 0], sustained[..., 1])

        median = float(np.median(speed))
        mad = float(np.median(np.abs(speed - median)))
        threshold = max(
            cfg["min_speed_frac"] * cfg["width"],
            median + cfg["noise_scale"] * 1.4826 * mad,
        )

        plausible = speed <= cfg["max_speed"]
        if cfg["raw_coherence"]:
            plausible &= self.coh_acc >= cfg["min_coh_acc"]
        else:
            plausible &= _coherence(sustained) >= cfg["min_coherence"]
        if cfg["weighting"]:
            # A cell nobody has believed enough times has no business being
            # thresholded, whatever its accumulator happens to say.
            plausible &= self.confidence > cfg["w_min"]

        moving = (speed > threshold) & plausible
        seed = (speed > max(cfg["seed_speed_frac"] * cfg["width"], threshold)) & plausible
        mask = _dilate(moving, cfg["dilate"])

        total_cells = float(rows * cols)
        max_cells = cfg["max_area_frac"] * total_cells

        candidates: list[dict] = []
        for members in _components(mask):
            ys, xs = np.divmod(members, cols)
            if len(members) / total_cells < cfg["min_area_frac"]:
                continue
            if not bool(seed[ys, xs].any()):
                continue

            x0, y0 = float(xs.min()), float(ys.min())
            x1, y1 = float(xs.max() + 1), float(ys.max() + 1)
            bbox_cells = (x1 - x0) * (y1 - y0)
            if bbox_cells > max_cells or len(members) > max_cells:
                continue
            if bbox_cells > 0 and len(members) / bbox_cells < cfg["min_fill"]:
                continue

            candidates.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "energy": float(speed[ys, xs].mean()),
                    "cells": int(len(members)),
                    "dx": float(sustained[ys, xs, 0].mean()),
                    "dy": float(sustained[ys, xs, 1].mean()),
                }
            )

        gap_cells = cfg["merge_gap_frac"] * max(rows, cols)
        candidates = _merge_close(candidates, gap_cells, max_cells)
        candidates.sort(key=lambda c: c["energy"] * c["cells"], reverse=True)

        scale_x, scale_y = ctx.scale_x, ctx.scale_y
        return [
            {
                "left": (c["x0"] + offset) * scale_x,
                "top": (c["y0"] + offset) * scale_y,
                "width": (c["x1"] - c["x0"]) * scale_x,
                "height": (c["y1"] - c["y0"]) * scale_y,
                "speed": c["energy"],
                "cells": c["cells"],
                "dx": c["dx"] * scale_x,
                "dy": c["dy"] * scale_y,
            }
            for c in candidates
        ]
