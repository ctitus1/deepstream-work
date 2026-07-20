"""Bulk-motion detection from GPU optical flow, independent of inference.

The problem this solves is that ``nvinfer`` cannot keep up with the source. The
main branch drops frames to stay live, so anything derived from it sees a
fraction of the video. Motion is exactly the signal that must not be sampled --
a target that moves between two frames the detector skipped simply never
happened as far as that branch is concerned.

So the motion detector hangs off its own tee immediately after the decoder,
ahead of the leaky queue that feeds ``nvstreammux`` and inference. It runs its
own ``nvstreammux`` at a reduced resolution and its own ``nvof`` (the hardware
optical-flow engine), and terminates in a ``fakesink``. Nothing it does touches
the buffer the inference path is working on, which is what keeps detection and
assessment running on untouched raw frames.

The two branches therefore see different frames, and their timelines are
reconciled by buffer PTS: results go into a ``MotionStore`` keyed by the source
timestamp, and the OSD probe -- running on the display frame, much later and
several frames behind -- asks for the newest result at or before its own PTS.

What comes out is a small number of boxes around *bulk* movement:

  1. ``nvof`` produces a flow vector per 4x4 block.
  2. The global (camera) component is estimated as the per-axis median and
     subtracted, so a pan or a handheld drift leaves near-zero residual and only
     motion *relative to the scene* survives.
  3. The residual magnitude is thresholded adaptively against the frame's own
     noise floor.
  4. The mask is dilated and connected components are merged, so a person whose
     arms are moving at a different velocity from their torso comes out as one
     box rather than three.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock
from time import perf_counter

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np
import pyds

# nvof reports flow in S10.5 fixed point: 5 fractional bits, so a raw unit is
# 1/32 of a pixel at the resolution nvof ran at.
FLOW_FIXED_POINT_SCALE = 32.0

# Mid grey, deliberately distinct from the confidence ramp (red->green) the
# detection boxes use, so a motion box is never mistaken for a detection.
MOTION_COLOR = (0.65, 0.65, 0.65, 1.0)


@dataclass(frozen=True)
class MotionConfig:
    """Tuning for the motion branch.

    The defaults target one 4K source with person-sized subjects. ``width`` and
    ``height`` are what the branch's own ``nvstreammux`` scales to before
    ``nvof`` sees anything: bulk motion survives heavy downscaling, and a
    smaller frame is both faster and less sensitive to per-pixel noise.
    """

    # Resolution the branch's own mux scales to before nvof sees anything.
    #
    # Higher than it looks like it needs to be, on purpose. A person walking in
    # a 4K aerial shot covers only ~1.7 px/frame once the frame is squeezed to
    # 640x360 -- indistinguishable from compression noise. At 960x540 the same
    # walk is ~2.5 px/frame, which is what gives the temporal filter below
    # something to lock onto.
    width: int = 960
    height: int = 540
    # nvof emits one vector per grid_size x grid_size block. 4 is the only value
    # the element accepts today, but the maths below reads it rather than
    # assuming it.
    grid_size: int = 4
    # Flow cells trimmed from each edge before anything is measured.
    #
    # nvof has nothing to match against outside the frame, so its border cells
    # are unreliable, and badly so: measured on a static aerial shot the first
    # columns averaged 1.9-10.6 px/frame against an interior average of 0.3, and
    # the bottom rows ran 1.1-1.5. Whole rows cleared the motion threshold on
    # their own, producing a dense full-width band every frame -- a "detection"
    # of the edge of the image. 12 rather than a token 2 or 3 because the
    # left-hand artifact was still saturating the first cells of the field at a
    # crop of 5, and at a crop of 12 the top-left corner alone still accounted
    # for 49% of every detection made across a whole clip.
    border: int = 20

    # --- The part that actually separates a target from noise -------------
    #
    # Weight of the running average that the per-cell residual flow feeds into.
    # This is the whole discriminator. A person walking pushes their cells the
    # same way frame after frame, so the averaged vector grows to the walk's
    # real speed. Compression artefacts, sensor noise and the residue of camera
    # shake point somewhere new every frame, so their average collapses toward
    # zero however large any single frame's vector was. Thresholding the
    # *average* therefore asks "has this been moving?", where thresholding one
    # frame could only ask "did this change?".
    #
    # 0.85 gives a time constant of roughly 6 frames: long enough to bury noise,
    # short enough that a target that starts walking is picked up within a fifth
    # of a second.
    decay: float = 0.85
    # Sustained speed a cell must hold to count as moving, as a fraction of the
    # branch's frame width per frame. Kept relative so it means the same thing
    # at any resolution or source size rather than being tuned to one clip.
    #
    # 0.0022 is ~2.1 px/frame at 960 wide, and it is where the two populations
    # actually separate on real footage: mapping the sustained field frame by
    # frame, a walking person's cells sit at 2.25-5.25 px/frame while a scene
    # with nothing moving in it never gets a compact blob above ~2.25.
    min_speed_frac: float = 0.0022
    # Adaptive term: threshold is at least this many robust sigmas above the
    # field's own median, so a noisier frame demands more before it reports
    # anything.
    noise_scale: float = 6.0
    # Hysteresis. A component has to contain at least one cell above the seed
    # speed to exist at all, but once it does it grows out to every connected
    # cell above the ordinary threshold.
    #
    # One threshold cannot do both jobs. Set high enough to reject the diffuse
    # structures a static scene throws up around high-contrast edges, it also
    # clips a real target down to its fastest few cells and the box shrinks to
    # a fragment of the person. Seeding high and growing low keeps the
    # rejection while still reporting the target's true extent.
    seed_speed_frac: float = 0.0040
    # Sustained speed above which a cell is discarded as impossible rather than
    # believed. Flow that fails on untextured ground returns 20-80 px/frame.
    max_speed: float = 18.0
    # Minimum local agreement, in [0, 1], for a cell to count as motion.
    min_coherence: float = 0.5

    # --- Shape filters: placeholders, deliberately permissive -------------
    #
    # These are the knobs that would let a number be tuned until the answer
    # looked right, which is exactly the wrong way round. They are parked wide
    # open so the detections below are the honest output of the flow analysis
    # above; tighten them later against a real requirement, not to flatter a
    # demo.
    # Smallest component worth reporting, as a fraction of the flow field. This
    # is the "how many pixels are moving" filter, and it is set from measurement
    # rather than taste: over a full clip, components in the half with a person
    # walking ran to a median of 347 cells against 131 for the half where
    # nothing moved. 0.005 is ~145 cells here, between the two.
    min_area_frac: float = 0.005
    max_area_frac: float = 1.0
    max_boxes: int = 64
    min_energy: float = 0.0
    energy_ratio: float = 0.0
    min_fill: float = 0.0
    # Cells dilated around every moving cell before components are merged, and
    # the gap under which two components are joined. Both exist to keep one
    # target whose limbs move at different speeds as one box.
    dilate: int = 2
    merge_gap_frac: float = 0.02
    # Appearance descriptor: bins per channel of the RG chroma histogram.
    feature_bins: int = 6


@dataclass(frozen=True)
class MotionBox:
    """One bulk-motion region, in source-frame pixels."""

    left: float
    top: float
    width: float
    height: float
    # Mean residual speed over the component, in downscaled px/frame. Ranking
    # key, and a rough "how hard is this thing moving" readout.
    energy: float
    # Flow cells the component covers: the "how many pixels are moving"
    # number, in units of grid_size squared at the branch's resolution.
    cells: int
    # Mean sustained flow direction, source-frame px/frame.
    dx: float
    dy: float
    # L2-normalised appearance descriptor for re-identification. Empty when the
    # branch could not read pixels for this frame.
    feature: tuple[float, ...] = ()

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class MotionResult:
    pts: int
    boxes: tuple[MotionBox, ...]
    stats: MotionStats = None


@dataclass(frozen=True)
class MotionStats:
    """Per-frame numbers behind a decision, for the OSD and for debugging."""

    threshold: float = 0.0
    sustained_median: float = 0.0
    sustained_p99: float = 0.0
    moving_cells: int = 0
    total_cells: int = 0
    components: int = 0
    global_dx: float = 0.0
    global_dy: float = 0.0
    elapsed_ms: float = 0.0


class MotionState:
    """Running average of the residual flow field.

    Owned by the motion branch's probe and touched from that thread only, so it
    needs no lock. Held across frames because that history *is* the signal: a
    single frame cannot tell a walking person from a compression artefact, and
    six frames can.
    """

    def __init__(self, decay: float):
        self.decay = decay
        self.sustained: np.ndarray | None = None

    def update(self, residual: np.ndarray) -> np.ndarray:
        """Fold this frame's residual into the average and return it.

        Vectors are averaged, not magnitudes. That is the point: two opposite
        1-px flickers average to zero, while two consecutive 1-px steps in the
        same direction average to 1. Averaging magnitudes would report both as
        moving.
        """
        if self.sustained is None or self.sustained.shape != residual.shape:
            self.sustained = residual.astype(np.float32, copy=True)
        else:
            self.sustained *= self.decay
            self.sustained += (1.0 - self.decay) * residual
        return self.sustained


class MotionStore:
    """PTS-keyed handoff between the motion branch and whoever draws.

    The two branches run in different streaming threads and at different rates,
    so this is the only shared state between them and it is guarded. It keeps a
    bounded history rather than a single slot because the display frame is
    always somewhat behind the motion frame -- by the time the OSD asks, the
    motion branch has usually moved on by several frames.
    """

    def __init__(self, depth: int = 90):
        self._depth = depth
        self._lock = Lock()
        self._results: list[MotionResult] = []
        self.frames_in = 0
        self.frames_processed = 0

    def put(self, result: MotionResult) -> None:
        with self._lock:
            self._results.append(result)
            if len(self._results) > self._depth:
                del self._results[: len(self._results) - self._depth]

    def latest_at_or_before(self, pts: int) -> MotionResult | None:
        """Newest result not from the future of ``pts``.

        Falling back to the newest result when everything is older matters for
        the file-playback case, where the display branch can briefly run ahead.
        Drawing a slightly stale box beats drawing nothing.
        """
        with self._lock:
            if not self._results:
                return None
            eligible = [r for r in self._results if r.pts <= pts]
            return eligible[-1] if eligible else self._results[-1]

    def stats(self) -> tuple[int, int]:
        with self._lock:
            return self.frames_in, self.frames_processed


# ---------------------------------------------------------------------------
# Flow field -> boxes
# ---------------------------------------------------------------------------


def estimate_global_motion(flow: np.ndarray) -> tuple[float, float]:
    """Per-axis median of the flow field: the bulk translation of the scene."""
    if flow.size == 0:
        return 0.0, 0.0
    return float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))


def _background_flow(flow: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Fit the camera's own contribution to the flow field, robustly.

    A single translation cannot describe what a moving camera does: panning,
    rotating, or changing altitude makes the background flow vary smoothly
    across the frame, and subtracting one median leaves a systematic gradient
    that reads as motion at whichever edge is furthest from the middle. An
    affine model (six parameters) covers all three, so what is left over is
    motion that is genuinely independent of the camera.

    The fit is iterated with outlier rejection because the targets are in the
    data too. A plain least-squares fit would be pulled toward them and would
    partly explain away the very thing being looked for; dropping the largest
    residuals each round lets the model settle on the background alone.
    """
    rows, cols = flow.shape[:2]
    ys, xs = np.mgrid[0:rows, 0:cols]
    # Normalised coordinates keep the least-squares problem well conditioned.
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
        # 1.4826 * MAD is the normal-consistent sigma estimate; the constant
        # floor keeps a perfectly clean fit from rejecting everything.
        limit = median + 3.0 * 1.4826 * max(mad, 1e-3) + 0.25
        tightened = residual <= limit
        if int(tightened.sum()) < 16:
            break
        keep = tightened

    if coef_x is None or coef_y is None:
        return np.zeros_like(flow)

    return np.stack(
        [(basis @ coef_x).reshape(rows, cols), (basis @ coef_y).reshape(rows, cols)],
        axis=-1,
    )


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2*radius+1) square window, edges included as-is."""
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
    """How much the local flow vectors agree on a direction, in [0, 1].

    ``|sum of vectors| / sum of |vectors|``: near 1 when a neighbourhood moves
    together, near 0 when its vectors point every which way.

    This is what separates a target from a featureless patch of sky or road.
    Optical flow is ill-posed without texture to match -- the estimator has
    nothing to lock onto and returns large arbitrary vectors, which is why whole
    bands of an aerial frame came back at 25 px/frame against an interior
    average of 0.3. Those vectors are big but incoherent. A person, however
    their limbs are moving, drags a neighbourhood in one direction.
    """
    sum_x = _box_sum(residual[..., 0].copy(), radius)
    sum_y = _box_sum(residual[..., 1].copy(), radius)
    vector_sum = np.hypot(sum_x, sum_y)
    magnitude_sum = _box_sum(np.hypot(residual[..., 0], residual[..., 1]), radius)
    return vector_sum / np.maximum(magnitude_sum, 1e-6)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation by OR-ing shifted copies.

    The mask is a few thousand cells, so shifting whole arrays is cheaper than
    any per-cell work and avoids a scipy dependency for one call.

    Shifted by slice assignment rather than np.roll: roll wraps, so a blob on
    one edge reappeared on the opposite edge and chained through everything in
    between, which turned every frame into one full-width band.
    """
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
    """Label 8-connected components, returning the cell indices of each.

    Union-find over only the set cells. The mask is small and typically sparse,
    so this stays well inside the frame budget without pulling in scipy.
    """
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

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Only the four already-visited neighbours are needed: the other four are
    # covered when their own cell is processed.
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((-1, 0), (0, -1), (-1, -1), (-1, 1)):
            neighbour = index.get((int(y) + dy, int(x) + dx))
            if neighbour is not None:
                union(i, neighbour)

    groups: dict[int, list[int]] = {}
    for i in range(len(ys)):
        groups.setdefault(find(i), []).append(i)

    # Flat indices into the mask, not positions in the nonzero list. Returning
    # the latter made every caller's divmod produce sequential coordinates --
    # which read back as a band starting at (0, 0) and spanning the full width,
    # on every frame, regardless of where the motion actually was.
    flat = ys.astype(np.int64) * mask.shape[1] + xs.astype(np.int64)
    return [flat[np.asarray(members, dtype=np.int64)] for members in groups.values()]


def _merge_close(boxes: list[dict], gap: float, max_cells: float) -> list[dict]:
    """Merge boxes whose gap is under ``gap`` pixels, repeatedly until stable.

    A person is not one rigid blob. Arms and legs move at their own velocities
    and frequently break off into their own components even after dilation;
    merging by proximity puts them back with the torso, which is what "one box
    around the whole group of moving pixels" asks for.
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
                    # Gap along each axis; negative means they already overlap.
                    gap_x = max(other["x0"] - head["x1"], head["x0"] - other["x1"])
                    gap_y = max(other["y0"] - head["y1"], head["y0"] - other["y1"])
                    if gap_x > gap or gap_y > gap:
                        continue
                    # Refuse a merge that would produce a box bigger than any
                    # plausible target. Without this, two distant blobs that
                    # each pass the gap test drag the box across the frame and
                    # every later blob joins it.
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


def motion_boxes(
    flow: np.ndarray,
    cfg: MotionConfig,
    state: MotionState,
    scale_x: float,
    scale_y: float,
) -> tuple[list[MotionBox], MotionStats]:
    """Bulk-motion boxes from one optical-flow field.

    ``flow`` is (rows, cols, 2) in branch-resolution pixels per frame.
    ``scale_x``/``scale_y`` convert a flow cell back to source-frame pixels.
    ``state`` carries the running average across frames and is mutated here.
    """
    started = perf_counter()
    if flow.ndim != 3 or flow.shape[2] != 2 or flow.size == 0:
        return [], MotionStats()

    # Trim the unreliable border before it can skew the background fit, the
    # noise floor, or the mask. Coordinates are shifted back by the same offset
    # when boxes are built.
    offset = cfg.border
    if offset > 0 and flow.shape[0] > 2 * offset and flow.shape[1] > 2 * offset:
        flow = flow[offset:-offset, offset:-offset]
    else:
        offset = 0

    gx, gy = estimate_global_motion(flow)
    residual = flow - _background_flow(flow)

    # Everything from here on reads the time-averaged field, never this frame's
    # raw residual. One frame cannot distinguish a walk from a codec artefact.
    sustained = state.update(residual)
    speed = np.hypot(sustained[..., 0], sustained[..., 1])

    # MAD rather than standard deviation: the target is itself a large outlier
    # and would inflate a variance-based estimate until it no longer cleared
    # its own threshold.
    median = float(np.median(speed))
    mad = float(np.median(np.abs(speed - median)))
    # Both terms are floors, and the median belongs in the adaptive one: an
    # offset distribution shifted the whole field up while a bare k*MAD stayed
    # put, so the threshold sat inside the noise instead of above it.
    threshold = max(
        cfg.min_speed_frac * cfg.width,
        median + cfg.noise_scale * 1.4826 * mad,
    )

    plausible = (speed <= cfg.max_speed) & (_coherence(sustained) >= cfg.min_coherence)
    moving = (speed > threshold) & plausible
    seed = (speed > max(cfg.seed_speed_frac * cfg.width, threshold)) & plausible
    mask = _dilate(moving, cfg.dilate)

    rows, cols = mask.shape
    total_cells = float(rows * cols)
    max_cells = cfg.max_area_frac * total_cells

    candidates: list[dict] = []
    for members in _components(mask):
        ys, xs = np.divmod(members, cols)
        if len(members) / total_cells < cfg.min_area_frac:
            continue
        # Hysteresis: no seed cell, no component.
        if not bool(seed[ys, xs].any()):
            continue

        x0, y0 = float(xs.min()), float(ys.min())
        x1, y1 = float(xs.max() + 1), float(ys.max() + 1)
        bbox_cells = (x1 - x0) * (y1 - y0)
        if bbox_cells > max_cells or len(members) > max_cells:
            continue
        if bbox_cells > 0 and len(members) / bbox_cells < cfg.min_fill:
            continue

        cell_speed = speed[ys, xs]
        candidates.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "energy": float(cell_speed.mean()),
                "cells": int(len(members)),
                "dx": float(sustained[ys, xs, 0].mean()),
                "dy": float(sustained[ys, xs, 1].mean()),
            }
        )

    components = len(candidates)
    gap_cells = cfg.merge_gap_frac * max(rows, cols)
    candidates = _merge_close(candidates, gap_cells, max_cells)

    if candidates and (cfg.min_energy > 0.0 or cfg.energy_ratio > 0.0):
        strongest = max(c["energy"] for c in candidates)
        floor = max(cfg.min_energy, cfg.energy_ratio * strongest)
        candidates = [c for c in candidates if c["energy"] >= floor]

    # Strongest first, so a cap (when one is set) keeps the most active.
    candidates.sort(key=lambda c: c["energy"] * c["cells"], reverse=True)

    boxes = [
        MotionBox(
            left=(cand["x0"] + offset) * scale_x,
            top=(cand["y0"] + offset) * scale_y,
            width=(cand["x1"] - cand["x0"]) * scale_x,
            height=(cand["y1"] - cand["y0"]) * scale_y,
            energy=cand["energy"],
            cells=cand["cells"],
            dx=cand["dx"] * scale_x,
            dy=cand["dy"] * scale_y,
        )
        for cand in candidates[: cfg.max_boxes]
    ]

    stats = MotionStats(
        threshold=threshold,
        sustained_median=median,
        sustained_p99=float(np.percentile(speed, 99)),
        moving_cells=int(moving.sum()),
        total_cells=int(total_cells),
        components=components,
        global_dx=gx,
        global_dy=gy,
        elapsed_ms=(perf_counter() - started) * 1000.0,
    )
    return boxes, stats


def appearance_feature(
    rgba: np.ndarray,
    box: MotionBox,
    cfg: MotionConfig,
    scale_x: float,
    scale_y: float,
) -> tuple[float, ...]:
    """Appearance descriptor for re-identifying a target across frames.

    A chromaticity histogram, not an RGB one. Dividing each channel by the
    pixel's own total intensity throws away brightness and keeps only colour,
    so the same person walking from sunlight into shadow still matches
    themselves -- which is the entire job of a re-id descriptor and the thing a
    raw RGB histogram is worst at.

    Two shape terms are appended (log aspect ratio, and extent as a fraction of
    the frame) because colour alone confuses two people in similar clothing,
    while their build and distance from the camera usually differ.

    Deliberately hand-built rather than a learned embedding: it needs no model,
    no export, and no engine, and it runs in microseconds per box, which is
    what keeps this branch at source frame rate. A CNN embedding would describe
    a target better and is the natural upgrade once the boxes themselves are
    trustworthy.
    """
    height, width = rgba.shape[:2]
    # Box coordinates are in source pixels; the surface is at branch resolution.
    x0 = max(0, min(width - 1, int(box.left / scale_x * cfg.grid_size)))
    y0 = max(0, min(height - 1, int(box.top / scale_y * cfg.grid_size)))
    x1 = max(x0 + 1, min(width, int((box.left + box.width) / scale_x * cfg.grid_size)))
    y1 = max(y0 + 1, min(height, int((box.top + box.height) / scale_y * cfg.grid_size)))

    patch = rgba[y0:y1, x0:x1, :3].astype(np.float32)
    if patch.size == 0:
        return ()

    total = patch.sum(axis=2, keepdims=True)
    # Near-black pixels have no meaningful chromaticity; their ratios are pure
    # noise, so they are dropped rather than allowed to vote.
    valid = total[..., 0] > 12.0
    if not bool(valid.any()):
        return ()

    chroma = (patch / np.maximum(total, 1e-3))[valid]
    bins = max(2, cfg.feature_bins)
    # Only red and green: the three chromaticities sum to 1, so blue is
    # redundant and storing it would add a dimension with no information.
    idx_r = np.clip((chroma[:, 0] * bins).astype(np.int32), 0, bins - 1)
    idx_g = np.clip((chroma[:, 1] * bins).astype(np.int32), 0, bins - 1)
    histogram = np.bincount(idx_r * bins + idx_g, minlength=bins * bins).astype(np.float32)

    aspect = np.log(max(box.width, 1.0) / max(box.height, 1.0))
    extent = float(box.cells) / max(1.0, (width / cfg.grid_size) * (height / cfg.grid_size))
    feature = np.concatenate([histogram, np.array([aspect, extent], dtype=np.float32)])

    norm = float(np.linalg.norm(feature))
    if norm <= 0.0:
        return ()
    return tuple(float(v) for v in feature / norm)


def feature_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine distance in [0, 2] between two descriptors; lower is more alike.

    Both are L2-normalised already, so the dot product is the cosine.
    """
    if not a or not b or len(a) != len(b):
        return 2.0
    return 1.0 - float(np.dot(np.asarray(a), np.asarray(b)))


# ---------------------------------------------------------------------------
# GStreamer / DeepStream glue
# ---------------------------------------------------------------------------


def _flow_field(frame_meta) -> np.ndarray | None:
    """Pull the optical-flow field off a frame's user meta, as (rows, cols, 2)."""
    user_list = frame_meta.frame_user_meta_list

    while user_list:
        user_meta = pyds.NvDsUserMeta.cast(user_list.data)
        if user_meta.base_meta.meta_type == pyds.NVDS_OPTICAL_FLOW_META:
            of_meta = pyds.NvDsOpticalFlowMeta.cast(user_meta.user_meta_data)
            # Returns the raw NvOFFlowVector pairs; shape varies by binding
            # version, so reshape from the meta's own dimensions rather than
            # trusting what came back.
            raw = pyds.get_optical_flow_vectors(of_meta)
            rows, cols = int(of_meta.rows), int(of_meta.cols)
            if rows <= 0 or cols <= 0 or raw is None:
                return None
            flow = np.asarray(raw, dtype=np.float32).reshape(rows, cols, 2)
            return flow / FLOW_FIXED_POINT_SCALE
        user_list = user_list.next

    return None


def _with_features(boxes, buf, frame_meta, cfg, scale_x, scale_y):
    """Attach appearance descriptors, or return the boxes untouched.

    Mapping the surface can fail (wrong memory type, a frame already unmapped),
    and a missing descriptor is not worth losing a detection over -- MotionBox
    treats an empty feature as "not computed" rather than as a match failure.
    """
    if not boxes:
        return boxes

    try:
        surface = pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
        # asarray, not array(copy=False): under numpy 2 the latter *raises*
        # when a copy would be needed rather than quietly making one, so every
        # descriptor failed and every box came back with no feature at all.
        rgba = np.asarray(surface)
    except Exception:
        return boxes

    described = []
    for box in boxes:
        try:
            feature = appearance_feature(rgba, box, cfg, scale_x, scale_y)
        except Exception:
            feature = ()
        described.append(replace(box, feature=feature) if feature else box)

    try:
        pyds.unmap_nvds_buf_surface(hash(buf), frame_meta.batch_id)
    except Exception:
        pass

    return described


def motion_probe(
    cfg: MotionConfig,
    store: MotionStore,
    src_w: int,
    src_h: int,
    debug: bool = False,
):
    """Probe for the motion branch: flow field in, boxes into the store.

    Attached on the ``nvof`` source pad. Reads only -- it must never modify the
    buffer, because the inference branch is working on the same decoded frames
    and the whole design depends on those staying raw.
    """
    # A flow cell covers grid_size downscaled pixels, and the branch's mux
    # scaled the source down to cfg.width x cfg.height. Compose both to land
    # back in source-frame coordinates.
    scale_x = cfg.grid_size * (src_w / float(cfg.width))
    scale_y = cfg.grid_size * (src_h / float(cfg.height))
    # One accumulator for the branch; only this probe's thread touches it.
    state = MotionState(cfg.decay)

    def _probe(_pad, info, _data):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            store.frames_in += 1

            flow = _flow_field(frame_meta)
            if flow is not None:
                boxes, stats = motion_boxes(flow, cfg, state, scale_x, scale_y)
                boxes = _with_features(
                    boxes, buf, frame_meta, cfg, scale_x, scale_y
                )
                store.put(
                    MotionResult(pts=int(buf.pts), boxes=tuple(boxes), stats=stats)
                )
                store.frames_processed += 1

                if debug:
                    detail = " ".join(
                        f"[{b.left:.0f},{b.top:.0f} {b.width:.0f}x{b.height:.0f} "
                        f"speed={b.energy:.2f} cells={b.cells} "
                        f"dir=({b.dx:+.1f},{b.dy:+.1f}) feat={len(b.feature)}]"
                        for b in boxes
                    )
                    print(
                        f"MOTION frame={store.frames_processed} "
                        f"boxes={len(boxes)} thr={stats.threshold:.3f} "
                        f"med={stats.sustained_median:.3f} "
                        f"p99={stats.sustained_p99:.2f} "
                        f"moving={stats.moving_cells}/{stats.total_cells} "
                        f"global=({stats.global_dx:+.2f},{stats.global_dy:+.2f}) "
                        f"ms={stats.elapsed_ms:.1f} {detail}",
                        flush=True,
                    )

            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe


def add_motion_box(batch_meta, frame_meta, box: MotionBox) -> None:
    """Draw one grey motion rectangle plus its label."""
    x1, y1 = round(box.left), round(box.top)
    x2, y2 = round(box.left + box.width), round(box.top + box.height)

    frame_h = int(frame_meta.source_frame_height or 1080)
    font_size = max(1, round(frame_h * 0.001))

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
        line.line_width = 2
        line.line_color.set(*MOTION_COLOR)

    text = meta.text_params[0]
    text.display_text = (
        f"motion speed={box.energy:.1f} cells={box.cells} "
        f"dir=({box.dx:+.0f},{box.dy:+.0f})"
    )
    text.x_offset = max(0, x1)
    # Below the top edge rather than above it, so the label does not collide
    # with the detection label sitting above the same region.
    text.y_offset = max(0, y1 + 2)
    text.font_params.font_name = "Serif"
    text.font_params.font_size = font_size
    text.font_params.font_color.set(*MOTION_COLOR)
    text.set_bg_clr = 1
    text.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

    pyds.nvds_add_display_meta_to_frame(frame_meta, meta)


def motion_overlay_probe(store: MotionStore):
    """Probe for the display path: draw the motion boxes for this frame's time.

    Attached on the OSD sink pad, which is downstream of every inference stage,
    so nothing here can reach a frame that has yet to be inferred on. Being last
    also means these rectangles are added to the display meta after the
    detection boxes, and so are drawn over them.

    The result it draws comes from a different frame than the one on screen --
    the motion branch runs ahead, unthrottled by inference -- which is why the
    lookup is by PTS rather than by frame number.
    """

    def _probe(_pad, info, _data):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        result = store.latest_at_or_before(int(buf.pts))
        if result is None or not result.boxes:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            for box in result.boxes:
                add_motion_box(batch_meta, frame_meta, box)
            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe
