"""fastMCD: motion-compensated dual-mode background modelling.

Yi et al., *Detection of Moving Objects with Non-Stationary Cameras in 5.8ms*
(CVPRW 2013). Reference implementation: https://github.com/kmyid/fastMCD --
the equations below were read off ``python/ProbModel.py`` and ``src/params.hpp``
there rather than from the paper, and the deviations are called out inline.

Why a background model rather than frame differencing: differencing only sees
change across a couple of frames, so a target moving slowly relative to the
sensor noise never separates. A model integrates evidence over hundreds of
frames, and a person who has been standing still is flagged the instant they
deviate from an established mean.

Why an age-weighted single Gaussian per block rather than MOG2/KNN: those are
better models for a *static* camera, but they cannot be resampled. When the
camera moves the model has to be warped into the new view, and a per-pixel
Gaussian *mixture* has no coherent bilinear interpolation -- mixing two
mixtures is not a mixture of the same order. A single (mean, variance, age)
triple does interpolate, with the mean-spread correction in ``_compensate``
keeping the result honest about how much the mixing widened it.

Mechanically, per frame:

  1. Block grid over the frame; each block carries two models, *apparent* and
     *candidate*, each ``(mean, var, age)``.
  2. Frame-to-frame homography from KLT tracks, then every model is resampled
     through it -- the bilinear mixture of the four blocks it overlaps.
  3. The block's observed mean updates whichever model it matches, with a
     ``1/(age+1)`` gain: converges immediately when the model is young,
     barely moves when it is old.
  4. When the candidate outlives the apparent model they swap. That is what
     stops a person who stops walking from being absorbed on the next frame,
     while still letting genuine scene change take over.
  5. Foreground where the observation is far from the apparent model *and*
     that model is old enough to be worth believing.

Never compose homographies across frames: the error compounds multiplicatively
and the whole frame lights up within a few seconds. Each frame warps the model
by its own single-step H and lets the age mechanism absorb the residual.
"""

from __future__ import annotations

import cv2
import numpy as np

NAME = "fastmcd"

IDENTITY = np.eye(3, dtype=np.float32)


class Approach:
    needs_flow = False
    needs_pixels = True

    def __init__(self, cfg: dict):
        c = dict(cfg or {})

        # Reference params.hpp values, kept as the defaults so any deviation is
        # visible as an override rather than buried in the source.
        self.block = int(c.get("block", 4))  # BLOCK_SIZE
        self.init_var = float(c.get("init_var", 400.0))  # INIT_BG_VAR
        self.min_var = float(c.get("min_var", 25.0))  # MIN_BG_VAR
        self.max_age = float(c.get("max_age", 30.0))  # MAX_BG_AGE
        self.theta_match = float(c.get("theta_match", 2.0))  # VAR_THRESH_MODEL_MATCH
        self.theta_fg = float(c.get("theta_fg", 4.0))  # VAR_THRESH_FG_DETERMINE

        # Minimum apparent-model age before a block may fire. The reference
        # uses >1, which is barely a guard at all; raising it trades the frames
        # just after a pan (where every model is freshly initialised) against
        # blindness to targets entering at the frame edge.
        self.age_min = float(c.get("age_min", 1.0))

        # Not in the reference: penalise age when a model is assembled from
        # four dissimilar sources, since a mixture is less trustworthy than any
        # pure model. 0 disables, reproducing the reference exactly.
        self.mix_purity = float(c.get("mix_purity", 0.0))

        # Registration. Features are detected once and then tracked forward,
        # re-detected only when the herd thins or has been running long enough
        # to have drifted and clustered -- detection costs ~6ms a frame and
        # tracking costs well under one.
        self.max_corners = int(c.get("max_corners", 600))
        self.corner_quality = float(c.get("corner_quality", 0.01))
        self.corner_min_dist = int(c.get("corner_min_dist", 8))
        self.ransac_thresh = float(c.get("ransac_thresh", 1.5))
        self.min_inliers = int(c.get("min_inliers", 12))
        self.redetect_below = int(c.get("redetect_below", 250))
        self.redetect_every = int(c.get("redetect_every", 20))

        # Post-processing, in branch-resolution pixels.
        self.dilate = int(c.get("dilate", 3))
        self.min_area = int(c.get("min_area", 48))
        self.merge_gap = int(c.get("merge_gap", 16))
        # Applied after merging, where the fragments of one target have been
        # put back together. Measured on this footage: boxes that land on the
        # mover have a median area of 706 branch px, boxes that land on a
        # stationary person 121 and boxes that land on nothing 241. A floor
        # here is the single most effective filter available, and it is a
        # statement about how big a person is, not a cap on box count.
        self.min_box_area = int(c.get("min_box_area", 300))

        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.age: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.prev_feats: np.ndarray | None = None
        self._grid = None
        self._since_detect = 0
        self.reg_failures = 0
        self.reg_identity = 0

    # ------------------------------------------------------------------ setup

    def _init_grid(self, gh: int, gw: int) -> None:
        b = self.block
        xs = (np.arange(gw, dtype=np.float32) + 0.5) * b
        ys = (np.arange(gh, dtype=np.float32) + 0.5) * b
        self._grid = np.meshgrid(xs, ys)

    def _init_models(self, block_mean: np.ndarray) -> None:
        shape = (2,) + block_mean.shape
        self.mean = np.repeat(block_mean[None], 2, axis=0).astype(np.float32)
        self.var = np.full(shape, self.init_var, dtype=np.float32)
        self.age = np.zeros(shape, dtype=np.float32)

    # ----------------------------------------------------------- registration

    def _detect(self, gray: np.ndarray) -> None:
        feats = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.corner_quality,
            minDistance=self.corner_min_dist,
        )
        self.prev_feats = feats
        self._since_detect = 0

    def _homography(self, gray: np.ndarray) -> np.ndarray:
        """Homography mapping CURRENT frame coords to PREVIOUS frame coords.

        That direction is what the model warp needs: each current block centre
        asks "where did I come from", and reads the model from there.

        Surviving tracks are carried into the next frame, so a feature is
        detected once and then followed for as long as it lasts. Periodic
        re-detection is still needed: tracks die unevenly, and a herd that has
        thinned into one textured corner of the frame fits a homography that is
        excellent there and wrong everywhere else.
        """
        feats = self.prev_feats
        if self.prev_gray is None or feats is None or len(feats) < 8:
            self.reg_identity += 1
            self._detect(gray)
            return IDENTITY

        cur, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, feats, None, winSize=(15, 15), maxLevel=3
        )
        matrix = None
        if cur is not None and status is not None:
            keep = status.ravel() == 1
            if int(keep.sum()) >= 8:
                src, dst = cur[keep], feats[keep]
                matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
                if matrix is None or inliers is None or int(inliers.sum()) < self.min_inliers:
                    matrix = None
                else:
                    # Carry only the inliers forward: an outlier is either a
                    # bad track or a point on the target, and neither should be
                    # allowed to describe the background next frame.
                    self.prev_feats = src[inliers.ravel().astype(bool)]
                    self._since_detect += 1

        if matrix is None:
            self.reg_failures += 1
            self._detect(gray)
            return IDENTITY

        if (
            self.prev_feats is None
            or len(self.prev_feats) < self.redetect_below
            or self._since_detect >= self.redetect_every
        ):
            self._detect(gray)
        return matrix.astype(np.float32)

    # ------------------------------------------------------ motion compensation

    def _compensate(self, matrix: np.ndarray, block_mean: np.ndarray):
        """Resample every model through ``matrix`` into the current view.

        Each current block centre is back-projected, landing somewhere between
        four previous blocks; the compensated model is their overlap-weighted
        mixture. The variance carries the *mean-spread* correction --
        ``sum w(var + (mixed_mean - mean)^2)`` -- without which the mixture
        claims the confidence of a single model while describing a blend of
        four, and the very next frame reads as foreground.
        """
        b = self.block
        gh, gw = block_mean.shape
        xs, ys = self._grid

        den = matrix[2, 0] * xs + matrix[2, 1] * ys + matrix[2, 2]
        den = np.where(np.abs(den) < 1e-9, np.float32(1e-9), den)
        px = (matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]) / den
        py = (matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]) / den

        # Continuous block-grid coordinates: block i has its centre at i.
        gx = px / b - np.float32(0.5)
        gy = py / b - np.float32(0.5)
        i0 = np.floor(gx).astype(np.int32).ravel()
        j0 = np.floor(gy).astype(np.int32).ravel()
        fx = (gx.ravel() - i0).astype(np.float32)
        fy = (gy.ravel() - j0).astype(np.float32)
        one = np.float32(1.0)

        neighbours = (
            (j0, i0, (one - fx) * (one - fy)),
            (j0, i0 + 1, fx * (one - fy)),
            (j0 + 1, i0, (one - fx) * fy),
            (j0 + 1, i0 + 1, fx * fy),
        )

        # Flat (2, N) views: one gather along a single axis rather than
        # advanced indexing across two.
        flat_mean = self.mean.reshape(2, -1)
        flat_var = self.var.reshape(2, -1)
        flat_age = self.age.reshape(2, -1)
        n = gh * gw

        acc_w = np.zeros(n, dtype=np.float32)
        acc_m = np.zeros((2, n), dtype=np.float32)
        acc_a = np.zeros((2, n), dtype=np.float32)
        top_w = np.zeros(n, dtype=np.float32)
        gathered = []

        for jn, iN, weight in neighbours:
            valid = (iN >= 0) & (iN < gw) & (jn >= 0) & (jn < gh)
            wv = weight * valid
            idx = np.clip(jn, 0, gh - 1) * gw + np.clip(iN, 0, gw - 1)
            means = flat_mean[:, idx]
            acc_m += wv * means
            acc_a += wv * flat_age[:, idx]
            acc_w += wv
            np.maximum(top_w, wv, out=top_w)
            gathered.append((wv, idx, means))

        safe = np.where(acc_w > 1e-6, acc_w, one)
        mean = acc_m / safe
        age = acc_a / safe

        var = np.zeros((2, n), dtype=np.float32)
        for wv, idx, means in gathered:
            var += wv * (flat_var[:, idx] + (mean - means) ** 2)
        var /= safe

        if self.mix_purity > 0.0:
            # A model assembled from four different sources is less reliable
            # than any one of them; purity is 1 for an exact hit, 0.25 for a
            # dead-centre four-way blend.
            age *= (top_w / safe) ** self.mix_purity

        # Blocks whose source lies outside -- or within one block of -- the
        # previous model edge are newly exposed scene. They get a fresh model,
        # which age_min then suppresses until it has earned belief.
        stale = (acc_w <= 1e-6) | (i0 < 1) | (i0 >= gw - 1) | (j0 < 1) | (j0 >= gh - 1)
        mean[:, stale] = block_mean.reshape(-1)[stale]
        var[:, stale] = self.init_var
        age[:, stale] = 0.0

        np.minimum(age, self.max_age, out=age)
        shape = (2, gh, gw)
        return mean.reshape(shape), var.reshape(shape), age.reshape(shape)

    # -------------------------------------------------------------------- run

    def process(self, ctx) -> list[dict]:
        if ctx.rgba is None:
            return []

        b = self.block
        frame = np.asarray(ctx.rgba)
        h = (min(frame.shape[0], ctx.height) // b) * b
        w = (min(frame.shape[1], ctx.width) // b) * b
        if h < b or w < b:
            return []

        gray8 = cv2.cvtColor(np.ascontiguousarray(frame[:h, :w]), cv2.COLOR_RGBA2GRAY)
        gray = gray8.astype(np.float32)
        gh, gw = h // b, w // b
        # INTER_AREA over an exact integer factor is the block mean, and is
        # several times quicker than a reshape-and-reduce over strided axes.
        block_mean = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)

        if self.mean is None or self.mean.shape[1:] != (gh, gw):
            self._init_grid(gh, gw)
            self._init_models(block_mean)
            self.prev_gray = gray8
            self._detect(gray8)
            return []

        matrix = self._homography(gray8)
        mean, var, age = self._compensate(matrix, block_mean)

        # --- dual mode: whichever model has outlived the other is the apparent
        # one. The displaced model is discarded and restarted on the current
        # observation, exactly as the reference does.
        swap = age[1] > age[0]
        mean = np.stack(
            [np.where(swap, mean[1], mean[0]), np.where(swap, block_mean, mean[1])]
        ).astype(np.float32)
        var = np.stack(
            [np.where(swap, var[1], var[0]), np.where(swap, self.init_var, var[1])]
        ).astype(np.float32)
        age = np.stack(
            [np.where(swap, age[1], age[0]), np.where(swap, 0.0, age[1])]
        ).astype(np.float32)

        # --- which model does the observation belong to
        match_a = (block_mean - mean[0]) ** 2 < self.theta_match * var[0]
        match_c = (block_mean - mean[1]) ** 2 < self.theta_match * var[1]
        selected = np.where(match_a, 0, 1).astype(np.int8)
        # Matching neither means the block is unlike anything we have on file:
        # restart the candidate's clock so it has to earn its way to apparent.
        age[1][~match_a & ~match_c] = 0.0

        chosen = np.arange(2, dtype=np.int8)[:, None, None] == selected

        # --- running-mean update with 1/(age+1) gain, selected model only
        gain = age / (age + 1.0)
        gain[age < 1.0] = 0.0
        gain[~chosen] = 1.0
        new_mean = (mean * gain + block_mean * (1.0 - gain)).astype(np.float32)

        # --- foreground, decided per pixel against the block's apparent model.
        # Deviation from the reference, which compares against the *previous*
        # frame's variance and age -- i.e. values sampled at the pre-warp
        # location. With a moving camera those belong to a different piece of
        # scene, so the compensated ones are used here instead.
        # Age and variance fold into one threshold image so the upsample is
        # done once: a model too young to trust gets an unreachable threshold
        # rather than a separate mask.
        limit = np.where(age[0] > self.age_min, self.theta_fg * var[0], np.inf)
        big_mean = _upsample(new_mean[0], w, h)
        big_limit = _upsample(limit, w, h)
        dist = (gray - big_mean) ** 2
        mask = (dist > big_limit).astype(np.uint8)

        # --- variance update. The reference takes the block *maximum* squared
        # deviation, not the mean: a block containing one bright edge should
        # not report itself as quiet.
        sel_mean = _upsample(np.where(selected == 0, new_mean[0], new_mean[1]), w, h)
        spread = _block_max((gray - sel_mean) ** 2, b)
        new_var = (var * gain + (1.0 - gain) * spread).astype(np.float32)
        fresh = chosen & (age == 0.0)
        new_var = np.where(fresh & (new_var < self.init_var), self.init_var, new_var)
        np.maximum(new_var, self.min_var, out=new_var)

        new_age = age.copy()
        new_age[chosen] += 1.0
        np.minimum(new_age, self.max_age, out=new_age)

        self.mean, self.var, self.age = new_mean, new_var, new_age
        self.prev_gray = gray8

        return self._boxes(mask, ctx)

    # ------------------------------------------------------------ postprocess

    def _boxes(self, mask: np.ndarray, ctx) -> list[dict]:
        if self.dilate > 0:
            kernel = np.ones((self.dilate, self.dilate), np.uint8)
            mask = cv2.dilate(mask, kernel)

        count, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        raw = []
        for k in range(1, count):
            x, y, bw, bh, area = stats[k]
            if area < self.min_area:
                continue
            raw.append([int(x), int(y), int(x + bw), int(y + bh), int(area)])

        merged = [m for m in _merge(raw, self.merge_gap) if m[4] >= self.min_box_area]

        sx = ctx.src_w / float(ctx.width)
        sy = ctx.src_h / float(ctx.height)
        return [
            {
                "left": float(x0 * sx),
                "top": float(y0 * sy),
                "width": float((x1 - x0) * sx),
                "height": float((y1 - y0) * sy),
                "cells": int(area),
            }
            for x0, y0, x1, y1, area in merged
        ]


def _upsample(grid: np.ndarray, w: int, h: int) -> np.ndarray:
    """Block-replicate a model grid up to frame resolution."""
    return cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)


def _block_max(image: np.ndarray, block: int) -> np.ndarray:
    """Maximum over each ``block`` x ``block`` tile.

    A dilation anchored at the tile's own corner is the same max, computed by
    OpenCV's separable morphology rather than by a strided numpy reduction over
    half a million elements.
    """
    spread = cv2.dilate(image, np.ones((block, block), np.uint8), anchor=(0, 0))
    return spread[::block, ::block]


def _merge(boxes: list[list[int]], gap: int) -> list[list[int]]:
    """Union boxes that come within ``gap`` px of each other.

    A walking person breaks into several components -- the legs move fastest,
    the torso barely at all -- and they should be reported as one target.
    """
    if gap <= 0 or len(boxes) < 2:
        return boxes

    changed = True
    while changed:
        changed = False
        out: list[list[int]] = []
        for box in boxes:
            for other in out:
                if (
                    box[0] - gap <= other[2]
                    and other[0] - gap <= box[2]
                    and box[1] - gap <= other[3]
                    and other[1] - gap <= box[3]
                ):
                    other[0] = min(other[0], box[0])
                    other[1] = min(other[1], box[1])
                    other[2] = max(other[2], box[2])
                    other[3] = max(other[3], box[3])
                    other[4] += box[4]
                    changed = True
                    break
            else:
                out.append(list(box))
        boxes = out
    return boxes
