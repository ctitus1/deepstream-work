"""Homography registration + gradient-normalised three-frame differencing.

The premise is a claim about what the baseline's false positives actually are.
They are diffuse structures hugging high-contrast edges in a scene where
nothing is moving, and that is the signature of sub-pixel *misregistration*,
not of sensor noise. If the background model leaves a residual alignment error
of ``eps`` pixels, the apparent intensity change is

    dI(x) ~= grad I(x) . eps

-- proportional to the local image gradient. A high-contrast edge therefore
produces a large residual for arbitrarily small ``eps``, so any detector that
thresholds a raw frame difference lights up on edges first, and the threshold
that finally silences the edges is far above what a real target needs.

So divide by the gradient::

    S(x) = D(x) / (|grad I_N(x)| + |grad I_warp(x)| + c)

The edge term is now bounded by ``eps`` itself -- a constant, independent of
contrast -- while a genuinely displaced object produces a difference set by its
contrast against the background it uncovered, which has nothing to do with the
static scene's local gradient there. A contrast-dependent false-positive rate
becomes a uniform one, thresholdable with a single global number. It is also
intrinsically safe in textureless regions: where ``|grad I| ~= 0`` there is no
signal and no noise, and ``c`` alone sets the scale.

Mechanically:

  1. ``goodFeaturesToTrack`` + pyramidal Lucas-Kanade, run forward and backward,
     keeping only points whose round trip closes to under half a pixel. Points
     are tracked *continuously* and their positions kept in a ring, so the
     correspondence between frame N and frame N-L is read straight out of the
     trace rather than composed from L pairwise homographies. Composition would
     accumulate exactly the sub-pixel drift this approach exists to avoid.
  2. ``findHomography`` with MAGSAC at a 1.5 px reprojection threshold. The
     threshold is not a free parameter: at 960x540 the walker moves ~2.5 px per
     frame, so a threshold much above 2 px admits the target as a background
     inlier and the homography partially explains it away.
  3. Warp, difference, normalise by the gradient, threshold ``S`` rather than
     ``D``.
  4. Two lags, not one. A two-frame difference fires at both the old and the new
     position of the object. Differencing frame N against N-k and against N-2k
     and intersecting the two leaves only the position at N, because that is the
     only place both fire. This is the three-frame trick rearranged to be
     causal: the textbook form brackets the current frame and needs a frame of
     look-ahead, and a lagged box is a box on the wrong frame as far as the
     scorer is concerned.
  5. Morphological clean, connected components, boxes in source pixels.

``k`` is the parameter that matters most. A person moving 2.5 px/frame against
a body 25-40 px wide uncovers almost no background at k=1; the difference is a
thin rim and the detector is being asked to find a target by its outline. At
k=4 the displacement is 10 px and a real fraction of the body sits over ground
it did not cover before.

Known limit: a homography is exact only for a planar scene, so tall structures
at low altitude keep a residual parallax that gradient normalisation does not
remove -- it is real image motion, not misregistration. ``debug`` reports the
median residual by image region so that can be seen rather than guessed at.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

NAME = "gradient-diff"

cv2.setNumThreads(2)


class Approach:
    needs_flow = False
    needs_pixels = True

    def __init__(self, cfg: dict):
        cfg = cfg or {}

        # Lag, in frames, between the reference and the current frame. The
        # second reference sits at 2k so that the intersection of the two
        # differences isolates the current position.
        self.k = int(cfg.get("k", 3))
        self.single = bool(cfg.get("single", False))  # skip the 2k lag, for A/B

        # Feature tracking.
        self.max_corners = int(cfg.get("max_corners", 800))
        self.quality = float(cfg.get("quality", 0.01))
        self.min_distance = int(cfg.get("min_distance", 8))
        self.block_size = int(cfg.get("block_size", 7))
        self.fb_tolerance = float(cfg.get("fb_tolerance", 0.5))
        self.redetect_below = int(cfg.get("redetect_below", 500))
        self.lk_win = int(cfg.get("lk_win", 21))
        self.lk_levels = int(cfg.get("lk_levels", 3))

        # Homography.
        self.ransac_px = float(cfg.get("ransac_px", 1.5))
        self.min_inliers = int(cfg.get("min_inliers", 60))

        # Normalised significance.
        self.blur = int(cfg.get("blur", 3))  # pre-difference smoothing, 0 = off
        self.c = float(cfg.get("c", 4.0))  # sensor noise floor, intensity units
        self.threshold = float(cfg.get("threshold", 0.8))

        # Post-processing.
        self.open_px = int(cfg.get("open_px", 3))
        self.close_px = int(cfg.get("close_px", 7))
        self.min_area = int(cfg.get("min_area", 90))
        self.border = int(cfg.get("border", 0))  # 0 = derive from the homography
        self.pad = float(cfg.get("pad", 0.0))  # box padding, fraction of size

        self.debug = bool(cfg.get("debug", False))

        lags = (self.k,) if self.single else (self.k, 2 * self.k)
        self.lags = lags
        self.max_lag = max(lags)

        # Ring of past frames as separate contiguous (luma, gradient magnitude)
        # planes. Interleaving them into one 2-channel image halves the warp
        # calls but leaves every downstream operand a strided view, and the
        # copies that forces cost several times what the extra warp saves.
        self.ring: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.max_lag + 1)
        # Point traces, aligned index-for-index across the ring.
        self.trace: deque[np.ndarray] = deque(maxlen=self.max_lag + 1)
        self.age: np.ndarray = np.zeros(0, np.int32)
        self.prev_gray: np.ndarray | None = None
        self.frames_since_detect = 0
        self.stats: list[tuple] = []

        # A one-frame step displaces a feature by ~2-3 px here, so three
        # pyramid levels is already generous; the fourth costs time and buys
        # nothing. The window stays wide, because it is the window that sets
        # how well the sub-pixel term is conditioned.
        self._lk = dict(
            winSize=(self.lk_win, self.lk_win),
            maxLevel=self.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

    # ---------------------------------------------------------------- helpers

    def _detect(self, gray: np.ndarray, existing: np.ndarray | None) -> np.ndarray:
        """New corners, kept away from the ones already being tracked."""
        want = self.max_corners - (0 if existing is None else len(existing))
        if want <= 0:
            return np.zeros((0, 2), np.float32)
        mask = None
        if existing is not None and len(existing):
            # Stamp the occupied pixels and grow them, rather than drawing a
            # circle per point: one dilation beats a thousand draw calls.
            mask = np.zeros(gray.shape, np.uint8)
            xy = np.round(existing).astype(np.int32)
            np.clip(xy[:, 0], 0, gray.shape[1] - 1, out=xy[:, 0])
            np.clip(xy[:, 1], 0, gray.shape[0] - 1, out=xy[:, 1])
            mask[xy[:, 1], xy[:, 0]] = 255
            r = self.min_distance
            mask = cv2.bitwise_not(cv2.dilate(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8)))
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=want,
            qualityLevel=self.quality,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=mask,
        )
        if pts is None:
            return np.zeros((0, 2), np.float32)
        return pts.reshape(-1, 2).astype(np.float32)

    def _track(self, prev: np.ndarray, cur: np.ndarray, pts: np.ndarray):
        """Forward-backward LK. Returns the tracked points and a keep mask."""
        if not len(pts):
            return pts, np.zeros(0, bool)
        fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts.reshape(-1, 1, 2), None, **self._lk)
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(cur, prev, fwd, None, **self._lk)
        fwd = fwd.reshape(-1, 2)
        back = back.reshape(-1, 2)
        fb = np.linalg.norm(pts - back, axis=1)
        keep = (
            (st_f.ravel() == 1)
            & (st_b.ravel() == 1)
            & (fb < self.fb_tolerance)
            & np.isfinite(fwd).all(axis=1)
        )
        return fwd, keep

    def _prune(self, keep: np.ndarray) -> None:
        for i in range(len(self.trace)):
            self.trace[i] = self.trace[i][keep]
        self.age = self.age[keep]

    def _append_points(self, new: np.ndarray) -> None:
        """Add freshly detected points. They have no history, so age 0 keeps
        them out of every long-lag homography until they have earned one."""
        if not len(new):
            return
        for i in range(len(self.trace)):
            self.trace[i] = np.vstack([self.trace[i], new])
        self.age = np.concatenate([self.age, np.zeros(len(new), np.int32)])

    def _valid_mask(self, h: int, w: int, homographies: dict) -> np.ndarray:
        """Where every warp had real source pixels to draw from, shrunk by the
        largest global displacement. The image edge is uncovered as the camera
        pans and will always difference badly."""
        mask = np.full((h, w), 255, np.uint8)
        margin = self.border
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
        for hom in homographies.values():
            moved = cv2.perspectiveTransform(corners, hom).reshape(-1, 2)
            quad = np.round(moved).astype(np.int32)
            layer = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(layer, quad, 255)
            mask = cv2.bitwise_and(mask, layer)
            if self.border == 0:
                shift = np.linalg.norm(moved - corners.reshape(-1, 2), axis=1).max()
                margin = max(margin, int(np.ceil(shift)) + 4)
        margin = min(margin, min(h, w) // 4)
        if margin > 0:
            mask = cv2.erode(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
        return mask

    # ------------------------------------------------------------------- main

    def process(self, ctx) -> list[dict]:
        if ctx.rgba is None:
            return []

        gray = cv2.cvtColor(ctx.rgba, cv2.COLOR_RGBA2GRAY)
        if self.blur >= 3:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        grayf = gray.astype(np.float32)

        # Sobel scaled by 1/4 so the magnitude is a per-pixel intensity slope,
        # which makes S read directly as "equivalent registration error in px".
        gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
        gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
        gmag = cv2.magnitude(gx, gy)

        h, w = gray.shape[:2]

        # --- track ---------------------------------------------------------
        if self.prev_gray is None:
            pts = self._detect(gray, None)
            self.trace.append(pts)
            self.age = np.zeros(len(pts), np.int32)
            self.prev_gray = gray
            self.ring.append((grayf, gmag))
            return []

        prev_pts = self.trace[-1]
        moved, keep = self._track(self.prev_gray, gray, prev_pts)
        self._prune(keep)
        self.age += 1
        self.trace.append(moved[keep])

        self.frames_since_detect += 1
        if len(self.trace[-1]) < self.redetect_below or self.frames_since_detect >= 30:
            self._append_points(self._detect(gray, self.trace[-1]))
            self.frames_since_detect = 0

        self.prev_gray = gray
        self.ring.append((grayf, gmag))

        # --- homography per lag ---------------------------------------------
        boxes: list[dict] = []
        if len(self.ring) <= self.max_lag:
            return boxes

        cur_pts = self.trace[-1]
        homographies: dict[int, np.ndarray] = {}
        for lag in self.lags:
            sel = self.age >= lag
            if int(sel.sum()) < self.min_inliers:
                return boxes
            src = self.trace[-1 - lag][sel]
            dst = cur_pts[sel]
            hom, inliers = cv2.findHomography(
                src, dst, cv2.USAC_MAGSAC, ransacReprojThreshold=self.ransac_px
            )
            if hom is None or inliers is None or int(inliers.sum()) < self.min_inliers:
                return boxes
            homographies[lag] = hom.astype(np.float32)

        # --- warp, difference, normalise -------------------------------------
        accept: np.ndarray | None = None
        residuals: list[np.ndarray] = []
        for lag, hom in homographies.items():
            ref_gray, ref_gmag = self.ring[-1 - lag]
            warp_gray = cv2.warpPerspective(ref_gray, hom, (w, h), flags=cv2.INTER_LINEAR)
            warp_gmag = cv2.warpPerspective(ref_gmag, hom, (w, h), flags=cv2.INTER_LINEAR)
            diff = cv2.absdiff(grayf, warp_gray)
            denom = cv2.add(cv2.add(gmag, warp_gmag), self.c)
            sig = cv2.divide(diff, denom)
            hit = (sig > self.threshold).astype(np.uint8)
            accept = hit if accept is None else cv2.bitwise_and(accept, hit)
            if self.debug:
                residuals.append(diff)

        if accept is None:
            return boxes

        accept &= self._valid_mask(h, w, homographies) > 0

        # --- clean and label --------------------------------------------------
        if self.open_px >= 3:
            accept = cv2.morphologyEx(
                accept, cv2.MORPH_OPEN, np.ones((self.open_px, self.open_px), np.uint8)
            )
        if self.close_px >= 3:
            accept = cv2.morphologyEx(
                accept, cv2.MORPH_CLOSE, np.ones((self.close_px, self.close_px), np.uint8)
            )

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(accept, 8)
        sx = ctx.src_w / float(ctx.width)
        sy = ctx.src_h / float(ctx.height)
        for i in range(1, count):
            x, y, bw, bh, area = stats[i]
            if area < self.min_area:
                continue
            if self.pad > 0:
                dx, dy = bw * self.pad, bh * self.pad
                x, y, bw, bh = x - dx, y - dy, bw + 2 * dx, bh + 2 * dy
            boxes.append(
                {
                    "left": float(x * sx),
                    "top": float(y * sy),
                    "width": float(bw * sx),
                    "height": float(bh * sy),
                    "area": int(area),
                    "cx": float(centroids[i][0] * sx),
                    "cy": float(centroids[i][1] * sy),
                }
            )

        if self.debug and residuals:
            self._log(ctx, residuals[0], accept)
        return boxes

    def _log(self, ctx, residual: np.ndarray, accept: np.ndarray) -> None:
        """Median post-alignment residual by image region, plus where the
        normalised score actually sits. Parallax from tall structures shows up
        as one region staying stubbornly high while the rest settle."""
        if ctx.frame_index % 60:
            return
        h, w = residual.shape
        cells = [
            float(np.median(residual[r * h // 3 : (r + 1) * h // 3, c * w // 3 : (c + 1) * w // 3]))
            for r in range(3)
            for c in range(3)
        ]
        print(
            f"[gradient-diff] f{ctx.frame_index} tracks={len(self.age)} "
            f"on={int(accept.sum())} residual_by_region="
            + " ".join(f"{v:.2f}" for v in cells),
            flush=True,
        )
