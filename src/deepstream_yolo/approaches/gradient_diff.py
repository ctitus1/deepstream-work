"""Homography registration + three-frame differencing, with the gradient
normalisation it was named for measured and switched off.

The premise. The baseline's remaining false positives are diffuse structures
hugging high-contrast edges in a scene where nothing is moving, and that is the
signature of sub-pixel *misregistration* rather than noise. If the background
model leaves a residual alignment error of ``eps`` pixels, the apparent
intensity change is

    dI(x) ~= grad I(x) . eps

-- proportional to the local image gradient, so a sharp edge produces a large
residual for arbitrarily small ``eps`` and any detector thresholding a raw
difference lights up on edges first. The fix follows immediately: divide by the
gradient,

    S(x) = D(x) / (|grad I_N(x)| + |grad I_warp(x)| + c)

which bounds the edge term by ``eps`` itself -- a constant, independent of
contrast -- turning a contrast-dependent false-positive rate into a uniform one
that a single global number can threshold.

The premise is sound and it is also, on this footage, wrong. Holding the
flat-region behaviour fixed at an effective cut of 36 intensity units and
varying only how strongly the gradient participates in the denominator, over
the whole 5361-frame video:

    c=22.5 t=1.6   strong      F1 0.764   prec 0.899   recall 0.664
    c=45   t=0.8   moderate    F1 0.902   prec 0.885   recall 0.920
    c=80   t=0.45  weak        F1 0.915   prec 0.904   recall 0.925
    none   |D|>36  off         F1 0.925   prec 0.920   recall 0.930

Monotone, and the direction is the interesting part: normalisation costs almost
nothing in precision and takes recall apart. That is not what a failed
false-positive filter looks like, it is what a signal attenuator looks like,
and the reason is in the premise's one unstated assumption. The derivation
treats ``grad I`` as a property of the static background, so that dividing by
it removes a background artefact. But a person is a high-contrast object, and
``grad I_N`` is largest precisely where the person now stands. The denominator
is therefore biggest exactly where the numerator is signal, and the operation
divides the target away along with the edges.

The other half of the answer is that the artefact being corrected for is not
there to correct. Sub-pixel misregistration is what happens when alignment is
mediocre; forward-backward-filtered KLT into a MAGSAC homography aligns this
scene well enough that the edge residual never dominates, so normalisation was
paying full price for a problem this registration does not have.

So the pipeline is kept and the division is not. ``normalise: true`` restores
it, because the argument for it is a good one and a scene with weaker
registration or lower-contrast targets could well invert this result.

What is actually doing the work, in order of how much:

  1. **Registration.** ``goodFeaturesToTrack`` + pyramidal Lucas-Kanade run
     forward and backward, keeping only points whose round trip closes to under
     half a pixel, into ``findHomography`` with MAGSAC at a 1.5 px reprojection
     threshold. That threshold is not free: the walker moves ~3 px/frame
     relative to the background at 960x540, so a threshold much above 2 px
     admits the target as a background inlier and the homography explains part
     of it away. Points are tracked *continuously* and their positions kept in
     a ring, so the correspondence between frame N and N-L is read straight out
     of the trace rather than composed from L pairwise homographies --
     composition accumulates exactly the drift that would put the edge artefact
     back.
  2. **Two lags, not one, and causal.** A two-frame difference fires at both
     the old and the new position of the object. Differencing frame N against
     N-k and against N-2k and intersecting leaves the position the two agree
     on, which is the current one. The textbook three-frame form brackets the
     current frame and needs look-ahead; a lagged box is a box on the wrong
     frame as far as the scorer is concerned, so it is rearranged backwards.
  3. **Lag and area floor together.** Recall is 0.508 at k=1 -- a body 63 px
     wide moving 3 px/frame offers a three-pixel rim to difference -- and
     saturates at 0.957 from k=2 on. On recall alone k=2 is the answer. But lag
     also sets blob size, and blob area is the only feature separating a true
     box from a false one; their widths, heights, aspects and fill ratios all
     overlap. A longer lag grows true blobs faster than false ones, so the area
     floor cuts deeper, and at each lag's preferred floor k=3 wins. Choosing
     either parameter alone points somewhere misleading.
  4. **Closing before labelling.** The intersection leaves a target as leading
     edge, trailing edge and whatever body texture differed between -- three or
     four fragments, individually under the area floor, collectively three
     false positives instead of one hit. An 11 px closing makes them one
     component.

Where it fails, measured on the final run. 243 of 3472 mover frames are missed,
in 77 streaks of which 27 are a single frame and the longest is 21. They are
not scattered: on missed frames the mover is moving at a median 1.68 branch
px/frame against 3.82 on frames that hit, and 35% of them are under 1 px/frame
against 3% of hits. The misses are the walker slowing down or standing still,
which is the honest floor of any differencing method -- a person who stops
moving stops producing a difference, and no threshold recovers them. The one
long streak is at frames 38-58, before the tracker has accumulated enough
history to hold a homography.

Of the 350 false positives, 165 land on the three stationary people, who sway
and shift weight; the same floor applies from the other side, since a
differencing method cannot tell a small real movement from the beginning of a
large one. The remaining 185 are spread thinly over the frame with a mild
concentration bottom-right.

Known limit not exercised by this footage: a homography is exact only for a
planar scene, so tall structures at low altitude keep a residual parallax that
no threshold distinguishes from a target, because it is real image motion.
``debug`` reports the median residual by image region so that can be seen
rather than guessed at; here it runs 2.0-6.3 intensity units with no region
diverging, which is why parallax is not among the failures above.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

NAME = "gradient-diff"

# Everything here was tuned on 30 fps footage; durations quoted in seconds are
# converted against the source rate.
REFERENCE_FPS = 30.0

# cv2.setNumThreads() is NOT called here, deliberately. It is process-global, so
# a module setting it at import throttles every other approach in a batched run
# and the video renderer besides -- this file capping OpenCV at 2 threads was
# measured inflating klt-homography from 8.3 to 21.0 ms/frame in a batch of
# four, on a 16-core machine. Thread policy belongs to whoever owns the process;
# eval/run_motion.py sets it once, visibly.


class Approach:
    needs_flow = False
    needs_pixels = True

    def __init__(self, cfg: dict):
        cfg = cfg or {}

        # Lag, in frames, between the reference and the current frame. The
        # second reference sits at 2k so that the intersection of the two
        # differences isolates the current position.
        # Differencing baseline in SECONDS. Was 3 frames, tuned at 30 fps; a
        # frame count silently halves the time window at 60 fps.
        self.k_s = float(cfg.get("k_s", 0.1))
        # Explicit frame override, else the reference-rate equivalent of k_s.
        # It has to be valid here, not just by the first frame: the ring buffers
        # below are sized from it.
        self.k_fixed = int(cfg.get("k", 0)) or 0
        self.k = self.k_fixed or max(1, int(round(self.k_s * REFERENCE_FPS)))
        self._timed = False
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

        # Deciding what counts as a difference.
        self.blur = int(cfg.get("blur", 3))  # pre-difference smoothing, 0 = off
        # Gradient normalisation, off by default because it was measured to
        # cost F1 rather than earn it -- see the module docstring. ``c`` is the
        # denominator floor and ``threshold`` the cut on S = D / (2|grad I|+c);
        # ``level`` is the cut on the raw difference when normalisation is off.
        self.normalise = bool(cfg.get("normalise", False))
        self.c = float(cfg.get("c", 45.0))
        self.threshold = float(cfg.get("threshold", 0.8))
        self.level = float(cfg.get("level", 36.0))

        # Post-processing.
        self.open_px = int(cfg.get("open_px", 3))
        self.close_px = int(cfg.get("close_px", 11))
        self.min_area = int(cfg.get("min_area", 250))
        self.border = int(cfg.get("border", 0))  # 0 = derive from the homography
        self.pad = float(cfg.get("pad", 0.0))  # box padding, fraction of size

        self.debug = bool(cfg.get("debug", False))

        self._size_ring()

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

    def _size_ring(self) -> None:
        """Derive the lag set and ring depth from the current k."""
        self.lags = (self.k,) if self.single else (self.k, 2 * self.k)
        self.max_lag = max(self.lags)

    def _calibrate_time(self, ctx) -> None:
        """Convert the differencing baseline from seconds to frames.

        Re-sizes the ring, because its depth is a function of k and k is only
        known once the frame rate is.
        """
        self._timed = True
        if self.k_fixed:
            return
        fps = float(getattr(ctx, "fps", 0.0) or REFERENCE_FPS)
        frames = max(1, int(round(self.k_s * fps)))
        if frames == self.k:
            return
        self.k = frames
        self._size_ring()
        self.ring = deque(self.ring, maxlen=self.max_lag + 1)
        self.trace = deque(self.trace, maxlen=self.max_lag + 1)

    def process(self, ctx) -> list[dict]:
        if not self._timed:
            self._calibrate_time(ctx)

        if ctx.rgba is None:
            return []

        gray = cv2.cvtColor(ctx.rgba, cv2.COLOR_RGBA2GRAY)
        if self.blur >= 3:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        grayf = gray.astype(np.float32)

        # Sobel scaled by 1/4 so the magnitude is a per-pixel intensity slope,
        # which makes S read directly as "equivalent registration error in px".
        if self.normalise:
            gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
            gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
            gmag = cv2.magnitude(gx, gy)
        else:
            gmag = None

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
            diff = cv2.absdiff(grayf, warp_gray)
            if self.normalise:
                warp_gmag = cv2.warpPerspective(ref_gmag, hom, (w, h), flags=cv2.INTER_LINEAR)
                sig = cv2.divide(diff, cv2.add(cv2.add(gmag, warp_gmag), self.c))
                hit = (sig > self.threshold).astype(np.uint8)
            else:
                hit = (diff > self.level).astype(np.uint8)
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
