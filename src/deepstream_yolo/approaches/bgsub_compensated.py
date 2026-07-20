"""Background subtraction against a background model warped to track the camera.

Frame differencing asks "what changed since last frame", which on compressed
aerial video answers mostly "the codec". This asks a different question: what
does this pixel *usually* look like, and is it that now. A per-pixel appearance
model averages the codec noise away instead of measuring it, and a textureless
patch simply keeps matching its own model rather than producing the garbage
dense flow invents where there is nothing to track.

The camera moves, so a background model in fixed image coordinates is worthless
after a second. Every frame the model is therefore re-registered to the new
view before it is used:

  1. Track sparse corners from the previous frame with LK, fit a homography
     with RANSAC. Inlier count is the health signal for everything below.
  2. ``warpPerspective`` the whole model -- mean, variance and a per-pixel age
     -- through that homography, so it arrives already aligned with the frame
     about to be tested. Warping the *model* to the frame rather than the frame
     to a keyframe is what keeps registration error from compounding: the model
     has a finite time constant, so anything the warp smears out decays away
     within a few dozen frames instead of accumulating over the sequence.
  3. Foreground is a per-pixel z-test against that model. The variance channel
     is what makes this survive imperfect registration -- an edge that jitters
     under sub-pixel misalignment learns a wide variance and stops firing,
     while a flat region keeps a tight one and stays sensitive.
  4. Morphological open/close to kill speckle, connected components for boxes.

``age`` exists because a warp exposes territory the model has never seen along
whichever border the camera advanced towards. Those pixels are marked unknown
and are not allowed to produce detections until they have been observed long
enough to mean something; without it the leading edge of the frame is a
permanent wall of false positives.

MOG2 and KNN are available via ``model=mog2|knn``. They cannot be warped -- the
mixture state is private -- so they have to run the other way round, on frames
warped into a keyframe's coordinates, which compounds registration error over
the whole sequence rather than over one frame. That is not a small difference:
tuned as well as either could be, MOG2 reaches F1 0.574 and KNN 0.476 where the
warped Gaussian reaches 0.920. They are kept because that comparison is the
evidence for warping the model rather than the frames; see ``_mixture``.

Measured over the whole clip, against the baseline in the same columns:

    approach              F1   prec  recall  box/f  onstat  onbg    ms
    bgsub-compensated  0.920  0.944   0.897   0.64      40   154   9.3
    baseline           0.422  0.713   0.300   0.27      18   401   8.3
"""

from __future__ import annotations

import cv2
import numpy as np

NAME = "bgsub-compensated"


class Approach:
    needs_flow = False
    needs_pixels = True

    def __init__(self, cfg: dict):
        cfg = cfg or {}
        self.model_kind = str(cfg.get("model", "gauss"))
        self.scale = float(cfg.get("scale", 0.75))

        # Registration
        self.max_corners = int(cfg.get("max_corners", 600))
        self.quality = float(cfg.get("quality", 0.01))
        self.min_distance = int(cfg.get("min_distance", 8))
        self.min_inliers = int(cfg.get("min_inliers", 25))

        # Gaussian model
        # Per-frame retention of the age variable (age += alpha*(1-age)), so
        # its time constant is 1/alpha frames. Quoted at 30 fps and rescaled to
        # keep the same constant in seconds at any rate.
        self.alpha = float(cfg.get("alpha", 0.005))
        self.k = float(cfg.get("k", 4.5))
        self.min_diff = float(cfg.get("min_diff", 12.0))
        self.init_var = float(cfg.get("init_var", 400.0))
        self.min_var = float(cfg.get("min_var", 100.0))
        self.max_var = float(cfg.get("max_var", 4000.0))
        self.age_min = float(cfg.get("age_min", 0.25))

        # MOG2 / KNN
        # Model memory in SECONDS. Was 300 frames, tuned at 30 fps.
        self.history_s = float(cfg.get("history_s", 10.0))
        self.history = int(cfg.get("history", 0)) or 0  # explicit frame override
        self._timed = False
        self.var_threshold = float(cfg.get("var_threshold", 24.0))
        self.detect_shadows = bool(cfg.get("detect_shadows", False))
        self.learning_rate = float(cfg.get("learning_rate", -1.0))
        self.reanchor_px = float(cfg.get("reanchor_px", 120.0))

        # Blobs
        self.open_k = int(cfg.get("open_k", 3))
        self.close_k = int(cfg.get("close_k", 17))
        self.min_area = int(cfg.get("min_area", 70))
        self.max_area_frac = float(cfg.get("max_area_frac", 0.08))
        self.panic_frac = float(cfg.get("panic_frac", 0.25))

        self.blur = int(cfg.get("blur", 5))

        self.prev_gray: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.age: np.ndarray | None = None
        self.sub = None
        self.anchor: np.ndarray | None = None
        self.shape: tuple[int, int] | None = None

    # ---------------------------------------------------------------- input

    def _gray(self, rgba: np.ndarray) -> np.ndarray:
        gray = rgba if rgba.ndim == 2 else cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.blur >= 3:
            gray = cv2.GaussianBlur(gray, (self.blur | 1, self.blur | 1), 0)
        return gray

    # --------------------------------------------------------- registration

    def _homography(self, prev: np.ndarray, cur: np.ndarray):
        """Homography mapping previous-frame coordinates onto the current frame."""
        pts = cv2.goodFeaturesToTrack(
            prev,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_distance,
            blockSize=7,
        )
        if pts is None or len(pts) < 12:
            return None, 0
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, cur, pts, None, winSize=(21, 21), maxLevel=3
        )
        if nxt is None or status is None:
            return None, 0
        keep = status.reshape(-1) == 1
        src, dst = pts.reshape(-1, 2)[keep], nxt.reshape(-1, 2)[keep]
        if len(src) < 12:
            return None, 0
        matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if matrix is None:
            return None, 0
        return matrix.astype(np.float64), int(inlier_mask.sum()) if inlier_mask is not None else 0

    # ----------------------------------------------------------- the models

    def _init_gauss(self, gray: np.ndarray) -> None:
        self.mean = gray.astype(np.float32)
        self.var = np.full(gray.shape, self.init_var, np.float32)
        self.age = np.zeros(gray.shape, np.float32)

    def _gauss(self, gray: np.ndarray, matrix) -> np.ndarray | None:
        if self.mean is None:
            self._init_gauss(gray)
            return None

        height, width = gray.shape
        if matrix is not None:
            flags = cv2.INTER_LINEAR
            self.mean = cv2.warpPerspective(self.mean, matrix, (width, height), flags=flags)
            self.var = cv2.warpPerspective(self.var, matrix, (width, height), flags=flags)
            self.age = cv2.warpPerspective(self.age, matrix, (width, height), flags=flags)

        fresh = self.age < 1e-3
        if fresh.any():
            # Territory the camera just advanced into: seed it from the frame so
            # it converges quickly, and mark it unusable until it has aged in.
            frame32 = gray.astype(np.float32)
            self.mean = np.where(fresh, frame32, self.mean)
            self.var = np.where(fresh, np.float32(self.init_var), self.var)

        frame32 = gray.astype(np.float32)
        diff = np.abs(frame32 - self.mean)
        thresh = np.maximum(self.k * np.sqrt(self.var), self.min_diff)
        fg = (diff > thresh) & (self.age >= self.age_min)

        # Every pixel updates at the same rate, foreground included. Holding the
        # model back where it fired is the textbook move and it measures worse
        # here -- see the commit that removed it.
        rate = np.float32(self.alpha)
        self.mean += rate * (frame32 - self.mean)
        self.var += rate * (diff * diff - self.var)
        np.clip(self.var, self.min_var, self.max_var, out=self.var)
        self.age += self.alpha * (1.0 - self.age)

        return fg.astype(np.uint8) * 255

    def _reanchor(self) -> None:
        if self.model_kind == "knn":
            self.sub = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.var_threshold * 16.0,
                detectShadows=self.detect_shadows,
            )
        else:
            self.sub = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )
        self.anchor = np.eye(3, dtype=np.float64)

    @staticmethod
    def _drift(anchor, shape) -> float:
        """How far the view has travelled from its keyframe, in pixels.

        The translation column alone is not that distance -- a pan that rotates
        or scales moves the image without moving the origin much -- so measure
        the worst corner displacement instead.
        """
        if anchor is None:
            return 0.0
        rows, cols = shape
        corners = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(corners, anchor).reshape(-1, 2)
        return float(np.max(np.linalg.norm(moved - corners.reshape(-1, 2), axis=1)))

    def _mixture(self, gray: np.ndarray, matrix) -> np.ndarray | None:
        """MOG2/KNN on frames warped into a keyframe's coordinates.

        The mixture state cannot be warped, so the frame is moved to the model
        instead of the model to the frame. That is only stable while the view
        overlaps the keyframe, hence the re-anchor -- and every re-anchor throws
        the learned model away.
        """
        height, width = gray.shape
        if self.sub is None:
            self._reanchor()

        if matrix is not None and self.anchor is not None:
            try:
                self.anchor = self.anchor @ np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                self._reanchor()

        if self._drift(self.anchor, gray.shape) > self.reanchor_px:
            self._reanchor()

        warped = cv2.warpPerspective(gray, self.anchor, (width, height), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(
            np.full(gray.shape, 255, np.uint8), self.anchor, (width, height), flags=cv2.INTER_NEAREST
        )
        mask = self.sub.apply(warped, learningRate=self.learning_rate)
        mask[valid == 0] = 0
        if not self.detect_shadows:
            mask[mask > 0] = 255
        else:
            mask = np.where(mask == 255, 255, 0).astype(np.uint8)

        back = cv2.warpPerspective(
            mask, np.linalg.inv(self.anchor), (width, height), flags=cv2.INTER_NEAREST
        )
        return back

    # ------------------------------------------------------------ the boxes

    def _boxes(self, fg: np.ndarray, factor: float) -> list[dict]:
        if self.open_k >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_k, self.open_k))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        if self.close_k >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_k, self.close_k))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        total = fg.shape[0] * fg.shape[1]
        if int(np.count_nonzero(fg)) > self.panic_frac * total:
            # Registration has failed badly enough that the frame is "moving".
            # Reporting anything here is guessing; drop the frame instead.
            return []

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(fg, 8)
        out: list[dict] = []
        max_area = self.max_area_frac * total
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > max_area:
                continue
            left = float(stats[i, cv2.CC_STAT_LEFT]) * factor
            top = float(stats[i, cv2.CC_STAT_TOP]) * factor
            width = float(stats[i, cv2.CC_STAT_WIDTH]) * factor
            height = float(stats[i, cv2.CC_STAT_HEIGHT]) * factor
            out.append(
                {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "area": area * factor * factor,
                    "cx": float(centroids[i][0]) * factor,
                    "cy": float(centroids[i][1]) * factor,
                }
            )
        return out

    # ---------------------------------------------------------------- entry

    REFERENCE_FPS = 30.0

    def _calibrate_time(self, ctx) -> None:
        fps = float(getattr(ctx, "fps", 0.0) or self.REFERENCE_FPS)
        if not self.history:
            self.history = max(1, int(round(self.history_s * fps)))
        if abs(fps - self.REFERENCE_FPS) > 1e-6:
            # Retention is (1-alpha) per frame; hold the time constant fixed.
            retain = min(max(1.0 - self.alpha, 1e-9), 1.0 - 1e-12)
            self.alpha = 1.0 - retain ** (self.REFERENCE_FPS / fps)
        self._timed = True

    def process(self, ctx) -> list[dict]:
        if not self._timed:
            self._calibrate_time(ctx)

        if ctx.rgba is None:
            return []
        gray = self._gray(ctx.rgba)

        if self.shape != gray.shape:
            self.shape = gray.shape
            self.prev_gray = None
            self.mean = self.var = self.age = None
            self.sub = None

        matrix, inliers = (None, 0)
        if self.prev_gray is not None:
            matrix, inliers = self._homography(self.prev_gray, gray)
            if inliers < self.min_inliers:
                matrix = None
        self.prev_gray = gray

        if self.model_kind == "gauss":
            fg = self._gauss(gray, matrix)
        else:
            fg = self._mixture(gray, matrix)
        if fg is None:
            return []

        # model px -> branch px -> source px
        factor = (ctx.src_w / float(ctx.width)) / self.scale
        return self._boxes(fg, factor)
