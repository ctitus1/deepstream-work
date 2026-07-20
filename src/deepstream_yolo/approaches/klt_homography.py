"""Sparse KLT tracks, a RANSAC homography for the camera, clustered outliers.

The baseline evaluates flow over every cell of the frame, including the large
uniform regions of ground and sky where a flow vector carries no information --
that is where most of its false positives come from. This approach never asks
the question there. Corner features only exist where there is texture, so a
featureless region contributes no points at all rather than contributing noise
that then has to be thresholded away.

The one measurement that decided the shape of everything else: at branch
resolution the walker displaces a median of 4 px between consecutive frames,
against a per-point tracking noise of roughly 0.3 px -- but the background
throws 20-130 residual outliers per frame at that separation, and the walker
carries only a handful of corners, so the mover cannot be picked out. Widen the
baseline to six frames and the walker's residual grows to 11-26 px while the
noise floor stays where it was. Motion integrates; jitter does not. So features
are *tracked* across a rolling window rather than matched between neighbouring
frames, and the camera model is fitted across that whole window.

Per frame:

  1. Every live track is followed one frame with pyramidal Lucas-Kanade and a
     forward-backward consistency check, so a track that cannot be re-found is
     dropped rather than believed.
  2. The population is topped back up with Shi-Tomasi corners, masked away from
     the tracks that already exist so new points land in uncovered ground.
  3. ``findHomography(..., RANSAC)`` over the window: where each track was
     ``lag`` frames ago against where it is now. The inliers are the background.
     A homography is the right model for a camera that shakes, pans, rotates or
     zooms over a mostly distant scene, and fitting it to point correspondences
     rather than to a dense field means a few bad vectors cannot drag it.
  4. Tracks the model cannot explain are candidate movers.
  5. Those are clustered by position -- union-find over a uniform grid, since
     sklearn is not in the image. A person's limbs move at different velocities
     but their corners are spatially adjacent, so position is the right key.
  6. A cluster must be re-found for several consecutive frames before it is
     reported. Sparse outliers flicker; a walking person does not.
"""

from __future__ import annotations

import cv2
import numpy as np

NAME = "klt-homography"


def _grid_clusters(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Single-link clustering by proximity, bucketed on a grid of cell=radius.

    Single-link is what suits a human target: the corners on a person form a
    chain down the body rather than a tight ball, so demanding that every pair
    be close would split one person into several boxes.
    """
    count = len(points)
    if count == 0:
        return []

    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(np.floor(points / radius).astype(np.int32)):
        buckets.setdefault((int(key[0]), int(key[1])), []).append(index)

    radius_sq = radius * radius
    # Half the neighbourhood only; the other half is covered when that bucket
    # is itself visited, and a pair must not be tested twice.
    offsets = ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 1))
    for (cx, cy), members in buckets.items():
        for dx, dy in offsets:
            other = buckets.get((cx + dx, cy + dy))
            if not other:
                continue
            same = dx == 0 and dy == 0
            for i in members:
                for j in other:
                    if same and j <= i:
                        continue
                    ri, rj = find(i), find(j)
                    if ri == rj:
                        continue
                    delta = points[i] - points[j]
                    if delta[0] * delta[0] + delta[1] * delta[1] <= radius_sq:
                        parent[max(ri, rj)] = min(ri, rj)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return [np.asarray(members, dtype=np.int32) for members in groups.values()]


class _Candidate:
    """A cluster followed across frames, so flicker can be rejected."""

    __slots__ = ("cx", "cy", "box", "hits", "misses", "points")

    def __init__(self, cx, cy, box, points):
        self.cx, self.cy, self.box, self.points = cx, cy, box, points
        self.hits, self.misses = 1, 0

    def update(self, cx, cy, box, points, smooth):
        self.cx = smooth * self.cx + (1.0 - smooth) * cx
        self.cy = smooth * self.cy + (1.0 - smooth) * cy
        self.box = tuple(smooth * a + (1.0 - smooth) * b for a, b in zip(self.box, box))
        self.points = points
        self.hits += 1
        self.misses = 0


class Approach:
    needs_flow = False
    needs_pixels = True

    defaults = {
        # feature population, in branch pixels
        "max_corners": 1600,
        "quality_level": 0.01,
        "min_distance": 5.0,
        "block_size": 5,
        "refresh_every": 3,
        # LK tracking
        "win_size": 21,
        "max_level": 3,
        "fb_threshold": 1.0,
        # temporal baseline the camera model is fitted over
        "lag": 6,
        # camera model
        "ransac_threshold": 2.0,
        "min_correspondences": 30,
        # what counts as a moving point
        "residual_floor": 6.0,
        "residual_scale": 8.0,
        # clustering, and the evidence a cluster must carry
        "cluster_radius": 26.0,
        "min_cluster_points": 3,
        "min_cluster_energy": 0.0,
        "min_coherence": 0.0,
        # temporal persistence
        "match_radius": 70.0,
        "min_hits": 3,
        "max_misses": 2,
        "smooth": 0.5,
        # reported box
        "box_pad": 10.0,
        "min_box": 30.0,
    }

    _INTS = (
        "max_corners",
        "block_size",
        "refresh_every",
        "max_level",
        "lag",
        "min_correspondences",
        "min_cluster_points",
        "min_hits",
        "max_misses",
    )

    def __init__(self, cfg: dict):
        settings = dict(self.defaults)
        settings.update({k: v for k, v in (cfg or {}).items() if k in self.defaults})
        for key, value in settings.items():
            setattr(self, key, int(value) if key in self._INTS else value)

        self._lk = {
            "winSize": (int(self.win_size), int(self.win_size)),
            "maxLevel": self.max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        }
        self.prev_gray: np.ndarray | None = None
        self.points = np.zeros((0, 2), np.float32)
        # trail[k] is where the live points were k frames ago; trail[0] is now.
        self.trail: list[np.ndarray] = []
        self.age = np.zeros(0, np.int32)
        self.candidates: list[_Candidate] = []
        self.frame = 0

    # -- feature population ----------------------------------------------

    def _gray(self, ctx) -> np.ndarray | None:
        rgba = ctx.rgba
        if rgba is None:
            return None
        frame = np.asarray(rgba)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        # The surface can be padded out to a hardware pitch, so take the real
        # width; and cvtColor copies, which matters because the caller unmaps
        # the surface the moment process() returns.
        frame = frame[: ctx.height, : ctx.width, :]
        code = cv2.COLOR_RGBA2GRAY if frame.shape[2] == 4 else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(frame, code)

    def _replenish(self, gray: np.ndarray) -> None:
        """Add corners where no track already sits."""
        wanted = self.max_corners - len(self.points)
        if wanted <= 0:
            return
        mask = None
        if len(self.points):
            mask = np.full(gray.shape, 255, np.uint8)
            radius = max(1, int(self.min_distance))
            for x, y in self.points.astype(np.int32):
                cv2.circle(mask, (int(x), int(y)), radius, 0, -1)
        found = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=wanted,
            qualityLevel=float(self.quality_level),
            minDistance=float(self.min_distance),
            blockSize=self.block_size,
            mask=mask,
        )
        if found is None:
            return
        fresh = found.reshape(-1, 2).astype(np.float32)
        self.points = np.vstack([self.points, fresh])
        # A new track has no history, so it stands still in every past slot and
        # is excluded from the fit until it is old enough to have one.
        self.trail = [np.vstack([past, fresh]) for past in self.trail]
        self.age = np.concatenate([self.age, np.zeros(len(fresh), np.int32)])

    def _advance(self, gray: np.ndarray) -> None:
        """Carry every live track one frame forward; drop the ones that fail."""
        if not len(self.points):
            return
        p0 = self.points.reshape(-1, 1, 2)
        p1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self._lk)
        if p1 is None:
            self._reset_tracks()
            return
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self._lk)
        if back is None:
            self._reset_tracks()
            return

        ok = (status.reshape(-1) == 1) & (status_back.reshape(-1) == 1)
        ok &= np.linalg.norm((p0 - back).reshape(-1, 2), axis=1) <= float(self.fb_threshold)
        moved = p1.reshape(-1, 2)
        height, width = gray.shape
        ok &= (moved[:, 0] >= 1) & (moved[:, 0] < width - 1)
        ok &= (moved[:, 1] >= 1) & (moved[:, 1] < height - 1)

        self.points = moved[ok]
        self.trail = [past[ok] for past in self.trail]
        self.age = self.age[ok] + 1

    def _reset_tracks(self) -> None:
        self.points = np.zeros((0, 2), np.float32)
        self.trail = []
        self.age = np.zeros(0, np.int32)

    # -- camera model ----------------------------------------------------

    def _moving_points(self):
        """Tracks whose displacement over the window the camera cannot explain.

        Returns their current positions and their residual *vectors*: the
        direction of the unexplained motion is evidence in its own right.
        """
        if len(self.trail) <= self.lag:
            return None, None
        mature = self.age >= self.lag
        if int(mature.sum()) < self.min_correspondences:
            return None, None

        then = self.trail[self.lag][mature]
        now = self.points[mature]
        matrix, inliers = cv2.findHomography(then, now, cv2.RANSAC, float(self.ransac_threshold))
        if matrix is None:
            return None, None

        predicted = cv2.perspectiveTransform(then.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        residual = np.linalg.norm(now - predicted, axis=1)

        inlier_mask = inliers.reshape(-1).astype(bool) if inliers is not None else None
        if inlier_mask is not None and int(inlier_mask.sum()) >= 8:
            spread = float(np.median(residual[inlier_mask]))
        else:
            spread = float(np.median(residual))
        threshold = max(float(self.residual_floor), float(self.residual_scale) * spread)

        moving = residual > threshold
        if inlier_mask is not None:
            # RANSAC has already ruled that these belong to the background.
            moving &= ~inlier_mask
        return now[moving], (now - predicted)[moving]

    # -- contract --------------------------------------------------------

    def process(self, ctx) -> list[dict]:
        gray = self._gray(ctx)
        if gray is None:
            return []
        self.frame += 1

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self._reset_tracks()
            self.prev_gray = gray
            self._replenish(gray)
            self.trail = [self.points.copy()]
            return []

        self._advance(gray)
        if self.refresh_every <= 1 or self.frame % self.refresh_every == 0:
            self._replenish(gray)
        self.trail.insert(0, self.points.copy())
        del self.trail[self.lag + 1 :]
        self.prev_gray = gray

        points, vectors = self._moving_points()

        clusters = []
        if points is not None and len(points):
            magnitude = np.linalg.norm(vectors, axis=1)
            for members in _grid_clusters(points, float(self.cluster_radius)):
                if len(members) < self.min_cluster_points:
                    continue
                strength = magnitude[members]
                # Energy rather than a count: two corners displaced a long way
                # are as much evidence as five displaced a little, and a person
                # far from the camera only ever offers the former.
                if float(strength.sum()) < float(self.min_cluster_energy):
                    continue
                # Points on one body share a direction of travel even though
                # limbs differ in speed. Foliage and parallax do not: their
                # residuals point every way at once and cancel.
                if self.min_coherence > 0.0:
                    group_vectors = vectors[members]
                    total = float(np.linalg.norm(group_vectors.sum(axis=0)))
                    if total < float(self.min_coherence) * float(strength.sum()):
                        continue
                group = points[members]
                left, top = group.min(axis=0)
                right, bottom = group.max(axis=0)
                clusters.append(
                    {
                        "cx": float(group[:, 0].mean()),
                        "cy": float(group[:, 1].mean()),
                        "box": (float(left), float(top), float(right), float(bottom)),
                        "points": int(len(members)),
                        "residual": float(strength.mean()),
                    }
                )

        return self._report(ctx, clusters)

    def _report(self, ctx, clusters: list[dict]) -> list[dict]:
        """Associate clusters with running candidates; emit the persistent ones."""
        unmatched = list(range(len(clusters)))
        for candidate in self.candidates:
            best, best_distance = -1, float(self.match_radius)
            for index in unmatched:
                cluster = clusters[index]
                distance = float(np.hypot(cluster["cx"] - candidate.cx, cluster["cy"] - candidate.cy))
                if distance < best_distance:
                    best, best_distance = index, distance
            if best >= 0:
                cluster = clusters[best]
                candidate.update(
                    cluster["cx"], cluster["cy"], cluster["box"], cluster["points"], float(self.smooth)
                )
                unmatched.remove(best)
            else:
                candidate.misses += 1

        for index in unmatched:
            cluster = clusters[index]
            self.candidates.append(
                _Candidate(cluster["cx"], cluster["cy"], cluster["box"], cluster["points"])
            )
        self.candidates = [c for c in self.candidates if c.misses <= self.max_misses]

        scale_x = ctx.src_w / float(ctx.width)
        scale_y = ctx.src_h / float(ctx.height)
        pad, floor = float(self.box_pad), float(self.min_box)

        boxes = []
        for candidate in self.candidates:
            if candidate.hits < self.min_hits:
                continue
            left, top, right, bottom = candidate.box
            left, top, right, bottom = left - pad, top - pad, right + pad, bottom + pad
            width, height = right - left, bottom - top
            if width < floor:
                left, width = candidate.cx - floor / 2.0, floor
            if height < floor:
                top, height = candidate.cy - floor / 2.0, floor
            boxes.append(
                {
                    "left": left * scale_x,
                    "top": top * scale_y,
                    "width": width * scale_x,
                    "height": height * scale_y,
                    "points": candidate.points,
                    "hits": candidate.hits,
                }
            )
        return boxes
