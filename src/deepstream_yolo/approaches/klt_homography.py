"""Sparse KLT tracks, a RANSAC homography for the camera, clustered outliers.

The baseline asks the flow question everywhere, including the large textureless
regions of ground and sky where the answer is meaningless -- that is where its
false positives come from. This approach never asks it there. Corner features
only exist where there is texture, so uniform regions contribute no points at
all rather than contributing noise that has to be thresholded away later.

Mechanically, per frame pair:

  1. Shi-Tomasi corners on the previous frame at branch resolution.
  2. Pyramidal Lucas-Kanade to the current frame, with a forward-backward
     consistency check so a track that cannot be re-found is discarded rather
     than believed.
  3. ``findHomography(..., RANSAC)`` over the surviving correspondences. The
     inliers are the background: a homography is the correct model for a camera
     that shakes, pans, rotates or zooms over a mostly distant scene, and it is
     fitted to point correspondences rather than to a dense field, so a handful
     of bad vectors cannot drag it.
  4. Points whose residual against that model exceeds a threshold scaled to the
     model's own fit quality are candidate movers.
  5. Those are clustered by position (union-find over a uniform grid -- sklearn
     is not in the image). A person's limbs move at different velocities but
     their corners are spatially adjacent, so position is the right key.
  6. A cluster must be re-found for several consecutive frames before it is
     reported. Sparse outliers flicker; a walking person does not.
"""

from __future__ import annotations

import cv2
import numpy as np

NAME = "klt-homography"


def _grid_clusters(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Union-find clustering by proximity, bucketed on a grid of cell=radius.

    Single-link clustering is what suits a human target: the corners on a
    person form a chain down the body rather than a tight ball, so demanding
    that every pair be close would split one person into several boxes.
    """
    count = len(points)
    if count == 0:
        return []

    parent = np.arange(count)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    buckets: dict[tuple[int, int], list[int]] = {}
    keys = np.floor(points / radius).astype(np.int32)
    for index, (cx, cy) in enumerate(keys):
        buckets.setdefault((int(cx), int(cy)), []).append(index)

    radius_sq = radius * radius
    for (cx, cy), members in buckets.items():
        # Only half the neighbourhood: the other half is covered when that
        # bucket is visited, and a pair must not be tested twice.
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 1)):
            other = buckets.get((cx + dx, cy + dy))
            if not other:
                continue
            for i in members:
                for j in other:
                    if i >= j and (dx, dy) != (0, 0):
                        continue
                    if i == j:
                        continue
                    delta = points[i] - points[j]
                    if delta[0] * delta[0] + delta[1] * delta[1] <= radius_sq:
                        union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return [np.asarray(members, dtype=np.int32) for members in groups.values()]


class _Track:
    """A candidate mover, kept across frames so flicker can be rejected."""

    __slots__ = ("cx", "cy", "box", "hits", "misses", "points")

    def __init__(self, cx: float, cy: float, box: tuple[float, float, float, float], points: int):
        self.cx = cx
        self.cy = cy
        self.box = box
        self.hits = 1
        self.misses = 0
        self.points = points

    def update(self, cx: float, cy: float, box, points: int, smooth: float) -> None:
        self.cx = smooth * self.cx + (1.0 - smooth) * cx
        self.cy = smooth * self.cy + (1.0 - smooth) * cy
        self.box = tuple(smooth * a + (1.0 - smooth) * b for a, b in zip(self.box, box))
        self.hits += 1
        self.misses = 0
        self.points = points


class Approach:
    needs_flow = False
    needs_pixels = True

    defaults = {
        # feature detection
        "max_corners": 1200,
        "quality_level": 0.01,
        "min_distance": 6.0,
        "block_size": 5,
        # LK tracking
        "win_size": 21,
        "max_level": 3,
        "fb_threshold": 1.0,
        # camera model
        "ransac_threshold": 2.0,
        "min_correspondences": 20,
        # what counts as a moving point
        "residual_floor": 1.2,
        "residual_scale": 3.0,
        # clustering, in branch pixels
        "cluster_radius": 26.0,
        "min_cluster_points": 3,
        "min_cluster_residual": 1.5,
        # temporal persistence
        "match_radius": 70.0,
        "min_hits": 3,
        "max_misses": 2,
        "smooth": 0.5,
        # reported box
        "box_pad": 8.0,
        "min_box": 24.0,
    }

    def __init__(self, cfg: dict):
        settings = dict(self.defaults)
        settings.update({k: v for k, v in (cfg or {}).items() if k in self.defaults})
        for key, value in settings.items():
            setattr(self, key, value)

        self.block_size = int(self.block_size)
        self.max_corners = int(self.max_corners)
        self.max_level = int(self.max_level)
        self.min_cluster_points = int(self.min_cluster_points)
        self.min_hits = int(self.min_hits)
        self.max_misses = int(self.max_misses)
        self.min_correspondences = int(self.min_correspondences)

        self._lk = {
            "winSize": (int(self.win_size), int(self.win_size)),
            "maxLevel": self.max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        }
        self.prev_gray: np.ndarray | None = None
        self.tracks: list[_Track] = []

    # -- helpers ---------------------------------------------------------

    def _gray(self, ctx) -> np.ndarray | None:
        rgba = ctx.rgba
        if rgba is None:
            return None
        # The surface may be padded to a hardware pitch; take the real width.
        frame = np.asarray(rgba)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        frame = frame[: ctx.height, : ctx.width, :]
        # cvtColor copies, which matters: the caller unmaps the surface as soon
        # as process() returns, so nothing that aliases it may be kept.
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY if frame.shape[2] == 4 else cv2.COLOR_RGB2GRAY)

    def _moving_points(self, gray: np.ndarray):
        """Correspondences that the camera model cannot explain."""
        prev = self.prev_gray
        corners = cv2.goodFeaturesToTrack(
            prev,
            maxCorners=self.max_corners,
            qualityLevel=float(self.quality_level),
            minDistance=float(self.min_distance),
            blockSize=self.block_size,
        )
        if corners is None or len(corners) < self.min_correspondences:
            return None, None

        p0 = corners.reshape(-1, 1, 2).astype(np.float32)
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **self._lk)
        if p1 is None:
            return None, None
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **self._lk)
        if back is None:
            return None, None

        ok = (status.reshape(-1) == 1) & (status_back.reshape(-1) == 1)
        fb = np.linalg.norm((p0 - back).reshape(-1, 2), axis=1)
        ok &= fb <= float(self.fb_threshold)
        if int(ok.sum()) < self.min_correspondences:
            return None, None

        a = p0.reshape(-1, 2)[ok]
        b = p1.reshape(-1, 2)[ok]

        matrix, inliers = cv2.findHomography(a, b, cv2.RANSAC, float(self.ransac_threshold))
        if matrix is None:
            return None, None

        predicted = cv2.perspectiveTransform(a.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        residual = np.linalg.norm(b - predicted, axis=1)

        inlier_mask = inliers.reshape(-1).astype(bool) if inliers is not None else None
        if inlier_mask is not None and inlier_mask.sum() >= 8:
            spread = float(np.median(residual[inlier_mask]))
        else:
            spread = float(np.median(residual))
        threshold = max(float(self.residual_floor), self.residual_scale * spread)

        moving = residual > threshold
        if inlier_mask is not None:
            # RANSAC already decided these belong to the background model.
            moving &= ~inlier_mask
        return b[moving], residual[moving]

    # -- contract --------------------------------------------------------

    def process(self, ctx) -> list[dict]:
        gray = self._gray(ctx)
        if gray is None:
            return []
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return []

        points, residual = self._moving_points(gray)
        self.prev_gray = gray

        clusters = []
        if points is not None and len(points):
            for members in _grid_clusters(points, float(self.cluster_radius)):
                if len(members) < self.min_cluster_points:
                    continue
                group = points[members]
                strength = float(np.mean(residual[members]))
                if strength < float(self.min_cluster_residual):
                    continue
                left, top = group.min(axis=0)
                right, bottom = group.max(axis=0)
                clusters.append(
                    {
                        "cx": float(group[:, 0].mean()),
                        "cy": float(group[:, 1].mean()),
                        "box": (float(left), float(top), float(right), float(bottom)),
                        "points": int(len(members)),
                        "residual": strength,
                    }
                )

        return self._report(ctx, clusters)

    def _report(self, ctx, clusters: list[dict]) -> list[dict]:
        """Associate clusters with running tracks and emit the persistent ones."""
        unmatched = list(range(len(clusters)))
        for track in self.tracks:
            best, best_distance = -1, float(self.match_radius)
            for index in unmatched:
                cluster = clusters[index]
                distance = np.hypot(cluster["cx"] - track.cx, cluster["cy"] - track.cy)
                if distance < best_distance:
                    best, best_distance = index, distance
            if best >= 0:
                cluster = clusters[best]
                track.update(
                    cluster["cx"], cluster["cy"], cluster["box"], cluster["points"], float(self.smooth)
                )
                unmatched.remove(best)
            else:
                track.misses += 1

        for index in unmatched:
            cluster = clusters[index]
            self.tracks.append(
                _Track(cluster["cx"], cluster["cy"], cluster["box"], cluster["points"])
            )

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        scale_x = ctx.src_w / float(ctx.width)
        scale_y = ctx.src_h / float(ctx.height)
        pad = float(self.box_pad)
        floor = float(self.min_box)

        boxes = []
        for track in self.tracks:
            if track.hits < self.min_hits:
                continue
            left, top, right, bottom = track.box
            left, top, right, bottom = left - pad, top - pad, right + pad, bottom + pad
            width, height = right - left, bottom - top
            if width < floor:
                left, width = track.cx - floor / 2.0, floor
            if height < floor:
                top, height = track.cy - floor / 2.0, floor
            boxes.append(
                {
                    "left": left * scale_x,
                    "top": top * scale_y,
                    "width": width * scale_x,
                    "height": height * scale_y,
                    "points": track.points,
                    "hits": track.hits,
                }
            )
        return boxes
