"""Control for ``fastmcd``: OpenCV MOG2, same post-processing, no warping.

Not a serious contender -- it exists to measure one specific claim. MOG2 is a
better *per-pixel* background model than fastMCD's single Gaussian: it carries
a mixture, so it can hold a multi-modal background (foliage, glint) that one
mean and variance cannot. What it cannot do is move. There is no coherent way
to resample a per-pixel Gaussian mixture through a homography -- interpolating
between two mixtures does not give a mixture of the same order, and the weights
and the per-component variances have no meaningful bilinear blend -- so when the
camera pans, MOG2's model refers to the wrong piece of scene and stays that way
until its learning rate walks it back.

fastMCD's age-weighted single Gaussian is the weaker model chosen because it
*can* be resampled. This module measures what that trade is worth on this
footage: identical grid-free pixel decisions, identical dilation, connected
components and merge, differing only in whether the model is motion
compensated.

Same knobs as ``fastmcd`` where they overlap, so the comparison is not
confounded by post-processing.
"""

from __future__ import annotations

import cv2
import numpy as np

from .fastmcd import _merge

NAME = "fastmcd-control-mog2"


class Approach:
    needs_flow = False
    needs_pixels = True

    def __init__(self, cfg: dict):
        c = dict(cfg or {})
        self.history = int(c.get("history", 500))
        self.var_threshold = float(c.get("var_threshold", 16.0))
        self.shadows = bool(c.get("shadows", False))
        self.learning_rate = float(c.get("learning_rate", -1.0))
        self.knn = bool(c.get("knn", False))

        self.dilate = int(c.get("dilate", 3))
        self.min_area = int(c.get("min_area", 48))
        self.merge_gap = int(c.get("merge_gap", 16))

        if self.knn:
            self.model = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.var_threshold * 25.0,
                detectShadows=self.shadows,
            )
        else:
            self.model = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.shadows,
            )

    def process(self, ctx) -> list[dict]:
        if ctx.rgba is None:
            return []

        frame = np.asarray(ctx.rgba)
        h = min(frame.shape[0], ctx.height)
        w = min(frame.shape[1], ctx.width)
        gray = cv2.cvtColor(np.ascontiguousarray(frame[:h, :w]), cv2.COLOR_RGBA2GRAY)

        mask = self.model.apply(gray, learningRate=self.learning_rate)
        mask = (mask > 200).astype(np.uint8)  # drop shadow label 127 if enabled

        if self.dilate > 0:
            mask = cv2.dilate(mask, np.ones((self.dilate, self.dilate), np.uint8))

        count, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        raw = []
        for k in range(1, count):
            x, y, bw, bh, area = stats[k]
            if area < self.min_area:
                continue
            raw.append([int(x), int(y), int(x + bw), int(y + bh), int(area)])

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
            for x0, y0, x1, y1, area in _merge(raw, self.merge_gap)
        ]
