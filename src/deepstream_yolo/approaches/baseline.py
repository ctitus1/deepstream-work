"""Baseline: hardware optical flow, affine background model, temporal average.

The approach the other branches are measured against. Mechanically:

  1. ``nvof`` flow field, border cropped (its edge cells are badly wrong).
  2. An affine background model fitted with outlier rejection, subtracted, so
     what remains is motion relative to the scene rather than to the camera.
  3. An exponential moving average of the residual *vectors*. Sustained motion
     accumulates; noise that points somewhere new each frame cancels.
  4. Hysteresis threshold, dilation, connected components, merge-by-proximity.

Kept as a thin adapter over ``deepstream_yolo.motion`` rather than a copy, so
the baseline being scored is the code that actually ships.
"""

from __future__ import annotations

from dataclasses import replace

from ..motion import MotionConfig, MotionState, motion_boxes

NAME = "baseline"


class Approach:
    needs_flow = True
    needs_pixels = False

    def __init__(self, cfg: dict):
        base = MotionConfig()
        overrides = {k: v for k, v in (cfg or {}).items() if hasattr(base, k)}
        self.cfg = replace(base, **overrides) if overrides else base
        self.state = MotionState(self.cfg.decay)

    def process(self, ctx) -> list[dict]:
        if ctx.flow is None:
            return []

        boxes, stats = motion_boxes(ctx.flow, self.cfg, self.state, ctx.scale_x, ctx.scale_y)
        return [
            {
                "left": b.left,
                "top": b.top,
                "width": b.width,
                "height": b.height,
                "speed": b.energy,
                "cells": b.cells,
                "dx": b.dx,
                "dy": b.dy,
            }
            for b in boxes
        ]
