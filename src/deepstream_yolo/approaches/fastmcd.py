"""fastMCD: motion-compensated dual-mode background modelling.

Skeleton. Yi et al., "Detection of Moving Objects with Non-Stationary Cameras
in 5.8ms" (CVPRW 2013).
"""

from __future__ import annotations

NAME = "fastmcd"


class Approach:
    needs_flow = False
    needs_pixels = True

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})

    def process(self, ctx) -> list[dict]:
        return []
