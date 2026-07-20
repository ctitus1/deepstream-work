"""Interchangeable motion-detection approaches, one module each.

Every module here exposes the same small contract so ``eval/run_motion.py`` can
run any of them over the same frames and ``eval/score.py`` can compare them on
the same metric:

    NAME = "short-name"

    class Approach:
        needs_flow = True      # give me ctx.flow  (nvof field)
        needs_pixels = False   # give me ctx.rgba  (frame pixels)

        def __init__(self, cfg: dict): ...

        def process(self, ctx) -> list[dict]:
            '''Boxes as {"left","top","width","height", ...} in SOURCE pixels.'''

Approaches are free to keep state across frames -- most need to. They must not
cap how many boxes they return: robustness to few or many targets is part of
what is being measured.
"""
