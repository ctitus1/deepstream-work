# eval/ — comparing motion approaches on one metric

Every approach is scored on the same question: **did it put a box on the person
who is actually moving, and only on them?**

## The three steps

```bash
# 1. Ground truth: detector boxes for every frame. Run once.
python3 eval/dump_detections.py                      # -> eval/detections.json

# 2. Run an approach over the whole video, motion only.
python3 eval/run_motion.py --approach baseline       # -> eval/runs/baseline.json

# 3. Score one or many runs against the ground truth.
python3 eval/score.py eval/runs/*.json
```

Run everything inside the container:
`docker compose run --rm -T deepstream-dev python3 eval/...`

## Why detections are precomputed

Detection and assessment cost far more than the motion analysis being studied.
Dumping detector boxes once means an experiment pays for one decode plus the
approach itself, and every approach is compared on identical frames.

## Ground truth: who is "the mover"

Derived from the data, not declared. `eval/tracks.py` links detections into
tracks, then identifies the **stationary** people — the easy half of the
problem, since someone standing still gives one long unbroken track whose
centroid never leaves a small box. Every detection that is *not* part of a
stationary track is a mover.

That direction matters. The walker outruns the detector, which loses and
reacquires them, so they arrive as dozens of short fragments — picking "the
mover" out of those is fragile. On this footage the two populations do not
overlap: the stationary tracks span 192-301 px over thousands of frames while
every fragment of the walker spans 1145 px or more.

Result on `streams/lorton-d4-rgb.mp4`: three stationary people, and a mover
present in 3472 of 5361 frames — from t≈30s onward, not only the second half.

## The metric

A motion box **hits** when its centroid falls inside a mover box for that frame.
Centroid-in-box rather than IoU on purpose: a flow blob legitimately covers the
space a target swept through, not its silhouette, so demanding IoU would punish
an approach for correctly reporting a trail.

| column | meaning |
| --- | --- |
| `prec` | hits / all boxes emitted |
| `recall` | frames where a mover was hit / frames containing a mover |
| `F1` | harmonic mean of the two — **the ranking metric** |
| `cover` | fraction of individual mover boxes hit (matters when >1 mover) |
| `onstat` | boxes landing on a *stationary* person — the worst false positive |
| `onbg` | boxes landing on nothing |
| `ms` | median ms/frame for the approach alone; must stay under 33 for 30fps |

There is deliberately **no cap on box count**. Robustness to few or many
targets is part of what is measured, so limiting output is not a way to score
well — precision already penalises spraying boxes.

## Adding an approach

Drop a module in `src/deepstream_yolo/approaches/`:

```python
NAME = "my-approach"

class Approach:
    needs_flow = True      # receive ctx.flow — nvof field, (rows, cols, 2) float32
    needs_pixels = False   # receive ctx.rgba — (h, w, 4) uint8 at branch resolution

    def __init__(self, cfg: dict): ...

    def process(self, ctx) -> list[dict]:
        """Boxes as {"left","top","width","height", ...} in SOURCE pixels."""
```

`ctx` also carries `frame_index`, `pts`, `src_w`, `src_h`, `width`, `height`,
`grid_size`, `scale_x`, `scale_y`. Approaches may keep state across frames;
most need to.

## Baseline

`baseline` is the shipping detector: nvof → affine background model → EMA of
residual vectors → hysteresis → connected components.

| approach | F1 | prec | recall | box/f | onstat | onbg | ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.422 | 0.713 | 0.300 | 0.27 | 18 | 401 | 8.3 |

Accurate when it fires, but finds the mover in only 30% of frames. Recall is
the obvious thing to beat.
