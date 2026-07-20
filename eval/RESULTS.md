# Tracking a moving target from a shaky, compressed aerial camera

Seven approaches, each built on its own branch, each scored by the same
harness against the same ground truth. This is what was measured and what it
means.

## Scoreboard

Whole video, 5361 frames. `F1` is the ranking metric. `onstat` counts boxes
landing on a person known to be standing still — the worst failure. `ms` is the
approach's own median cost per frame; the budget is 33 (30 fps).

| branch | F1 | prec | recall | box/f | onstat | onbg | ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`feature/klt-homography`** | **0.937** | 0.936 | 0.937 | 0.65 | 80 | 141 | **8.3** |
| `feature/fastmcd` | 0.773 | 0.733 | 0.817 | 0.78 | 140 | 980 | 23.1 |
| `feature/trajectory-filter` | 0.752 | 0.797 | 0.711 | 0.58 | 40 | 593 | 12.8 |
| `feature/quadratic-flow-cost` | 0.624 | 0.558 | 0.707 | 0.86 | 56 | 1975 | 23.4 |
| baseline, retuned only | 0.632 | 0.603 | 0.665 | 0.71 | 33 | 1487 | 16.0 |
| baseline as shipped | 0.422 | 0.713 | 0.300 | 0.27 | 18 | 401 | 8.3 |

Every number is from one scorer run over one `detections.json`, not from the
branches' own reports.

`feature/bgsub-compensated` and `feature/gradient-diff` are pushed but had not
produced a scored whole-video run when this was written.

## The winner: sparse KLT tracks, homography over a rolling window

`feature/klt-homography` — F1 0.937, precision 0.936, recall 0.937, at the same
8.3 ms/frame as the baseline it more than doubles. It is the best on every axis
except `onstat`.

Corner features are detected, tracked with pyramidal Lucas-Kanade under
forward-backward checking, and a homography is fitted by RANSAC. Its inliers
are the background; its outliers are clustered into targets.

**The single change that made it work was the temporal baseline.** Fitted
between *adjacent* frames — the obvious reading — it scored **0.073**, far worse
than the baseline it was meant to beat. At branch resolution the walker
displaces ~4 px/frame against ~0.3 px of tracking noise, which sounds ample and
is not: the background still throws 20-130 residual outliers per frame while the
walker carries only a handful of corners. Fitting instead between where each
track was **8 frames ago** and where it is now, the walker's residual grows to
11-26 px while the noise floor stays put. Motion integrates; jitter does not.
That one change took it from 0.073 to 0.853, and tuning carried it to 0.937.

Why it beats the dense-flow family on this footage: a corner detector never
places a feature in a textureless region, so the failure mode that dominated the
baseline — optical flow returning large arbitrary vectors over uniform ground,
20-80 px/frame of pure noise — cannot occur. It is structurally absent rather
than filtered out afterwards.

## What the losing approaches proved

**Track-before-detect is nearly free precision** (`trajectory-filter`, 0.752).
Its `n_init` sweep is the cleanest result in the set: as confirmation tightens
from 2 to 25 frames, recall barely moves (0.678 → 0.711) while precision goes
0.234 → 0.797. Surviving tracks live a median ~190 frames and spurious ones ~7,
so confirmation can be made very expensive before it costs a real detection. The
cost the metric does not show is 0.83 s of latency before a new target appears.

**Background modelling wins recall** (`fastmcd`, 0.773, recall 0.817). Its MOG2
control — same post-processing, only the motion compensation removed — scored
**0.025** and put 8527 boxes on stationary people, confirming that an
un-warpable background model is worthless on a moving camera.

**The baseline's operating point was the bug, not its algorithm.** Simply
loosening three thresholds takes it 0.422 → 0.632 with no code change. Its
shipped configuration was tuned for per-frame precision and paid recall 0.300
for it.

## Three negative results worth more than the wins

**The `nvof` cost plane is the wrong signal.** DeepStream exposes `output-cost`
and pyds does not bind it, but it is reachable via `pyds.get_ptr` + ctypes.
Measured, cost runs *lower* on the unreliable frame border (4.6 vs 7.4 interior)
and *higher* on genuinely moving cells (15.7 vs 7.1). Used as a confidence
weight it collapses the detector to F1 0.021 — it starves the target and trusts
the artifacts. It tracks how much motion a cell contains, not how trustworthy
its vector is. The sign was not flipped to rescue it; that would be fitting
interpretation to outcome.

**The quadratic background model was worth −0.003.** Predicted to be the
highest-value fix — camera motion over a plane is a homography whose flow
expansion is quadratic, and the dropped x² term is maximal exactly where the
baseline's errors concentrated. The defect is real, but the shipped 20-cell
border crop had already discarded the annulus where quadratic differs from
affine. It had been paid for by throwing away the frame edge.

**Repairs can be an expensive way to buy tuning.** `quadratic-flow-cost` reached
0.624, but its own control showed the unrepaired model at the same loosened
thresholds reaches 0.616 — in 13.7 ms instead of 23.4, with half the errors on
stationary people. Three principled repairs were worth +0.008.

## Where the winner still fails

- **`onstat` 80, against the baseline's 18.** The people labelled "stationary"
  still shift 1-4 branch px per frame, and over an 8-frame window that is real
  motion; one of the three accounts for two-thirds. Anything sensitive enough to
  catch a distant walker calls a weight-shifting bystander a mover too.
- **Single-link clustering at a 100 px radius** merges two targets closer than
  that into one box. Only 38 frames here contain two movers, so the metric never
  charged for it — it would matter on busier footage.
- **~10-frame warm-up** while the window fills, and residual recall loss
  concentrated in the last ~600 frames.

## Combination: attempted, not concluded

`fastmcd` (best recall) feeding `trajectory-filter`'s confirmation (best
precision) is the combination the results argue for, and
`src/deepstream_yolo/approaches/fastmcd_tracked.py` implements it on
`feature/optical-flow`. It did not complete a scored whole-video run before this
was written, so **no claim is made for it**. The pairing worth trying next is
trajectory confirmation behind the KLT front end, aimed squarely at the `onstat`
80.

## Ground truth, and a correction

Derived from the detector rather than declared. `eval/tracks.py` links
detections into tracks and identifies the *stationary* people — one long
unbroken track each, centroid never leaving a small box. Every other detection
is a mover. That direction is deliberate: the walker outruns the detector and
arrives as dozens of short fragments, so picking them out directly is fragile,
while the three stationary tracks (spread 192-301 px over thousands of frames)
are unmistakable against every walker fragment (1145 px and up).

**The stated expectation was that the first half contains no movement and a
fourth person enters in the second half. The data disagrees**: the mover is
present in 3472 of 5361 frames, from t≈30s, detected at confidence 0.6-0.7 with
a consistent person-sized area in 262-300 of every 300 frames from t=40s. All
scoring uses the derived ground truth, not the stated split.

## Reproducing

```bash
python3 eval/dump_detections.py                       # once
python3 eval/run_motion.py --approach klt_homography
python3 eval/score.py eval/runs/*.json
```

See `eval/README.md` for the metric and the plugin interface.

**A harness caveat.** One `fastmcd` run returned zero boxes across all 5361
frames at a normal 20.5 ms/frame — `get_nvds_buf_surface` handed back a surface
it had not filled, and `run_motion.py` swallows that silently, so it presents as
a bad score rather than an error. An anomalously empty result deserves a second
run before it is believed.
