# Sensitivity levers

Every approach has one primary knob that decides **how much movement is enough**.
All four run the same direction: **smaller admits smaller movement, larger
demands more**. They are not in the same units, and two of them are not really
measuring movement at all — which is the main thing to know before turning one.

Pass any of them through `--cfg`:

```bash
python3 eval/run_motion.py --approach klt_homography --cfg '{"residual_floor": 6.0}'
```

## The one lever, per approach

| approach | lever | default | unit | what it literally means |
| --- | --- | --- | --- | --- |
| `klt_homography` | `residual_floor` | 12.0 | branch px | how far a tracked point must have drifted from where the camera model says it should be, measured over `lag` frames |
| `bgsub_compensated` | `k` | 4.5 | sigmas | how many standard deviations a pixel must sit from its learned background value |
| `gradient_diff` | `level` | 36.0 | grey levels (0-255) | how much a pixel's brightness must change between two registered frames |
| `baseline` | `min_speed_frac` | 0.0022 | fraction of frame width, per frame | how fast a cell must be moving |

Branch pixels are at the analysis resolution (960x540 by default), not source
pixels. One branch pixel is four source pixels at 4K.

## Two of these measure movement; two measure change

This distinction decides whether the lever does what its name suggests.

**`klt_homography` and `baseline` are true movement thresholds.**
`residual_floor` is a distance — pixels of unexplained displacement. Halve it
and a target moving half as far is picked up, regardless of what it looks like.
`min_speed_frac` is a speed, with the same property.

**`bgsub_compensated` and `gradient_diff` are change thresholds.** They ask how
different a pixel is from what was expected, and a pixel's brightness changes
because something moved *and* because that something had contrast against what
it covered. So a dark target crossing dark ground can move a long way and
barely trip `level`, while a white shirt against asphalt trips it while barely
moving. Turning these down makes the detector more sensitive to *contrast*, and
only indirectly to motion.

If you want a lever that means "how much movement", `residual_floor` on
`klt_homography` is the one that literally is that. It is also the
best-scoring approach, so this is not a trade.

## The characteristic length: `target_height`

`klt_homography` measures every pixel-valued parameter against **how tall a
target is expected to be, in source image pixels** — `target_height`, default
440. That is not a frame-size normalisation: the same camera at twice the range
halves the target without changing a pixel of resolution, and a 4K frame of a
distant field has smaller targets than a 720p frame of a close one. Frame size
is the wrong ruler; target size is the right one.

440 is measured, not chosen: the walking casualty in
`streams/lorton-d4-rgb.mp4` is a median 440 px tall across 3510 detections in a
3840x2160 frame. Every other pixel value in that approach was tuned at that
size, so at 440 the scale factor is exactly 1 and nothing is altered.

**Set it for the camera and the range, not for the resolution.** Measured on the
same clip re-encoded to 1080p, where the same person is 220 px tall:

| source | `target_height` | F1 | recall |
| --- | --- | --- | --- |
| 3840x2160 | 440 (default) | 0.937 | 0.937 |
| 1920x1080 | 220 (correct) | 0.941 | 0.942 |
| 1920x1080 | 440 (left wrong) | 0.805 | 0.703 |

Getting it right makes half the resolution behave identically; leaving it wrong
costs 0.13 F1, nearly all of it recall — the thresholds end up asking for twice
the movement the target can produce.

What it does not do is recover information that is not there. Running the *same*
source through a smaller branch (`--width 480`) scores 0.655: the scaling keeps
the parameters honest, but halving the branch halves the target's displacement
while the tracker's own jitter floor stays put, so the signal-to-noise ratio
genuinely falls. Consistent semantics, not equivalent performance.

`gradient_diff` and `bgsub_compensated` do not have this yet — their `min_area`
and morphology kernels are still raw branch pixels.

## Durations are in seconds, not frames

Every temporal parameter is quoted in **seconds** and converted against the
source frame rate. They were all tuned on 30 fps footage, so a parameter
counted in frames silently meant something different on anything else.

| approach | parameter | default | was |
| --- | --- | --- | --- |
| `klt_homography` | `lag_s` | 0.267 s | 8 frames |
| | `refresh_every_s` | 0.1 s | 3 frames |
| | `min_hits_s` | 0.1 s | 3 frames |
| | `max_misses_s` | 0.067 s | 2 frames |
| `gradient_diff` | `k_s` | 0.1 s | 3 frames |
| `bgsub_compensated` | `history_s` | 10 s | 300 frames |

Exponential-average factors — `klt_homography`'s `smooth`, `bgsub`'s `alpha`,
`baseline`'s `decay` — are not durations, but their meaning is one: they retain
a fraction of the old value *per frame*. They keep their names and their
30 fps values, and are raised to `30/fps` so they forget at the same rate in
seconds. At 60 fps a decay of 0.85 becomes 0.922, and 0.922² = 0.85.

`klt_homography` also accepts `lag` (and `gradient_diff` a `k`,
`bgsub_compensated` a `history`) as an explicit frame count, which overrides
the seconds value when you want to pin frames directly.

**What it is worth**: the 60 fps thermal clip detected in 7.9% of frames with
the old frame-based `lag: 8`, which spanned only 0.133 s there. The same
configuration expressed as 0.267 s detects in **23.3%** — the target had not
changed, only how long the detector was willing to watch it.

30 fps behaviour is unchanged by construction: all four approaches reproduce
their previous scores exactly (0.937 / 0.925 / 0.920 / 0.422).

## What "enough movement" is measured over

The threshold is meaningless without the window it is measured across, and this
is the lever most likely to be the actual answer when small movement is being
missed:

| approach | lever | default | effect |
| --- | --- | --- | --- |
| `klt_homography` | `lag_s` | 0.267 s | seconds between the two positions the camera model is fitted over |
| `gradient_diff` | `k_s` | 0.1 s | seconds between the two images differenced |
| `bgsub_compensated` | `history_s` | 10 s | seconds of appearance the background model averages |
| `baseline` | `decay` | 0.85 | per-frame retention of the flow average, rescaled by frame rate |

Longer windows accumulate real motion while noise cancels, so **raising `lag`
is often a better way to catch slow targets than lowering
`residual_floor`** — it raises the signal instead of lowering the bar. This was
the single largest effect measured anywhere in the comparison: `klt_homography`
fitted at a 1-frame baseline scores F1 0.073, and at `lag_s: 0.267` scores 0.937. Nothing else
came close to mattering that much.

The cost is latency and smearing: a target is only reported once it has been
tracked for `lag` frames, and a box drawn from an 8-frame baseline covers where
the target has been as well as where it is.

## The adaptive half of the threshold

Two approaches take the larger of a fixed floor and a multiple of the noise they
measure in that frame, so raising the floor alone may change nothing:

- `klt_homography`: `threshold = max(residual_floor, residual_scale * spread)`,
  where `spread` is the median residual of the RANSAC background inliers.
  Default `residual_scale` 8.0.
- `bgsub_compensated`: `threshold = max(k * sigma, min_diff)`, with `min_diff`
  12.0 grey levels as the absolute floor.

If a change to the floor has no effect, the adaptive term is what is binding —
turn that instead. `baseline` has the same structure (`min_speed_frac` against
`noise_scale`, default 6.0), and it is why loosening its thresholds took three
coordinated changes rather than one.

## How big, and how long

Sensitivity is not only a threshold. A target can clear the movement bar and
still be discarded for being too small or too brief:

| what it gates | klt_homography | bgsub_compensated | gradient_diff | baseline |
| --- | --- | --- | --- | --- |
| minimum size | `min_cluster_points` 2 | `min_area` 70 px | `min_area` 250 px | `min_area_frac` 0.005 |
| maximum size | — | `max_area_frac` 0.08 | — | `max_area_frac` 1.0 |
| must persist | `min_hits` 3 | — | — | — |
| may vanish for | `max_misses` 2 | — | — | — |
| must have travelled | `min_travel` 20.0 px | — | — | — |

`min_area` is the one most likely to be silently eating detections of a distant
target: `gradient_diff`'s 250 px floor at 960x540 is a target roughly 16x16
branch pixels, which is not large.

`min_travel` on `klt_homography` is a second movement threshold, applied to the
cluster rather than to individual points — a cluster that never goes anywhere is
dropped however much its points jitter. Lowering `residual_floor` without also
lowering `min_travel` will not help a very slow target.

## Levers that are honestly abstract

These do not have a plain-language meaning and are best left alone unless a
measurement points at them:

- `residual_scale`, `noise_scale` — multiples of a robust noise estimate.
- `ransac_threshold` (klt 2.0, gradient `ransac_px` 1.5) — how far a point may
  sit from the camera model and still count as background. **Raising this is
  actively dangerous**: past roughly the target's own per-frame displacement,
  RANSAC starts accepting the target as background and the camera model
  partially explains it away.
- `min_coherence`, `min_fill`, `energy_ratio`, `merge_gap_frac` — shape and
  agreement filters on the baseline. Measured at zero effect on this footage.
- `quality`/`quality_level`, `min_distance`, `block_size` — corner detector
  settings. They change where features exist at all, which moves everything
  downstream in ways that are hard to predict.

## If you want one lever across all four

There isn't one today, and a single number cannot mean the same thing to all
four while two of them threshold intensity and two threshold distance. What
would work is a `sensitivity` multiplier per approach that scales its own
primary lever and its window together — `0.5` meaning "half the movement is
enough" — implemented separately for each so the mapping stays honest. That is
a small change to each module; it has not been made.
