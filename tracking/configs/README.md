# tracking/configs — tracker config starting points

**Placeholder directory.** No config is checked in yet, by the same convention as
`configs/README.md`: tracker YAMLs will carry machine-specific absolute paths
(engine files, ReID models) and are treated as local runtime artifacts. This file
records where the stock configs live and what to change in a copy.

## What ships in the container

One low-level tracker library implements every algorithm below:

    /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so

That path is the value for `nvtracker`'s `ll-lib-file`. The algorithm is chosen
entirely by the YAML passed as `ll-config-file`:

| Config (in `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/`) | Algorithm | Notes |
| --- | --- | --- |
| `config_tracker_IOU.yml` | IOU only | No state estimator, no visual features. Cheapest; useful as a floor in the metrics table. |
| `config_tracker_NvSORT.yml` | SORT | Kalman filter + IOU/size association. `stateEstimatorType: 2`, no ReID. |
| `config_tracker_NvDCF_perf.yml` | NvDCF (perf) | Adds a discriminative correlation filter for visual similarity. `stateEstimatorType: 1`. |
| `config_tracker_NvDCF_max_perf.yml` | NvDCF (max perf) | Cheapest NvDCF variant. |
| `config_tracker_NvDCF_accuracy.yml` | NvDCF + ReID reassociation | `reidType: 2`. **Needs a ReID model — see below.** |
| `config_tracker_NvDeepSORT.yml` | DeepSORT | `reidType: 1`, Mahalanobis gating. **Needs a ReID model — see below.** |

### Copy this one first

`config_tracker_NvDCF_perf.yml`. It is the best accuracy/cost balance that has no
external model dependency, and it is the configuration NVIDIA's own perf numbers
are quoted against. Baseline `config_tracker_NvSORT.yml` next (motion-only, so it
isolates how much the visual term is buying us) and `config_tracker_IOU.yml` as
the floor.

Copy into this directory, do not edit the originals:

```bash
cp /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml \
   tracking/configs/tracker_nvdcf_perf.yml
```

### ReID model is NOT in this image

`config_tracker_NvDeepSORT.yml` and `config_tracker_NvDCF_accuracy.yml` reference

    /opt/nvidia/deepstream/deepstream/samples/models/Tracker/resnet50_market1501.etlt

That directory **does not exist in `deepstream-work:7.1`** (verified). Both configs
will fail to initialize until the model is fetched — the DeepStream sample models
are a separate download. Treat those two rows as blocked, not as options.

## Key knobs

Names below are verbatim from the shipped YAMLs. Values differ per config; the
ones in parentheses are from `config_tracker_NvDCF_perf.yml`.

**BaseConfig**
- `minDetectorConfidence` (0.0430) — detections below this never enter tracking.
  Our PGIE already thresholds; set this in relation to the `--conf` the app runs
  with, or it silently double-filters.

**TargetManagement** — controls how tracks are born and killed. This is where the
occlusion behaviour lives.
- `maxTargetsPerStream` (150) — cap; costs GPU memory.
- `probationAge` (2) — frames before a tentative track is considered valid.
- `maxShadowTrackingAge` (51) — how long a track survives with no detection.
  The single most important knob for occlusion robustness; raising it trades id
  persistence against ghost tracks.
- `earlyTerminationAge` (1) — kills tentative tracks fast.
- `minIouDiff4NewTarget` (0.7418) — suppresses new tracks overlapping existing ones.
- `minTrackerConfidence` (0.4009) — below this the target drops to shadow mode.
- `enableBboxUnClipping` (1) — keeps boxes sane at frame borders.

**DataAssociator** — detection-to-track matching.
- `associationMatcherType` — `GREEDY=0`, `CASCADED=1`.
- `checkClassMatch` (1) — only associate same-class objects. We are person-only
  (`PERSON_CLASS_ID = 0` in `detection_overlay.py`), so this is nearly a no-op
  for us but leave it on.
- `minMatchingScore4Overall` / `4Iou` / `4SizeSimilarity` / `4VisualSimilarity` —
  per-term thresholds.
- `matchingScoreWeight4Iou` / `4SizeSimilarity` / `4VisualSimilarity` — term weights.
- `tentativeDetectorConfidence`, `minMatchingScore4TentativeIou` — the low-confidence
  detection path.

**StateEstimator** — the motion model.
- `stateEstimatorType`: `DUMMY=0`, `SIMPLE=1`, `REGULAR=2`.
- NvDCF style: `processNoiseVar4Loc`, `processNoiseVar4Size`, `processNoiseVar4Vel`,
  `measurementNoiseVar4Detector`, `measurementNoiseVar4Tracker`.
- NvSORT/DeepSORT style: `noiseWeightVar4Loc`, `noiseWeightVar4Vel` (noise scales
  with box height), `useAspectRatio`.
- `usePrediction4Assoc` (NvSORT) — associate against the predicted state rather
  than the last observed one. Matters on our RTSP path, where frames get dropped.

**VisualTracker** (NvDCF only)
- `visualTrackerType`: `DUMMY=0`, `NvDCF=1`.
- `useColorNames` / `useHog` — feature channels.
- `featureImgSizeLevel` (2, range 1-5) — the main NvDCF perf/accuracy dial.
- `filterLr`, `filterChannelWeightsLr`, `gaussianSigma` — correlation filter update.

**ReID** (NvDeepSORT / NvDCF_accuracy — blocked, see above)
- `reidType`: `DUMMY=0`, `NvDEEPSORT=1`, `reid-based reassociation=2`, `both=3`.
- `reidFeatureSize`, `reidHistorySize`, `inferDims`, `networkMode`.
- `tltEncodedModel`, `tltModelKey`, `modelEngineFile` — the missing paths.
- `minMatchingScore4ReidSimilarity`, `matchingScoreWeight4ReidSimilarity`,
  `thresholdMahalanobis`.

## nvtracker element properties

`ll-lib-file` and `ll-config-file` are the two that select everything above. The
rest of the element properties (`tracker-width`, `tracker-height`, `gpu-id`,
`display-tracking-id`, `tracking-id-reset-mode`, `compute-hw`,
`user-meta-pool-size`, `sub-batches`) should be confirmed on a GPU host with:

```bash
gst-inspect-1.0 nvtracker
```

That command cannot run in this container without a GPU — the plugin fails to
load with `libcuda.so.1: cannot open shared object file`, so the property list
above is from documentation, not from this image. Confirm before relying on it.

`tracker-width` and `tracker-height` must be multiples of 32; they are the
resolution the tracker's internal processing runs at, independent of
`streammux`'s `width`/`height`.

## TODO

- [ ] Copy `config_tracker_NvDCF_perf.yml` here and record the diff from stock.
- [ ] Decide the `minDetectorConfidence` / PGIE `--conf` relationship.
- [ ] Sweep `maxShadowTrackingAge` against the id-switch count from
      `../metrics/score_tracks.py`.
- [ ] Fetch the DeepStream sample ReID model if DeepSORT is worth evaluating.
