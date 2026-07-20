# tracking/ — custom motion tracker work area

**Everything in this directory is a placeholder.** No tracker is implemented, no
library is built, no metric is computed. The files exist so the shape of the work
is obvious and so the real implementation has somewhere to land. Python stubs
raise `NotImplementedError`; the C++ skeleton returns `NvMOTStatus_Error`. Nothing
here silently returns a plausible-looking result.

## Goal

The pipeline today is detection-only: `nvinfer` (PGIE, YOLO) emits per-frame
boxes, and `src/deepstream_yolo/detection_overlay.py` assigns a *per-frame*
`person_index` into `obj.misc_obj_info[0]`. That id is re-derived from scratch on
every frame — it is an ordering artifact, not an identity. Anything that needs
temporal continuity (motion, dwell, trajectory, "is this the same person the
assessment SGIE scored two seconds ago") has nothing to stand on.

A motion tracker fills that gap: it associates detections across frames, assigns
persistent `object_id`s, and holds targets through short occlusions and detector
dropouts using a motion model. Concretely we want:

- persistent per-person ids stable across occlusion and missed detections
- bbox smoothing / gap filling from the state estimator when the detector misses
- per-target velocity and trajectory available downstream
- ids that survive the RTSP path's frame drops (`configure_latest_queue` sets
  `leaky=2` on the queues, so the tracker must tolerate dropped frames)

## Two viable routes

### Route A — configure and tune a shipped NVIDIA tracker (start here)

DeepStream 7.1 ships one low-level tracker library that implements several
algorithms, selected by its YAML config:

    /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so

Configs live in `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/`
(IOU, NvSORT, NvDeepSORT, NvDCF perf/accuracy/max_perf). See
[`configs/README.md`](configs/README.md) for the exact paths, which to copy, and
the knobs that matter.

This is the route that should be attempted and measured first. It requires no
C++ and no build step — only a `nvtracker` element and a YAML file. Route B is
only justified if a tuned NvDCF/NvSORT is measurably insufficient on our footage.

### Route B — custom NvMOT low-level library

`nvtracker` dlopen's whatever `.so` is named by its `ll-lib-file` property and
calls a fixed C ABI (`NvMOT_Query`, `NvMOT_Init`, `NvMOT_Process`,
`NvMOT_RemoveStreams`, `NvMOT_DeInit`, plus `NvMOT_RetrieveMiscData`). Replacing
that library replaces the entire association/motion model while keeping the
GStreamer plumbing, batching, and metadata attachment.

The API is defined in
`/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvdstracker.h`. The real
signatures, the ownership rules, and the build wiring are documented in
[`nvmot_custom/README.md`](nvmot_custom/README.md), with a compile-shaped
skeleton in [`nvmot_custom/nvmot_custom_tracker.cpp`](nvmot_custom/nvmot_custom_tracker.cpp).

## Where nvtracker slots into the pipeline

`src/deepstream_yolo/pipeline.py::build_pipeline()` currently branches directly
off `pgie`. In `build_pipeline` today, the PGIE output fans out three ways
(roughly lines 292-311):

```python
if detect_tee and detect_output_size:
    pgie.link(detect_tee)
    ...
elif assessment_queue:
    pgie.link(assessment_queue)
else:
    pgie.link(convert)
```

`nvtracker` belongs between PGIE and every one of those branches — it must see
the detector's object meta and everything downstream must see the tracked ids.
The change is mechanical:

1. create `tracker = element("nvtracker", "tracker")` alongside the other
   elements, gated on a new `tracker_config` argument (same pattern as
   `assessment_config` gating `sgie`);
2. set `ll-lib-file`, `ll-config-file`, `tracker-width`, `tracker-height`, and
   `gpu-id` on it (`tracker-width`/`-height` must be multiples of 32);
3. insert it into the `elements` list immediately after `pgie` and before
   `detect_tee`;
4. replace the three `pgie.link(...)` calls with `pgie.link(tracker)` followed by
   the same fan-out from `tracker`;
5. add a `tracker: Gst.Element | None` field to `PipelineParts`.

Ordering notes that matter:

- **Upstream of the assessment SGIE.** `sgie` runs with `process-mode=2`
  (secondary, per-object). With the tracker in front of it, each assessment
  result is attached to an object that already has a persistent `object_id`, so
  `assessment_runtime.py` can accumulate per-track rather than per-frame.
- **Upstream of `raw_tee`? No.** `raw_tee` sits between `streammux` and `pgie`
  and carries pre-inference frames; the tracker needs object meta and must stay
  after `pgie`.
- **Upstream of the overlay probe.** `bbox_probe()` is attached at the PGIE src
  pad. Once a tracker exists, that probe should move to the tracker src pad so
  the drawn label can use `obj_meta.object_id` instead of the synthetic
  `person_index` from `set_detection_id()`. Until the tracker is real, leave the
  probe where it is.
- `obj_meta.object_id` is `UNTRACKED_OBJECT_ID` (`0xFFFFFFFFFFFFFFFF`) when no
  tracker is present — any consumer must handle that sentinel.

## Evaluation

Route A vs. Route B, and any tuning within either, gets decided by numbers, not
by watching the display sink.

- **Accuracy.** MOT Challenge metrics — MOTA, MOTP, IDF1, HOTA, plus ID switch
  and fragmentation counts. `metrics/score_tracks.py` is the placeholder CLI for
  this; it defines the MOT txt format both ground truth and predictions must be
  written in.
- **Ground truth.** We have none yet. A few hundred frames of
  a clip from `streams/` hand-labelled with persistent ids is the minimum
  useful set; that labelling effort is the real prerequisite for this whole
  directory.
- **Predictions.** Need an exporter: a pad probe on the tracker src pad walking
  `frame_meta.obj_meta_list` and writing one MOT txt line per object per frame.
  Does not exist yet.
- **Cost.** Per-stage latency and end-to-end FPS with and without the tracker.
  `src/deepstream_yolo/timing.py` already has the timing plumbing to reuse.
- **Stability under drops.** The RTSP path drops frames by design. Compare id
  churn on the file source vs. the RTSP source for the same clip.

## Layout

```
tracking/
  README.md                       this file
  configs/README.md               shipped tracker configs, what to copy, key knobs
  nvmot_custom/README.md          NvMOT C ABI, real signatures, ll-lib-file wiring
  nvmot_custom/nvmot_custom_tracker.cpp   skeleton, all entry points TODO
  nvmot_custom/Makefile           untested placeholder build (CUDA_VER convention)
  metrics/score_tracks.py         placeholder MOT metrics CLI
```

## TODO before any of this is real

- [ ] Label ground-truth tracks on a clip from `streams/`.
- [ ] Write the tracker-src-pad MOT txt exporter probe.
- [ ] Implement `metrics/score_tracks.py` against py-motmetrics or TrackEval.
- [ ] Add the `nvtracker` element to `pipeline.py` behind a `--tracker-config` flag.
- [ ] Baseline Route A: NvDCF_perf and NvSORT, scored.
- [ ] Only then decide whether Route B is worth writing.
