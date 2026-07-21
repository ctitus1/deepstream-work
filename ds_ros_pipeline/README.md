# ds_ros_pipeline — Signal-Driven DeepStream + ROS2 Pipeline

One Python process, one container: a ROS2 Humble node (`ds_pipeline`) driving
two independent GStreamer pipelines. The **live pipeline** decodes one
swappable video source, paces it to real time, stamps every frame with
NTP-synced wall time at ingest, and fans out to a continuous low-res preview,
an on-demand frame grabber, and a dynamically attached disk recorder. The
**batch pipeline** runs yolo12x detection — and, only when asked, injury-CLIP
assessment — over frames the grabber queued. The live pipeline contains no
inference elements at all; a running batch can never stall it. All control is
ROS2 services; all outputs carry the ingest timestamp.

Everything here is *what it does and how to run it*. For *why* each element,
property, and ordering is the way it is (all of it empirically validated
in-container), see [DESIGN.md](DESIGN.md) — the exact settings there are
load-bearing, not tuning.

## Architecture

```
live pipeline (no inference)
  source bin ──► identity pace ──► tee t_ingest
  (file: AU-replay   (sync=true      ├─ q_grab(leaky,4) ─ mux ─ conv RGBA ─ fakesink
   feeder loop;       for file        │    └─ grab probe: mosaic/vlm/enqueue/snapshot
   rtsp: never        sources —       ├─ q_preview(leaky,1) ─ conv ─ 640x360 JPEG ─ appsink
   loops)             30 fps real     │    └─ /ds/preview/compressed, ~30 Hz
                      time)           └─ q_rec(leaky,30) ─ ... recorder, attached on demand

batch pipeline (persistent, idle until fed)
  appsrc ◄─ batch worker ◄─ queued frames from the grab probe
  ─► nvvideoconvert ─► nvstreammux b8 ─► yolo12x nvinfer ─► valve ─► injury-CLIP nvinfer ─► fakesink
        (valve open only for *_assess runs)
```

Every tee branch sits behind its own leaky queue, so no slow consumer can
stall the source or a sibling (DESIGN.md §3.2). File sources are replayed in
the compressed domain by an in-memory access-unit feeder — that is what makes
seamless looping trivial and why the decoder never sees a loop boundary.

## Prerequisites

- **Existing repo setup done** with the yolo12x detector: `scripts/setup.sh
  --model yolo12x.pt` must have produced
  `configs/generated/config_infer_primary_yolo12x_640_640x384.txt`,
  `models/yolo12x_640_640x384.onnx`, the labels snapshot, and
  `lib/libnvdsinfer_custom_impl_Yolo.so`; the injury CLIP b8 config and
  engine (`configs/generated/config_infer_secondary_injury_clip_vit_l14_336_b8.txt`)
  are reused verbatim.
- **A built `cdcl_umd_msgs` colcon workspace.** Set `CDCL_ROS_WS` (default
  `/home/user/ros2_ws`); it is mounted read-only *at the path it was built
  at* (symlink-install workspaces break silently anywhere else — same caveat
  as the root compose). `run.sh` refuses to start if
  `$CDCL_ROS_SETUP` does not exist rather than failing on the first publish.
- **Host clock NTP-synced.** The container shares the host clock and every
  published stamp is `time.time_ns()` at ingest — if the host loses NTP sync,
  every stamp is wrong. This is an operational prerequisite, not something
  the node can detect for you.
- **Disk speed for recording.** H.265 at 200 Mbps CBR is 25 MB/s (1.5 GB/min)
  — any SSD. Raw I420 at 2560×1440×30 fps is **166 MB/s** — a real SSD (SATA
  is OK, NVMe comfortable, HDD not viable). The recorder's disk queue rides
  out ~1.5 s stalls; beyond that, whole frames are dropped and counted.
- A test video, default `streams/lorton-d4-rgb-nano.mp4` (2560×1440, 30 fps,
  HEVC; 287 frames / 9.564 s per loop as the pipeline sees it).

## Bring-up

```bash
docker compose -f ds_ros_pipeline/compose.yaml up --build ds-ros-pipeline
```

That builds `deepstream-work:ds-ros` (a ROS2 Humble layer on top of the
existing `deepstream-work:7.1` — the base image and every existing compose
service are untouched) and starts the node via `run.sh`.

**First run builds the batch-8 yolo12x TensorRT engine (~1–3 min).** Watch
the log; services are advertised but detection runs wait until nvinfer
finishes. The engine is cached at
`models/yolo12x_640_640x384.onnx_b8_gpu0_fp16.engine` through the bind
mount, so it is a one-time cost.

Run `ros2` commands from a second shell inside the same container:

```bash
docker compose -f ds_ros_pipeline/compose.yaml exec ds-ros-pipeline bash -lc \
  'source /opt/ros/humble/setup.bash && source $CDCL_ROS_SETUP && ros2 topic hz /ds/preview/compressed'
```

Expect ~29–30 Hz preview (not ~200 Hz — the trunk is paced to real time) and
the `loop_count` value in `/ds/status` incrementing every ~9.6 s. Override
any parameter at launch by appending to the compose command, e.g.

```bash
docker compose -f ds_ros_pipeline/compose.yaml run --rm ds-ros-pipeline \
  bash -lc 'ds_ros_pipeline/run.sh --ros-args -p source.loop:=false'
```

The full validated test matrix (16 scripted checks, one observable each) is
DESIGN.md §10.

## Services

All are `std_srvs/srv/Trigger` unless noted. Responses carry the observable
result in `message`; failures are `success=false` with a reason, never an
exception.

| Service | Semantics |
|---|---|
| `/ds/capture/mosaic` | Arm one-shot: the **next** frame is JPEG-encoded full-res and published once on `/mosaic_compressed`. Blocks ≤2 s; `message` = the stamp used. |
| `/ds/capture/vlm` | Same, but publishes raw `rgb8` on `/vlm_raw` (~11 MB/msg). |
| `/ds/batch/enqueue` | Queue the next frame for batch inference. `message` = resulting depth; `success=false` if the queue (cap `batch.capacity`) is full. N calls queue N distinct frames. |
| `/ds/batch/clear` | Empty the *pending* queue (a snapshot already taken by a running batch is unaffected). |
| `/ds/batch/run_detect` | Run detection over everything queued; one `TargetBoxArray` per frame on `/ds/detections`. Blocks ≤30 s. `success=false` if the queue is empty or a continuous mode is on. |
| `/ds/batch/run_detect_assess` | Same, plus injury assessment: one `CasualtyImageCompressed` per detected person on `/ds/assessments`. Assessment runs **only** here / in continuous-assess mode. |
| `/ds/mode/continuous_detect` (`SetBool`) | `true`: auto-enqueue every `continuous.stride`-th frame (default 3 ≈ 10 Hz) and auto-run. `false`: stop and discard any not-yet-run pending frames (count reported in the response; a run already in flight completes). Manual run services are rejected while on. |
| `/ds/mode/continuous_detect_assess` (`SetBool`) | Same with assessment. Turning either mode on turns the other off. |
| `/ds/record/start` | Attach the H.265/MPEG-TS recorder branch. `message` = file path. `success=false` if already recording or the source ended. Records seamlessly across loop wraps. |
| `/ds/record/start_raw` | Attach the raw I420 recorder branch instead. |
| `/ds/record/stop` | Detach and drain the recorder. `message` = path, frames written, frames dropped, `drained=true|false`. `drained=false` means the drain timed out (`record.stop_timeout`) and the file was force-finalized — still playable by container choice. The live pipeline is never stalled by a stop. Idempotent after a source-EOS finalization. |
| `/ds/snapshot` | Next frame → full-res PNG in `snapshot.output_dir`. Blocks until the file is closed; `message` = path. |
| `/ds/snapshot_raw` | Same as `.ppm` (raw RGB, trivially parseable) + `.json` sidecar with the stamp. |

Two capture calls arriving before the next frame **coalesce**: one message,
both callers succeed with the same stamp. `enqueue` deliberately does not —
it is a counter. Blocking services do not serialize unrelated ones (four
callback groups; a snapshot in flight does not delay a `record/start`, an
`enqueue`, or a concurrent `capture/vlm`).

## Topics

| Topic | Type | QoS | Notes |
|---|---|---|---|
| `/mosaic_compressed` | `sensor_msgs/CompressedImage` | RELIABLE, KEEP_LAST 5, TRANSIENT_LOCAL | one-shot; latched, so `echo` started *after* the call still receives it |
| `/vlm_raw` | `sensor_msgs/Image` | RELIABLE, KEEP_LAST 1, TRANSIENT_LOCAL | full-res `rgb8`, ~11 MB/msg |
| `/ds/preview/compressed` | `sensor_msgs/CompressedImage` | BEST_EFFORT, KEEP_LAST 1 (sensor data) | continuous ~30 Hz, 640×360 JPEG q75 |
| `/ds/detections` | `cdcl_umd_msgs/TargetBoxArray` | RELIABLE, KEEP_LAST 10 | one per batched frame; `source_img` = 640×368 JPEG of that frame |
| `/ds/assessments` | `cdcl_umd_msgs/CasualtyImageCompressed` | RELIABLE, KEEP_LAST 10 | one per assessed person; 8 `clip_rgb_*` annotations |
| `/ds/status` | `diagnostic_msgs/DiagnosticArray` | RELIABLE, KEEP_LAST 1 | 1 Hz, see below |

Every `header.stamp` is the frame's resolved ingest time (NTP-synced wall
clock), copied one-stamp-everywhere like the existing `ros_bridge.py`.

### `/ds/status` observables

One `DiagnosticStatus` named `ds_pipeline`, level OK while running, WARN
after the source ended. Key/values:

| Key | Meaning |
|---|---|
| `state` | `running` or `ended` (non-looping source hit EOS; node stays alive) |
| `mode` | continuous mode: `off`, `detect`, or `detect_assess` |
| `queue_depth` | pending batch-queue depth |
| `recording` | recorder state: `idle`, `recording`, or `finalized` (source EOS finalized it) |
| `loop_count` | the feeder's loop counter — increments every ~9.6 s with the default clip; its steady cadence is the pacing observable |
| `enqueue_drops` | manual enqueues refused because the queue was full |
| `continuous_skips` | continuous-mode frames skipped because the queue was full |
| `copy_failures` | surface→numpy copy failures |
| `resolve_skips` | frames skipped for an unresolvable ntp stamp |

## Parameters

All declared by `config.py`, all overridable via `--ros-args -p name:=value`.

| Parameter | Default | Meaning |
|---|---|---|
| `source.uri` | `file://streams/lorton-d4-rgb-nano.mp4` | `file://` or `rtsp(s)://`; the source bin is the single swap point |
| `source.loop` | `true` | file variant only: seamless AU-replay loop; `false` = one pass then EOS (see below) |
| `source.max_preload_mb` | `1024` | refuse (with a clear log) to preload an AU list bigger than this; use `loop:=false` for long files |
| `batch.capacity` | `16` | batch queue cap (~236 MB host RAM at full) |
| `batch.engine_batch` | `8` | engine/mux batch size; drop to 4 if VRAM is tight |
| `continuous.stride` | `3` | continuous mode samples every Nth frame (3 ≈ 10 Hz at paced 30 fps) |
| `continuous.run_size` | `4` | continuous worker auto-runs at this queue depth |
| `preview.width` / `preview.height` / `preview.quality` | `640` / `360` / `75` | preview branch geometry / JPEG quality |
| `record.bitrate` | `200000000` | H.265 CBR bitrate (bps) |
| `record.output_dir` | `outputs/ds_ros` | recordings + sidecars |
| `record.stop_timeout` | `5.0` | seconds to wait for the asynchronous drain before force-finalizing |
| `snapshot.output_dir` | `outputs/ds_ros` | snapshots + sidecars |
| `detections.image_width` / `detections.image_height` | `640` / `368` | `source_img` JPEG size in detections |
| `frame_id` | `ds_camera` | frame_id on every published header |

## Recordings & snapshots on disk

All under `outputs/ds_ros/` by default; filenames embed the UTC ingest time
of the first frame (`20260720T153001.123Z` format).

| Output | Files | Format |
|---|---|---|
| `/ds/record/start` | `rec_<UTC>.ts` + `rec_<UTC>.jsonl` | H.265 in MPEG-TS, CBR 200 Mbps, 1 s closed GOPs, parameter sets repeated at every IDR |
| `/ds/record/start_raw` | `rec_<UTC>_2560x1440_I420.yuv` + `.jsonl` | headerless raw I420 frames |
| `/ds/snapshot` | `snap_<UTC>.png` + `snap_<UTC>.json` | lossless full-res PNG |
| `/ds/snapshot_raw` | `snap_<UTC>.ppm` + `.json` | P6 PPM (header + RGB pixels) |

The `.jsonl` sidecar has one `{"pts": …, "ntp_ns": …, "utc": "…"}` line per
frame **actually written** (frames dropped under disk pressure are counted in
the stop response, not in the sidecar); snapshot `.json` sidecars carry the
same fields.

Both recording containers are chosen for crash tolerance: `kill -9` (or a
drain timeout) loses at most the ~1 s tail after the last flushed GOP of a
`.ts`, and any frame-aligned prefix of the raw `.yuv` is valid. Play the raw
file with:

```bash
ffplay -f rawvideo -pixel_format yuv420p -video_size 2560x1440 \
  outputs/ds_ros/rec_<UTC>_2560x1440_I420.yuv
```

(Adjust `-video_size` to the source resolution baked into the filename.)

## Where files land (the one-folder rule and its exception)

Everything new lives in `ds_ros_pipeline/`, including the runtime-generated
batch-8 yolo12x nvinfer config (`ds_ros_pipeline/generated/`, gitignored).
The **single exception** is TensorRT engine files: nvinfer caches them next
to their ONNX in `models/` (`models/yolo12x_640_640x384.onnx_b8_gpu0_fp16.engine`)
— that is nvinfer's own behavior and matches the repo's existing engine
convention.

## Loop behavior and RASL notes

With `source.loop:=true` (default) the file's compressed access units are
replayed with monotonic timestamps — the decoder never sees a boundary, and
pacing, recording, and timestamps are seamless across any number of wraps.
Two related, expected artifacts (details: DESIGN.md §5, §11 risk 11):

- **Cold start**: the decoder discards the clip's ~16 leading (RASL)
  pictures once at process start — the very first loop delivers 271/287
  frames. Standard HEVC CRA-start behavior; do not chase it as a drop bug.
- **Loop wraps**: at every wrap those RASL pictures *are* emitted, but they
  decode against the previous loop's tail rather than the pre-clip frames
  they were encoded against. Timing is measured gapless; the first ~0.5 s
  after each wrap (every ~9.6 s) may show brief visual corruption in preview
  and recordings. Cosmetic, test-asset-only (the rtsp variant never loops).

With `source.loop:=false` the feeder pushes one pass and signals EOS: an
active recording is finalized by that EOS (the `.ts` is complete and valid),
`/ds/status` flips to `ended` and keeps publishing, capture/snapshot/enqueue/
record-start calls fail fast with `"source ended"`, `record/stop` returns the
finalized file's stats, and `run_detect*` still work on frames already
queued. The process runs until SIGINT.

## Troubleshooting

- **`run.sh` exits immediately complaining about `CDCL_ROS_SETUP`** — the
  `cdcl_umd_msgs` workspace is missing or mounted at the wrong path. Set
  `CDCL_ROS_WS` to the workspace root (the directory containing
  `install/setup.bash`); compose mounts it read-only at that same path.
- **Long pause before the first detection run** — the one-time b8 engine
  build (~1–3 min). Subsequent starts load the cached engine from `models/`.
- **First loop is 16 frames short** — expected cold-start RASL discard, once
  per process (above).
- **Brief garbage right after a loop wrap** — expected RASL wrap artifact
  (above); timing and counts are unaffected.
- **`enqueue` returns `success=false`** — batch queue full
  (`batch.capacity`); run or clear it, or lower the enqueue rate.
- **`run_detect` returns `success=false`** — empty queue, or a continuous
  mode is on (manual runs are rejected while it is).
- **`record/stop` reports `drained=false`** — the drain hit
  `record.stop_timeout` (wedged disk, usually with the raw variant). The
  file is force-finalized and every byte on disk is still playable.
- **Frames dropped during recording** — the disk could not sustain the rate;
  drops are whole raw frames counted in the stop response, and the `.jsonl`
  sidecar matches what is actually in the file. Check the disk-speed
  prerequisite (raw needs 166 MB/s).
- **Stamps look wrong / offset from wall clock** — check host NTP sync
  (`chronyc tracking` / `timedatectl`); the node stamps from the system
  clock and cannot correct an unsynced host.
- **VRAM pressure on an 8 GB GPU** — `nvidia-smi` while recording +
  continuous assess (DESIGN.md §11 risk 1); fallbacks are
  `batch.engine_batch:=4` and shrinking the recorder queue.
- **`/vlm_raw` subscribers on a remote host miss messages** — ~11 MB
  messages are fine over localhost (`network_mode: host`); remote DDS
  consumers may need Fast DDS large-message tuning.
- **Startup refuses a large file for looping** — the AU preload exceeds
  `source.max_preload_mb`; loop RAM cost scales with clip size (~120 MB for
  the 10 s test clip). Use `source.loop:=false` for long files.

## Development

`ds_node.py` is the entrypoint; `run.sh` sources the ROS environments, puts
`src/` on `PYTHONPATH` for the shared `deepstream_yolo` helpers, and execs
it. Modules import each other flat (`import frames`). The GPU/ROS-free unit
tests run anywhere:

```bash
python3 ds_ros_pipeline/tests.py        # no dependencies beyond the stdlib
python3 -m pytest ds_ros_pipeline/tests.py   # equivalent, if pytest is installed
```
