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
  ─► nvvideoconvert ─► nvstreammux b1 ─► yolo12x nvinfer ─► valve ─► injury-CLIP nvinfer ─► fakesink
        (valve open only for *_assess runs)
```

**Why the mux is `batch-size=1`.** Every batched frame enters on the mux's
single `sink_0` pad, so the legacy `nvstreammux` labels them all
`source_id=0`. When it packs several such frames into one batch, `nvinfer`
annotates only `batch_id=0` and every later frame emerges with zero objects
— a 6-frame run published boxes on frame 0 and empty arrays on frames 1–5,
and it did so even when all six pushed buffers were byte-identical copies,
so this is the mux/`nvinfer` single-source contract, not frame content.
`batch-size=1` makes the mux emit one frame per batch, which is the case
`nvinfer` handles correctly for one source; frames still stream through
back-to-back, they are just no longer inferred 8-up. The nvinfer configs
stay `b8` (the engine accepts any batch in [1, 8]). Recovering true batched
inference means giving the mux **one sink pad per frame** — a separate
`appsrc ! nvvideoconvert` chain per batch slot, so each frame arrives as its
own source — which is the canonical DeepStream multi-stream topology but
multiplies the (load-bearing, DESIGN.md §3.3) converter pools.

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

Pipeline only:

```bash
docker compose -f ds_ros_pipeline/compose.yaml up --build ds-ros-pipeline
```

Pipeline **plus a Foxglove bridge**, ready to connect Studio to
`ws://localhost:8765` — this is the one to use when you want to watch and
click rather than script:

```bash
docker compose -f ds_ros_pipeline/compose.yaml up --build
```

Omitting the service name starts everything in the file; naming
`ds-ros-pipeline` starts only the pipeline, so the DESIGN.md §10 test
commands behave exactly as they always did. See
[Foxglove Studio](#foxglove-studio) below for what to do once connected.

Either command builds `deepstream-work:ds-ros` (a ROS2 Humble layer on top of
the existing `deepstream-work:7.1` — the base image and every existing compose
service are untouched) and starts the node via `run.sh`. The second also
builds `deepstream-work:ros-humble` for the bridge if it is not already
present.

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

### DeepStream 9.0 machines

The stack defaults to the originally validated DS 7.1 + ROS2 Humble pairing.
On a machine whose base image is `deepstream-work:9.0` (Ubuntu 24.04 —
required for Blackwell-generation GPUs, whose sm_120 DS 7.1's TensorRT cannot
build engines for), select the DS 9.0 + ROS2 Jazzy pairing in an untracked
`ds_ros_pipeline/.env` (compose reads it automatically):

```
DS_VERSION=9.0
DS_ROS_DISTRO=jazzy
DS_CDCL_ROS_WS=/home/user/ros2_ws_jazzy
```

`DS_CDCL_ROS_WS` points **both** containers — pipeline and Foxglove bridge —
at a `cdcl_umd_msgs` workspace built for Jazzy/Python 3.12; the Humble/3.10
host workspace cannot be sourced under Jazzy. Build it once with the ds-ros
image itself (it carries colcon for exactly this):

```bash
docker compose -f ds_ros_pipeline/compose.yaml build ds-ros-pipeline
mkdir -p /home/user/ros2_ws_jazzy
docker run --rm -v /home/user/ros2_ws/src:/cdcl_src:ro \
  -v /home/user/ros2_ws_jazzy:/home/user/ros2_ws_jazzy \
  deepstream-work:ds-ros bash -c 'source /opt/ros/jazzy/setup.bash && \
    colcon build --paths /cdcl_src/cdcl_umd_msgs \
      --build-base /tmp/colcon_build \
      --install-base /home/user/ros2_ws_jazzy/install'
```

The Humble foxglove bridge interoperates with the Jazzy pipeline over DDS
unchanged — verified end to end on this pairing: every topic (including both
`cdcl_umd_msgs` ones) deserializes, every service is callable, preview holds
~30 Hz, and detection/assessment, recording, and snapshots all pass. The
node suppresses its Jazzy-only `~/get_type_description` service so the
humble-era bridge does not log a type-resolution WARN on every graph poll.

## Foxglove Studio

A second, optional service (`ds-ros-foxglove`) serves the whole ROS graph over
a websocket, so you can watch the streams and fire the signals by hand.

### Launching it

1. **Start the stack with the bridge.** Either bring both up together, or add
   the bridge to a pipeline that is already running — DDS discovery makes the
   order irrelevant, and the bridge can be restarted on its own at any time:

   ```bash
   # both, from cold
   docker compose -f ds_ros_pipeline/compose.yaml up --build

   # or: pipeline already up, add the bridge
   docker compose -f ds_ros_pipeline/compose.yaml up -d ds-ros-foxglove
   ```

2. **Check it is listening.** The log should end with a line naming the port,
   followed by one `Advertising new channel` line per topic:

   ```bash
   docker logs ds-ros-foxglove | grep -E 'Server listening|Advertising'
   ```

3. **Connect a Foxglove app** to **`ws://localhost:8765`** — see
   [Viewing from a desktop app](#viewing-from-a-desktop-app) below.

4. **Stop it** without touching the pipeline:

   ```bash
   docker compose -f ds_ros_pipeline/compose.yaml stop ds-ros-foxglove
   ```

`FOXGLOVE_PORT` moves the websocket off 8765; any other `foxglove_bridge`
launch argument goes through `FOXGLOVE_ARGS` (example under *Large raw
frames* below). Neither needs a file edit:

```bash
FOXGLOVE_PORT=9000 docker compose -f ds_ros_pipeline/compose.yaml up -d ds-ros-foxglove
```

### Viewing from a desktop app

The bridge listens on `0.0.0.0:8765` of the **host** (the service uses
`network_mode: host`, so there is no port to publish and no container
address to look up).

**On this machine.** Foxglove Studio is already installed here (2.57.0):

```bash
foxglove-studio        # or launch "Foxglove" from the desktop
```

In the app: *Open connection… → Foxglove WebSocket →* `ws://localhost:8765`
*→ Open*. The connection dialog remembers it, so later sessions are one
click.

**From another machine on the LAN.** Use the host's address instead of
`localhost` — this host is `192.168.1.3`:

```
ws://192.168.1.3:8765
```

Nothing needs to change server-side; just make sure port 8765 is not blocked
by a firewall between the two machines.

**In a browser** (`app.foxglove.dev`) the same `ws://localhost:8765` works,
because browsers exempt localhost from the mixed-content rule that otherwise
blocks an insecure websocket from an HTTPS page. That exemption does *not*
extend to a remote host, so use the desktop app when connecting to
`192.168.1.3`.

**Version note:** this bridge (3.4.2) speaks only the newer
`foxglove.sdk.v1` subprotocol and rejects a client offering just the legacy
`foxglove.websocket.v1` with `400 Bad Request`. Studio 2.x offers both and
negotiates `foxglove.sdk.v1` — verified against the installed 2.57.0, whose
bundle declares `SUPPORTED_SUBPROTOCOLS = ["foxglove.websocket.v1",
"foxglove.sdk.v1"]`. Only a genuinely old (1.x-era) build would fail to
connect, and the fix there is updating Studio, not a bridge flag.

### Once connected

Both of these are verified working against a live pipeline:

- **Every topic**, including the two `cdcl_umd_msgs` ones —
  `/uas4/target_detections`, `/uas4/target_detections/vlm`, and
  `/uas4/target_detections/mosaic` —
  deserialize because the colcon overlay is mounted and sourced **and the
  bridge is built for the same ROS distro as the pipeline** (see below). Use an
  Image panel on `/ds/preview/compressed` for the continuous stream,
  `/uas4/target_detections/mosaic` for the one-shot, and a Raw Message panel on
  `/ds/status`.
- **Every signal**, from Studio's Service Call panel — all services are
  advertised and were confirmed callable end to end (`enqueue` answered in
  0.01 s, `capture/mosaic` in 0.20 s, `snapshot` in 0.22 s). This is the
  fastest way to drive the pipeline by hand: fire `/ds/capture/mosaic` and
  watch exactly one frame appear, or `/ds/capture/vlm` and watch the boxes
  arrive on `/uas4/target_detections/vlm`.

#### The bridge must match the pipeline's ROS distro

`ds_ros_pipeline/compose.yaml` builds the bridge with
`ROS_DISTRO=${DS_ROS_DISTRO}` and tags it `deepstream-work:ros-${DS_ROS_DISTRO}`,
so on a DS 9.0 machine you get a **Jazzy** bridge, not a Humble one. This is
not cosmetic. `cdcl_umd_msgs/TargetBoxArray` embeds a `sensor_msgs/Range`,
Jazzy's `Range` added a `float32 variance` field that Humble's does not have,
and it sits **immediately before** `uav_target_boxes`. A Humble bridge reading
a Jazzy publisher therefore runs 4 bytes short, reads the sequence length from
the wrong offset, logs `sequence size exceeds remaining buffer`, and hands
Studio an **empty box array**. Every field before the `Range` — header, `seq`,
`source_img` — decodes correctly, so the symptom looks like "detection is
broken" rather than "the bridge is the wrong distro".

If you ever see empty `uav_target_boxes` in Studio while
`ros2 topic echo` inside the pipeline container shows boxes, this is why:
check `docker images | grep ros-` and confirm the bridge tag matches
`DS_ROS_DISTRO`.

The bridge runs in a small `ros:${DS_ROS_DISTRO}-ros-base` image that carries
`foxglove_bridge` — layering it onto the 21.6 GB
DeepStream image for a pure-visualization add-on would be the expensive way
round. The two containers find each other over host networking and host IPC,
the same way the root compose's ROS services already do, which is why the
bridge needs no GPU and can come and go independently of the pipeline.

## Services

All are `std_srvs/srv/Trigger` unless noted. Responses carry the observable
result in `message`; failures are `success=false` with a reason, never an
exception.

| Service | Semantics |
|---|---|
| `/ds/capture/mosaic` | Arm one-shot: the **next** frame is JPEG-encoded full-res (q90) and published once as a `TargetBoxArray` on `/uas4/target_detections/mosaic` — the image and its stamp, an **empty** `uav_target_boxes` (nothing is inferred on this path), and `use_for_mosaic=true`. Blocks ≤2 s; `message` = the stamp used. |
| `/ds/capture/vlm` | Arm one-shot: the **next** frame is run through **detection** (bypassing the batch queue), then one `TargetBoxArray` is published on `/uas4/target_detections/vlm` — field-for-field what `run_detect` would publish for that frame, except every box has `use_for_assessment=true`. Nothing else is published. Blocks for the capture (≤2 s) plus the run (≤10 s); rejected while a continuous mode is on; `message` = the stamp used + run counts. |
| `/ds/batch/enqueue` | Queue the next frame for batch inference. `message` = resulting depth; `success=false` if the queue (cap `batch.capacity`) is full. N calls queue N distinct frames. |
| `/ds/batch/clear` | Empty the *pending* queue (a snapshot already taken by a running batch is unaffected). |
| `/ds/batch/run_detect` | Run detection over everything queued; one **detection** `TargetBoxArray` per frame (boxes with empty `annotations` + compressed frame image) on `/uas4/target_detections`. Blocks ≤30 s. `success=false` if the queue is empty or a continuous mode is on. |
| `/ds/batch/run_detect_assess` | Same, plus injury assessment; per frame one **assessment** `TargetBoxArray` on `/uas4/target_detections` — the same boxes, with the 8 `clip_rgb_*` heads filled into the `annotations` of every assessed box (the plain detection array is not additionally published). Assessment runs **only** here / in continuous-assess mode. Every detection is a person (see Detection scope), so in practice every box is assessed. |
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
`enqueue`, or a concurrent `capture/mosaic`).

## Topics

| Topic | Type | QoS | Notes |
|---|---|---|---|
| `/uas4/target_detections/mosaic` | `cdcl_umd_msgs/TargetBoxArray` | RELIABLE, KEEP_LAST 5, TRANSIENT_LOCAL | capture/mosaic output: no boxes, `use_for_mosaic=true`, `source_img` = **full-res** q90 JPEG (not the 640×368 detection image — a mosaic is stitched from these). Latched, so `echo` started *after* the call still receives it |
| `/ds/preview/compressed` | `sensor_msgs/CompressedImage` | BEST_EFFORT, KEEP_LAST 1 (sensor data) | continuous ~30 Hz, 640×360 JPEG q75 |
| `/uas4/target_detections` | `cdcl_umd_msgs/TargetBoxArray` | RELIABLE, KEEP_LAST 10 | one per batched frame (the shared TBA topic, named like `ros_bridge.py`'s): detect runs → `annotations` empty; assess runs → the same boxes with the 8 `clip_rgb_*` heads on assessed ones; `source_img` = 640×368 JPEG of the frame; boxes are in **`source_img` pixel coordinates** (see below); `use_for_assessment=false` on every box |
| `/uas4/target_detections/vlm` | `cdcl_umd_msgs/TargetBoxArray` | RELIABLE, KEEP_LAST 10, TRANSIENT_LOCAL | capture/vlm output: one for the captured frame, message-identical to what a detect run would put on `/uas4/target_detections` except `use_for_assessment=true` on every box and `do_assessment=true` on the array. Latched, like the other one-shot outputs, so an `echo` started *after* the call still receives it. `seq` is counted separately from the batch topic's |
| `/ds/status` | `diagnostic_msgs/DiagnosticArray` | RELIABLE, KEEP_LAST 1 | 1 Hz, see below |

Every stamp field in every message is the source frame's resolved ingest
time (NTP-synced wall clock), carried with the frame through the whole
pipeline — `BatchItem`s travel with their stamp, so a `TargetBoxArray`
header and its embedded `source_img` header repeat the identical value,
exactly like the existing `ros_bridge.py`.

### Bounding-box coordinate space

`target_bbox` is expressed in **`source_img` pixels**, not source-frame
pixels. nvinfer reports boxes against the full frame (2560×1440 for the
default clip) while `source_img` is a `detections.image_width` ×
`detections.image_height` JPEG (640×368), so `ros_io` scales every box by
`(image_width / frame_width, image_height / frame_height)` at publish time.

Without that scaling, anything overlaying the boxes on the image carried in
the same message — Foxglove's Image panel included — draws them about 4×
too large and mostly off-frame. This also matches `src/ros_bridge.py`, whose
bboxes and `source_img` were already the same space.

x and y scale independently, because `source_img` is resized to exactly
640×368 without preserving aspect (2560×1440 is 1.78, 640×368 is 1.74). A
single uniform factor would leave boxes progressively misplaced toward the
bottom of the frame.

Change `detections.image_width` / `image_height` and the boxes follow
automatically — the factors are derived per frame from the actual frame
dimensions, not hardcoded. `integration/verify_pipes.py` decodes the
published JPEG and asserts every box lies inside it.

### Detection scope: people only

The pgie emits **nothing but COCO class 0 (`person`)**, at or above
`detect.min_confidence` (default `0.4`). Downstream code may therefore
assume every `Detection` and every `TargetBox` is a person — there is no class field to branch on.

This is enforced inside nvinfer, not by filtering afterwards.
`infer_configs.render_batch_yolo_config` writes:

```ini
[class-attrs-all]
pre-cluster-threshold=2.0    ; unreachable — confidence is a probability

[class-attrs-0]
pre-cluster-threshold=0.4    ; detect.min_confidence
```

Every non-person class is given a threshold no detection can ever meet, so
those boxes are discarded during bbox parsing: no object meta is created, no
probe sees them, nothing reaches a message. The `[class-attrs-0]` section
also copies the template's `nms-iou-threshold` / `topk` so the person class
does not silently inherit a different clustering policy.

The model itself is still the 80-class COCO YOLO (`num-detected-classes=80`)
— only what survives parsing changes. To widen the scope, add a
`[class-attrs-<id>]` section per class you want back.

Verified live: `integration/verify_pipes.py` asserts on every pipe that no
non-person class and no sub-threshold confidence ever appears in a published
message.

### The detection index

One index identifies a detection everywhere it appears, for a given frame.
`det_collect` (the pgie src-pad probe, upstream of the valve) walks the
frame's `obj_meta_list` and **writes** each object's ordinal position into
`obj_meta.object_id` — which is otherwise the untracked sentinel, there
being no tracker in this pipeline. From then on that number is the identity:

| Where | Field | Meaning |
|---|---|---|
| `TargetBoxArray` | position in `uav_target_boxes` | box `i` **is** detection `i` |
| `FrameResult` | `assessments[i]` | the 8 `clip_rgb_*` heads for box `i` |

The point of stamping the index into the meta is the assessment join. The
SGIE runs *after* the valve, so its probe sees the object list a second
time; having it read `object_id` back — rather than re-deriving a position
by counting the list again — means the two sides cannot drift. Two
independent counters would agree only for as long as the SGIE never
reorders, inserts, or drops an object meta, and if it ever did, one person's
injury assessment would be silently attached to a different person's box.
Reading the index back makes that join correct by construction.

Publication preserves the ordering explicitly (`_indexed_detections`
sorts by `object_id` and warns if the indices are not contiguous `0..n-1`),
so array position never has to be inferred from the order a probe happened
to emit.

This is verified live, not just asserted: the injury SGIE is configured
`operate-on-class-ids=0`, so in a frame holding a mix of classes the
annotations must land on exactly the `person` boxes and no others —
checked position by position against runs carrying up to 15 people
interleaved with `car`/`chair`/`truck`/`bottle` boxes (before the pgie was
narrowed to people, which is what made that check discriminating).

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
| `batch.engine_batch` | `8` | engine batch size; drop to 4 if VRAM is tight. Not the mux batch size — see Architecture |
| `detect.min_confidence` | `0.4` | minimum YOLO confidence for a person detection. Applied **inside** nvinfer (`[class-attrs-0] pre-cluster-threshold`), so weaker boxes never become object meta. Baked into the generated config at startup — setting it at runtime does nothing |
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
- **`ds-ros-pipeline` exits 139 (SIGSEGV) seconds after starting, with
  `N zombie ports cleaned` in the log** — a Fast DDS participant was killed
  without releasing its shared-memory segments (`docker kill`, or killing a
  `ros2` process by hand), and the next participant can fault on the
  leftovers during `rclpy` node construction. `run.sh` runs `fastdds shm
  clean` at startup precisely for this, which handles it in almost every
  case; observed once in 11 starts while processes were being killed
  manually. Just relaunch — `up -d` again succeeds, and 4/4 deliberate
  kill-then-relaunch cycles came up clean. If it ever repeats, `docker
  compose ... down` and confirm no stray participant is holding
  `/dev/shm/fastrtps_*`.
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
- **`capture/vlm` succeeds but the box array is empty** — the captured
  frame genuinely held no person at or above `detect.min_confidence`.
  Call it again, or lower the threshold.
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
