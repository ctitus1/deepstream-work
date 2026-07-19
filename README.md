# DeepStream YOLO Parser

Development container and scripts for running YOLO detections, injury
assessment, RTSP video input, and ROS Humble publishing through NVIDIA
DeepStream.

The container targets **DeepStream 7.1 by default**. The release is a build
argument, so 8.0 and 9.0 build from the same Dockerfile:

```bash
scripts/build.sh                      # DeepStream 7.1
DS_VERSION=9.0 scripts/build.sh       # DeepStream 9.0
```

The image is tagged `deepstream-work:<DS_VERSION>`, so several releases can
coexist. See [docker/README.md](docker/README.md) for what changes per release.

Work in this repo is organized into four areas:

| Area | Directory | What lives there |
| --- | --- | --- |
| Runtime pipeline | `src/` | the shared `deepstream_yolo` package and the app entrypoints |
| Model training | `training/` | dataset prep, fine-tuning, and the handoff into export |
| Validation | `validation/` | detection accuracy, benchmarking, timestamp diagnostics |
| Motion tracking | `tracking/` | placeholders for a future custom tracker |

The two normal runtime workflows are:

- Parser app: run `src/parser_app.py` directly with a display window and
  console logs.
- ROS publisher: run `scripts/run_stack.sh` to start the RTSP server,
  DeepStream frame source, ROS publisher, and Foxglove Bridge together.

The main entrypoints are intentionally short:

- `src/parser_app.py`: display-oriented DeepStream app.
- `src/ros_source.py`: DeepStream frame source for ROS publishing.
- `src/ros_bridge.py`: ROS Humble publisher bridge.
- `scripts/run_stack.sh`: full RTSP/ROS/Foxglove workflow helper.

The ROS publisher workflow publishes detections as `TargetBoxArray` messages
and assessments as `CasualtyImageCompressed` messages from a ROS Humble
container.

## Shared Setup

Build the DeepStream development image from the host:

```bash
scripts/build.sh
```

For the ROS publisher workflow, also build the ROS Humble image:

```bash
docker compose --profile ros build
```

The ROS Humble image includes Foxglove Bridge for visualization and expects the
`cdcl_umd_msgs` workspace at `/home/user/ros2_ws` by default. Override that path
with `CDCL_ROS_WS=/path/to/ros2_ws` if needed.

Enter the DeepStream development container:

```bash
scripts/run.sh
```

Inside the container, build the custom YOLO parser library:

```bash
scripts/build_yolo_parser.sh
```

Export the default YOLO model and generate DeepStream configs:

```bash
scripts/setup_and_export_yolo.sh yolo12x-custom.pt 640 streams/dtc-d4-trimmed.mp4
```

Export the injury assessment model:

```bash
scripts/setup_injury_model.sh models/injury.pt 8
```

Generated configs are written to `configs/generated/`. The export scripts manage
`.venv-yolo/` automatically and do not require activating a virtual environment.

## Video Input

RTSP is the default and preferred input path for both workflows. Start a local
RTSP stream from a DeepStream container shell:

```bash
scripts/start_rtsp_stream.sh streams/dtc-d4-trimmed.mp4
```

By default this serves:

```text
rtsp://127.0.0.1:8555/dtc-d4-trimmed
```

Both the parser app and the ROS DeepStream source use that URL by default. To
serve a different local video or mount:

```bash
RTSP_PORT=8560 RTSP_MOUNT=test scripts/start_rtsp_stream.sh streams/my-video.mp4
```

Then pass the matching RTSP URL with `--stream`:

```bash
python3 src/parser_app.py --stream rtsp://127.0.0.1:8560/test
docker compose --profile ros run --rm deepstream-ros-source \
  scripts/run_source.sh --stream rtsp://127.0.0.1:8560/test
```

For quick debugging, both DeepStream apps can also read a local file directly:

```bash
python3 src/parser_app.py --stream streams/dtc-d4-trimmed.mp4
docker compose --profile ros run --rm deepstream-ros-source \
  scripts/run_source.sh --stream streams/dtc-d4-trimmed.mp4
```

Local-file input is useful for development, but RTSP better matches the live
pipeline: it is paced by the stream clock, drops late frames instead of queueing
them, and can expose network/reference timestamp metadata.

## Parser App

![DeepStream parser app flow](outputs/diagrams/parser_flow.svg)

Run the parser app from another DeepStream container shell:

```bash
python3 src/parser_app.py
```

The plain command defaults to:

```bash
python3 src/parser_app.py \
  --model yolo12x-custom.pt \
  --long-side 640 \
  --enable-assessment
```

Useful parser options:

```bash
python3 src/parser_app.py --no-assessment
python3 src/parser_app.py --show-assessed-only
python3 src/parser_app.py --rtsp-latency-ms 0
python3 src/parser_app.py --show-gst-scan-warnings
python3 src/parser_app.py --record
python3 src/parser_app.py --record outputs/run42.mp4
python3 src/parser_app.py --help
```

`--record` writes the annotated video (detection boxes and assessment text
burned in) to an mp4. With no path it auto-names one under `outputs/`. The
recording branch taps the frame after `nvdsosd` and stays on the GPU, so it
costs a hardware encode rather than a readback. Recording needs a clean
shutdown to finalize the mp4 container: quit with `q` or Ctrl-C rather than
killing the process, otherwise the file has no moov atom and will not play.

By default, every display frame is shown; assessment overlay text appears only
on frames where fresh assessment tensor output is present. Use
`--show-assessed-only` to display only frames with updated assessments.

Assessment logs are grouped by frame timestamp:

```text
ASSESS frame=915 timestamp=22:46:20.242Z timestamp_source=ref detect_ms=4.26 detect_fps=234.74 assess_ms=7.58 assess_fps=131.93
  object=0 bbox=1278,358,362,128 person 0 injuries: | manikin  hem-  resp- | head-  torso- | upper+  lower+  eyes_nt
  object=1 bbox=562,639,335,131 person 1 injuries: | human  hem-  resp- | head-  torso+ | upper+  lower+  eyes_nt
```

`object=` matches the `person #` assessment label. By default, every fresh
assessment for every frame is logged. Set `--assessment-log-interval` to a
positive number to sample logs, or a negative number to disable assessment logs.
`detect_ms` is wall-clock time from mux output to detection output, and
`assess_ms` is wall-clock time from detection output to assessment output. The
matching FPS values are computed from those stage times.

## ROS Publisher

![DeepStream ROS data flow](outputs/diagrams/data_flow.svg)

The editable draw.io sources and preview generator live in
`outputs/diagrams/`.

The ROS publishing workflow uses two containers:

- `deepstream-ros-source`: runs `scripts/run_source.sh`, which starts
  `src/ros_source.py`. It forks raw, detect, and assess frame outputs,
  downsizes each image to `640x368`, JPEG-compresses them, and sends frame
  metadata over local TCP.
- `ros-humble-publisher`: runs `scripts/run_bridge.sh`, which starts
  `src/ros_bridge.py`. It receives those frames and publishes ROS Humble
  `cdcl_umd_msgs` messages with the JPEG image embedded in each message.

`scripts/run_stack.sh` starts those containers plus the RTSP server and
Foxglove Bridge. With `--bag`, it also runs `scripts/record_bag.sh` to record
all ROS topics to MCAP.

From a host shell, start the full RTSP, ROS publisher, Foxglove, and DeepStream
source stack:

```bash
scripts/run_stack.sh
```

Press Ctrl-C in that shell to stop and remove the containers started by the
script. To also record all ROS topics to an MCAP bag under `outputs/rosbags/`:

```bash
scripts/run_stack.sh --bag
```

To serve a different video:

```bash
scripts/run_stack.sh --video streams/my-video.mp4 --rtsp-mount my-video
```

Connect Foxglove Studio to:

```text
ws://localhost:8765
```

To run the components separately for debugging, start the publisher, Foxglove,
and DeepStream source from separate host shells:

```bash
docker compose --profile ros run --rm ros-humble-publisher
```

```bash
docker compose --profile ros run --rm ros-foxglove-bridge
```

```bash
docker compose --profile ros run --rm deepstream-ros-source
```

Published topics:

```text
/uas4/image
/uas4/target_detections
/casualty_image/compressed/annotated
```

Foxglove should show:

```text
/uas4/image [sensor_msgs/msg/CompressedImage]
/uas4/target_detections [cdcl_umd_msgs/msg/TargetBoxArray]
/casualty_image/compressed/annotated [cdcl_umd_msgs/msg/CasualtyImageCompressed]
```

The bridge listens on `0.0.0.0:5609` for raw image frames, `0.0.0.0:5610` for
detect frames, and `0.0.0.0:5611` for assess frames. The DeepStream source
connects to `127.0.0.1:5609`, `127.0.0.1:5610`, and `127.0.0.1:5611` by
default. Both services use host networking.

Each ROS publisher node logs the metadata associated with the published
message. The image node publishes the raw input frame as a compressed image
before detection. The detect node publishes one `TargetBoxArray` per detect
frame; each person bbox becomes a `TargetBox` with bbox coordinates scaled to
the compressed `640x368` detect image, YOLO confidence, and `DETECTION_YOLO`.
The assess node publishes one `CasualtyImageCompressed` per assessed bbox with
bbox coordinates scaled to the compressed `640x368` assessment image, embedded
image, and injury probabilities as `Annotation[]`. Wire metadata also includes
`source_bbox` and source/image dimensions for debugging. Annotation field names
use the existing `clip_rgb_<injury_head>` convention, such as
`clip_rgb_severe_hemorrhage`, and observations are probability vectors in the
class-index order used by the injury model. Detect frames publish continuously;
compressed frame outputs publish only when fresh metadata matches the compressed
payload. Multiple casualties from the same frame are published as separate
`CasualtyImageCompressed` messages with the same frame timestamp and the same
image `data_source_id`; bbox fields identify the casualty within that image.
Raw image, detection, and assessment metadata all use the immutable source frame
timestamp captured before detection. The bridge copies that same timestamp into
each compressed image and ROS message header for that frame.

Use `ROS_DOMAIN_ID` if your ROS graph needs a non-default domain:

```bash
ROS_DOMAIN_ID=7 docker compose --profile ros run --rm ros-humble-publisher
```

To run Foxglove Bridge on a different port:

```bash
FOXGLOVE_PORT=8766 docker compose --profile ros run --rm ros-foxglove-bridge
```

## RTSP Timing

The RTSP pipeline preserves reference timestamp metadata when GStreamer exposes
it. Local MP4 streams served by `scripts/start_rtsp_stream.sh` get network time
from the RTSP server clock; original camera wall-clock time is only available if
the upstream source provides it.

RTSP jitterbuffer latency defaults to 0 ms: old network/decode buffers are
dropped instead of queued, and detection runs on the newest frames available.
Override with `--rtsp-latency-ms` only if a stream needs extra buffering. RTSP
streams are paced by the stream clock; late display frames are dropped instead
of queued.

## Model Training

`training/` covers fine-tuning a detector and handing it to the export path
this repo already uses. See [training/README.md](training/README.md).

```bash
python3 training/prepare_dataset.py --format coco \
  --annotations data/instances.json --images data/images --out datasets/injury
python3 training/finetune.py --weights yolo11n.pt --data datasets/injury/injury.yaml
python3 training/export_to_deepstream.py runs/detect/train/weights/best.pt --long-side 640
```

Training runs in `.venv-yolo` (see `requirements/training.txt`), not the
DeepStream interpreter. `export_to_deepstream.py` deliberately refuses when a
fine-tuned model's class count disagrees with `labels/coco_labels.txt`: the
export script installs the 80-class COCO labels unconditionally, so a custom
model would otherwise deploy with the wrong labels and a wrong
`num-detected-classes`.

## Validation and Benchmarking

`validation/` answers three separate questions. See
[validation/README.md](validation/README.md).

```bash
python3 validation/smoke_pipeline.py --frames 60          # does the pipeline run at all
python3 validation/accuracy/dump_detections.py --out dets.json
python3 validation/accuracy/score_detections.py --detections dets.json --gt gt.json
python3 validation/benchmark/benchmark_pipeline.py --frames 300
```

`smoke_pipeline.py` is headless (`display=False`) and bounded, so it works over
SSH and in CI as a build gate: it fails if the parser library is missing, the
engine cannot be built, or no frames arrive.

Accuracy numbers are easy to misread. The deployed `nvinfer` thresholds
(`pre-cluster-threshold=0.25`, `nms-iou-threshold=0.45`) are much stricter than
the ultralytics `val` defaults (`conf=0.001`, `iou=0.7`), and the pipeline runs
non-square (for example 640x384) while ultralytics only evaluates square. Both
gaps make a correctly deployed model look worse than it is; the validation
README explains how to compare like with like.

## Motion Tracking

`tracking/` is a placeholder work area for a custom motion tracker. Nothing
there is implemented yet — the Python stubs raise `NotImplementedError` rather
than return fake results. It records the integration surface: the `NvMOT` entry
points a low-level tracker library must export, how `nvtracker` loads one via
`ll-lib-file`, the six tracker configs DeepStream 7.1 ships, and how tracking
would be scored. See [tracking/README.md](tracking/README.md).

## Local Artifacts

Large runtime artifacts are intentionally ignored by Git, including
`.venv-yolo/`, `external/`, `lib/*.so`, `models/*`, `configs/generated/`,
`outputs/`, and `__pycache__/`.

`streams/` is ignored because videos are user-provided input media. Cleanup
scripts do not remove it.

Preview cleanup:

```bash
scripts/clean_artifacts.sh
```

Remove generated artifacts:

```bash
scripts/clean_artifacts.sh --force
```

Use `--include-models` only when you also want to remove generated/downloaded
model artifacts.
