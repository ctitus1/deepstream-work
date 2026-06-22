# DeepStream YOLO Parser

Development container and scripts for running YOLO detections, optional injury
assessment, and RTSP video input through NVIDIA DeepStream.

## Setup

Build and enter the development container from the host:

```bash
scripts/build.sh
scripts/run.sh
```

Inside the container, build the custom YOLO parser library:

```bash
scripts/build_yolo_parser.sh
```

Export the YOLO model and generate DeepStream configs:

```bash
scripts/setup_and_export_yolo.sh yolo12x-custom.pt 640 streams/dtc-d4-trimmed.mp4
```

Generated configs are written to `configs/generated/`. The export scripts manage
`.venv-yolo/` automatically and do not require activating a virtual environment.

## Run With RTSP

Start a local RTSP stream from one container shell:

```bash
scripts/start_rtsp_stream.sh streams/dtc-d4-trimmed.mp4
```

By default this serves:

```text
rtsp://127.0.0.1:8555/dtc-d4-trimmed
```

Run the parser app from another container shell:

```bash
python3 src/deepstream_yolo_parser_app.py --model yolo12x-custom.pt --long-side 640
```

The parser defaults to the RTSP URL above. To use a different RTSP stream:

```bash
python3 src/deepstream_yolo_parser_app.py \
  --model yolo12x-custom.pt \
  --long-side 640 \
  --stream rtsp://127.0.0.1:8560/test
```

For quick local-file debugging, pass a file path:

```bash
python3 src/deepstream_yolo_parser_app.py \
  --model yolo12x-custom.pt \
  --long-side 640 \
  --stream streams/dtc-d4-trimmed.mp4
```

The RTSP pipeline preserves reference timestamp metadata when GStreamer exposes
it. Local MP4 streams served by `scripts/start_rtsp_stream.sh` get network time
from the RTSP server clock; original camera wall-clock time is only available if
the upstream source provides it. RTSP jitterbuffer latency defaults to 0 ms: old
network/decode buffers are dropped instead of queued, and detection runs on the
newest frames available. Override with `--rtsp-latency-ms` only if a stream needs
extra buffering.

## Injury Assessment

Export the injury model:

```bash
scripts/setup_injury_model.sh models/injury.pt 8
```

Run YOLO detections with injury assessment:

```bash
python3 src/deepstream_yolo_parser_app.py \
  --model yolo12x-custom.pt \
  --long-side 640 \
  --enable-assessment
```

Assessment logs look like:

```text
ASSESS frame=915 timestamp=22:46:20.242Z timestamp_source=ref compute_ms=11.84 fps=29.91
  object=0 bbox=1278,358,362,128 person 0 injuries: | manikin  hem-  resp- | head-  torso- | upper+  lower+  eyes_nt
  object=1 bbox=562,639,335,131 person 1 injuries: | human  hem-  resp- | head-  torso+ | upper+  lower+  eyes_nt
```

`object=` matches the `person #` assessment label. By default, every fresh
assessment for every frame is logged. Set `--assessment-log-interval` to a
positive number to sample logs, or a negative number to disable assessment logs.
Assessment overlay text is only shown on frames where fresh assessment tensor
output is present. `compute_ms` is wall-clock time from mux output to assessment
output for that frame, and `fps` is the assessed-frame output rate once a prior
assessed frame exists.

To display only frames with updated assessments:

```bash
python3 src/deepstream_yolo_parser_app.py \
  --model yolo12x-custom.pt \
  --long-side 640 \
  --enable-assessment \
  --show-assessed-only
```

## Useful Options

```bash
python3 src/deepstream_yolo_parser_app.py --help
python3 src/deepstream_yolo_parser_app.py --show-gst-scan-warnings
python3 src/deepstream_yolo_parser_app.py --rtsp-latency-ms 0
RTSP_PORT=8560 RTSP_MOUNT=test scripts/start_rtsp_stream.sh streams/my-video.mp4
```

## Local Artifacts

Large runtime artifacts are intentionally ignored by Git, including `.venv-yolo/`,
`external/`, `lib/*.so`, `models/*`, `configs/generated/`, `outputs/`, and
`__pycache__/`.

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
