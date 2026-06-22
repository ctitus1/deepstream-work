# DeepStream YOLO Parser

Development container and scripts for running YOLO models through NVIDIA
DeepStream with the DeepStream-Yolo parser.

## Typical workflow

Build and enter the development container from the host:

```bash
scripts/build.sh
scripts/run.sh
```

If `docker-compose.yml` changes, exit any existing container and start a fresh
one with `scripts/run.sh` so the updated mount path, hostname, and hosts entries
are applied.

Inside the container, build the custom YOLO parser library if needed. The script
normalizes CUDA versions like `13.1.0` to the package suffix used by apt, such as
`13-1`.

```bash
scripts/build_yolo_parser.sh
```

Export a model and generate local DeepStream configs. This creates or repairs
`.venv-yolo/` automatically and calls `.venv-yolo/bin/python3` directly, so you
do not need to activate the virtual environment.

```bash
scripts/setup_and_export_yolo.sh yolo12x-custom.pt 640 streams/dtc-d4-trimmed.mp4
```

Generated DeepStream configs are written to `configs/generated/`.

Run the Python parser app:

```bash
python3 src/deepstream_yolo_parser_app.py --model yolo12x-custom.pt --long-side 640
```

The app reuses matching cached ONNX artifacts when available and regenerates a
local inference config under `configs/generated/`.

## Injury assessment

`models/injury.pt` is a CLIP ViT-L/14@336 image encoder with custom injury
classification heads. Inspect the checkpoint:

```bash
PYTHONPATH=src python3 -m deepstream_yolo.injury inspect --model models/injury.pt
```

Export it to ONNX and generate the secondary DeepStream config:

```bash
scripts/setup_injury_model.sh models/injury.pt 8
```

The injury exporter uses the active `python3` when it already has `torch`,
`clip`, and `onnx`; otherwise it refreshes and uses `.venv-yolo/`.

This writes `models/injury_clip_vit_l14_336.onnx`, a matching metadata file,
and `configs/generated/config_infer_secondary_injury_clip_vit_l14_336_b8.txt`.
DeepStream builds the TensorRT fp16 engine from that ONNX on first use, the same
way the YOLO path uses generated ONNX plus `nvinfer`.

Run YOLO detections and injury assessment together:

```bash
python3 src/deepstream_yolo_parser_app.py \
  --model yolo12x-custom.pt \
  --long-side 640 \
  --enable-assessment
```

The pipeline uses leaky one-buffer queues before detection and before injury
assessment, so if inference falls behind it keeps the newest frame/bbox work and
drops stale buffers. For now assessment runs as soon as person detections arrive;
the secondary inference/probe boundary is the intended place to add ROS-triggered
detection or assessment gates later.

By default, the app suppresses startup-only `gst-plugin-scanner` warnings about
optional GStreamer plugins with missing codec/runtime libraries. Runtime
DeepStream warnings and errors are still shown. To see the suppressed startup
warnings while debugging:

```bash
python3 src/deepstream_yolo_parser_app.py --show-gst-scan-warnings
```

## Export environment

To install or refresh only the YOLO export dependencies:

```bash
scripts/setup_yolo_export_env.sh
```

The export scripts intentionally avoid relying on a bare `python` command inside
the container. They use `.venv-yolo/bin/python3` and `.venv-yolo/bin/yolo`
explicitly.

## Generated artifacts

Large runtime artifacts are intentionally ignored by Git:

- `.venv-yolo/`
- `external/`
- `lib/*.so`
- `models/*`
- `configs/generated/`
- `outputs/`
- `__pycache__/`

`streams/` is also ignored, but it is treated as user-provided local media, not
generated output. Put input videos there; cleanup scripts do not remove it.

Preview cleanup targets:

```bash
scripts/clean_artifacts.sh
```

Remove generated artifacts:

```bash
scripts/clean_artifacts.sh --force
```

Model artifacts are preserved unless `--include-models` is provided. There is
intentionally no cleanup flag for `streams/`.

## Troubleshooting

If `sudo` reports `unable to resolve host docker`, exit the current container and
start a fresh one with `scripts/run.sh` so Docker Compose applies the `extra_hosts`
entry.

If the export environment is broken, rerun `scripts/setup_yolo_export_env.sh`.
The script recreates `.venv-yolo/` only when its Python interpreter is missing.
