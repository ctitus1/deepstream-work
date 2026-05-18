#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv-yolo

source .venv-yolo/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  ultralytics \
  onnx \
  onnxsim \
  onnxruntime \
  onnxscript \
  open_clip_torch \
  timm \
  einops
