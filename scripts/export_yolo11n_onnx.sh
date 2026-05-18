#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv-yolo/bin/activate

mkdir -p models

yolo export \
  model=yolo11n.pt \
  format=onnx \
  opset=17 \
  simplify=True \
  imgsz=640 \
  dynamic=False

mv -f yolo11n.pt models/
mv -f yolo11n.onnx models/

echo "Created:"
ls -lh models/yolo11n.pt models/yolo11n.onnx
