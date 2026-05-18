#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv-yolo/bin/activate

python external/DeepStream-Yolo/utils/export_yolo11.py \
  -w /home/user/deepstream-work/models/yolo11n.pt \
  -s 640 \
  --opset 18 \
  --simplify

rm -f models/*.engine