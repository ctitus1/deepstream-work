#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-models/injury.pt}"
BATCH_SIZE="${2:-8}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"

cd "$ROOT_DIR"
mkdir -p models configs/generated outputs

if [ ! -f "$MODEL" ]; then
    echo "Missing injury model checkpoint: $MODEL"
    exit 1
fi

has_export_deps() {
    "$1" - <<'PY' >/dev/null 2>&1
import clip
import onnx
import torch
PY
}

if [ -z "$PYTHON_BIN" ]; then
    if has_export_deps python3; then
        PYTHON_BIN="python3"
    else
        scripts/setup_yolo_export_env.sh
        PYTHON_BIN="$ROOT_DIR/.venv-yolo/bin/python3"
    fi
fi

if ! has_export_deps "$PYTHON_BIN"
then
    echo "Missing injury export dependencies in: $PYTHON_BIN"
    echo "Need Python packages: torch, clip, onnx"
    echo "Use scripts/setup_yolo_export_env.sh, or set PYTHON_BIN."
    exit 1
fi

PYTHONPATH=src "$PYTHON_BIN" -m deepstream_yolo.injury export \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE"
