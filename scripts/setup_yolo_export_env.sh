#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR=".venv-yolo"

if [ ! -x "$VENV_DIR/bin/python3" ]; then
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python3"

"$PYTHON_BIN" -m pip install --upgrade 'pip' 'setuptools<82' wheel

"$PYTHON_BIN" -m pip install -r requirements/yolo-export.txt
