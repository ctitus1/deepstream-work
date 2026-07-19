#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY_SITE="$PWD/.venv-yolo/lib/python3.12/site-packages"
export PYTHONPATH="$PY_SITE:${PYTHONPATH:-}"

exec /usr/bin/python3 "$@"
