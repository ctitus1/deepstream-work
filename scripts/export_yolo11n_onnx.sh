#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LONG_SIDE="${1:-640}"
STREAM="${2:-streams/dtc-d3-trimmed-short.mp4}"

exec "$SCRIPT_DIR/setup_and_export_yolo.sh" yolo11n.pt "$LONG_SIDE" "$STREAM"
