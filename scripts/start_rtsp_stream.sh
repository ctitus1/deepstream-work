#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-streams/dtc-d4-trimmed.mp4}"
PORT="${RTSP_PORT:-8555}"
MOUNT="${RTSP_MOUNT:-dtc-d4-trimmed}"

python3 scripts/rtsp_video_server.py "$VIDEO" --port "$PORT" --mount "$MOUNT"
