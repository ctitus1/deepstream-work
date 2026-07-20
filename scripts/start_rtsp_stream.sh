#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# With no arguments the server picks the video and derives the mount name from
# it, matching the client defaults in src/deepstream_yolo/paths.py. Passing
# RTSP_PORT/RTSP_MOUNT still overrides either one.
ARGS=()
[ "$#" -gt 0 ] && ARGS+=("$1")
[ -n "${RTSP_PORT:-}" ] && ARGS+=(--port "$RTSP_PORT")
[ -n "${RTSP_MOUNT:-}" ] && ARGS+=(--mount "$RTSP_MOUNT")

exec python3 scripts/rtsp_video_server.py ${ARGS[@]+"${ARGS[@]}"}
