#!/usr/bin/env bash
# Run the DeepStream parser app: detections and injury assessment in a window.
#
# One command for what used to take three shells. It serves the video over RTSP,
# waits for the server to accept connections, then starts the app against it,
# and tears both down together on exit however that exit happens.
#
# RTSP rather than a plain file because it matches the live pipeline: paced by
# the stream clock, dropping late frames instead of queueing them. Pass a
# --stream of your own to skip the local server entirely.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh

usage() {
    cat <<'EOF'
Usage:
  scripts/parser.sh [options] [-- app-args...]

Serves a video over RTSP and runs the parser app against it. Ctrl-C stops both.

Options:
  --video PATH        Video to serve. Default: the video in streams/
  --stream URL        Use an existing stream and do not start a local server.
  --rtsp-port PORT    Local RTSP port. Default: 8555
  --record [PATH]     Record annotated video; omit PATH to auto-name under outputs/.
  --no-assessment     Detection only, no injury assessment.
  --model NAME        Detector checkpoint. Default: yolo12n.pt
  --long-side N       Inference long side. Default: 640
  --setup             Run scripts/setup.sh first.
  -h, --help          Show this help.

Anything after -- goes to src/parser_app.py; see --help there for the full set.

Examples:
  scripts/parser.sh
  scripts/parser.sh --record
  scripts/parser.sh --video streams/other.mp4
  scripts/parser.sh --stream rtsp://camera.local:8554/live
  scripts/parser.sh -- --show-assessed-only --debug
EOF
}

VIDEO=""
STREAM=""
RTSP_PORT="${RTSP_PORT:-8555}"
RUN_SETUP=0
APP_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)      VIDEO="$2"; shift 2 ;;
        --stream)     STREAM="$2"; shift 2 ;;
        --rtsp-port)  RTSP_PORT="$2"; shift 2 ;;
        --setup)      RUN_SETUP=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; APP_ARGS+=("$@"); break ;;
        # Everything else (--record, --no-assessment, --model, --long-side, ...)
        # is forwarded to parser_app.py in order, which owns those options.
        *)            APP_ARGS+=("$1"); shift ;;
    esac
done

[ "$RUN_SETUP" -eq 1 ] && scripts/setup.sh

# A display app needs an X server; say so now rather than failing inside GStreamer.
[ -n "${DISPLAY:-}" ] || die "DISPLAY is not set. The parser app opens a window;
for a headless check use: python3 validation/smoke_pipeline.py --frames 60"

install_lifecycle_traps
sweep_stale_containers

if [ -n "$STREAM" ]; then
    # Someone else owns the stream; we only run the app.
    step "Using stream: $STREAM"
else
    # Kept project-relative: this string is passed into the container, where the
    # repo lives at a different absolute path.
    VIDEO="$(project_relative "${VIDEO:-$(default_media)}")"
    [ -f "$VIDEO" ] || die "video not found: $VIDEO
streams/ holds: $(ls streams/ 2>/dev/null | tr '\n' ' ')"

    MOUNT="$(basename "${VIDEO%.*}")"
    STREAM="rtsp://127.0.0.1:${RTSP_PORT}/${MOUNT}"

    require_port_free "RTSP server" "$RTSP_PORT"

    step "Serving $VIDEO on $STREAM"
    if in_deepstream_container; then
        RTSP_PORT="$RTSP_PORT" RTSP_MOUNT="$MOUNT" \
            scripts/start_rtsp_stream.sh "$VIDEO" &
        track_pid "$!"
    else
        require_docker
        export_host_ids
        RTSP_CONTAINER="deepstream-rtsp-$$"
        track_container "$RTSP_CONTAINER"
        docker compose run --rm -T \
            --name "$RTSP_CONTAINER" \
            --label "$DSW_LABEL=$DSW_RUN_ID" \
            -e RTSP_PORT="$RTSP_PORT" \
            -e RTSP_MOUNT="$MOUNT" \
            deepstream-dev scripts/start_rtsp_stream.sh "$VIDEO" &
        track_pid "$!"
    fi

    wait_for_port "RTSP server" "$RTSP_PORT" 60
fi

step "Starting the parser app"
log "  quit with q, or Ctrl-C"
log ""

# Run the app in the foreground: it owns the terminal for keyboard controls, and
# its exit status becomes this script's, so a failure is not swallowed. set +e
# around it so an abnormal exit is described before the traps take over.
set +e
if in_deepstream_container; then
    python3 src/parser_app.py --stream "$STREAM" ${APP_ARGS[@]+"${APP_ARGS[@]}"}
    APP_STATUS=$?
else
    # The container needs the host X socket to open a window.
    xhost +local:docker >/dev/null 2>&1 || warn "xhost failed; the window may not open"
    PARSER_CONTAINER="deepstream-parser-$$"
    track_container "$PARSER_CONTAINER"
    docker compose run --rm \
        --name "$PARSER_CONTAINER" \
        --label "$DSW_LABEL=$DSW_RUN_ID" \
        deepstream-dev \
        python3 src/parser_app.py --stream "$STREAM" ${APP_ARGS[@]+"${APP_ARGS[@]}"}
    APP_STATUS=$?
    xhost -local:docker >/dev/null 2>&1 || true
fi
set -e

report_exit "$APP_STATUS" "the parser app"
exit "$APP_STATUS"
