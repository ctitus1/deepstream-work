#!/usr/bin/env bash
# Run the full ROS publishing stack with one command.
#
# Starts, in dependency order: the RTSP server, the ROS Humble publisher bridge,
# Foxglove Bridge, an optional bag recorder, and the DeepStream source. Every
# container is labelled and tracked, so any exit -- Ctrl-C, SIGTERM, a crashed
# component, or a closed terminal -- takes the whole stack down with it.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh

usage() {
    cat <<'EOF'
Usage:
  scripts/ros.sh [options] [-- source-args...]

Starts the RTSP server, ROS Humble publisher, Foxglove Bridge, and the
DeepStream ROS source. Ctrl-C stops and removes everything it started.

Options:
  --video PATH          Video to serve. Default: the video in streams/
  --rtsp-port PORT      RTSP port. Default: 8555
  --foxglove-port PORT  Foxglove websocket port. Default: 8765
  --bag                 Also record all ROS topics to MCAP under outputs/rosbags/.
  --build               Force a rebuild of the ROS profile images. They are
                        built automatically when their inputs change.
  --setup               Run scripts/setup.sh first.
  -h, --help            Show this help.

Environment:
  CDCL_ROS_WS      Host workspace holding cdcl_umd_msgs. Default: /home/user/ros2_ws
  ROS_DOMAIN_ID    ROS domain. Default: 0
  BAG_OUTPUT       Bag path. Default: outputs/rosbags/deepstream-<run-id>

Anything after -- goes to src/ros_source.py.

Examples:
  scripts/ros.sh
  scripts/ros.sh --bag
  scripts/ros.sh --video streams/other.mp4
  scripts/ros.sh -- --jpeg-quality 90
EOF
}

VIDEO=""
RTSP_PORT="${RTSP_PORT:-8555}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"
BAG=0
BUILD=0
RUN_SETUP=0
SOURCE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)          VIDEO="$2"; shift 2 ;;
        --rtsp-port)      RTSP_PORT="$2"; shift 2 ;;
        --foxglove-port)  FOXGLOVE_PORT="$2"; shift 2 ;;
        --bag)            BAG=1; shift ;;
        --build)          BUILD=1; shift ;;
        --setup)          RUN_SETUP=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        --)               shift; SOURCE_ARGS+=("$@"); break ;;
        *)                SOURCE_ARGS+=("$1"); shift ;;
    esac
done

in_deepstream_container && die "run this from the host: it orchestrates several containers."

require_docker
export_host_ids

[ "$RUN_SETUP" -eq 1 ] && scripts/setup.sh

# Kept project-relative: every container below resolves it against the mount.
VIDEO="$(project_relative "${VIDEO:-$(default_media)}")"
[ -f "$VIDEO" ] || die "video not found: $VIDEO
streams/ holds: $(ls streams/ 2>/dev/null | tr '\n' ' ')"

MOUNT="$(basename "${VIDEO%.*}")"
RTSP_URL="rtsp://127.0.0.1:${RTSP_PORT}/${MOUNT}"
# One id per run, shared with the container label so names and label agree.
BAG_OUTPUT="${BAG_OUTPUT:-outputs/rosbags/deepstream-${DSW_RUN_ID}}"

# The ROS workspace supplies the custom message types; without it the bridge
# starts and then fails to import cdcl_umd_msgs several seconds later. Resolved
# and exported here so the compose services inherit it.
require_ros_workspace >/dev/null

install_lifecycle_traps
sweep_stale_containers

# The ROS images are built here rather than in setup.sh: only this entrypoint
# needs them, and the ROS Humble base costs a gigabyte that a parser-only user
# should never pay. Stamped on the same inputs the image is built from, so a
# change to the ROS Dockerfile or to the UID mapping in compose rebuilds without
# anyone having to remember --build. The image check is separate from the stamp
# because a stamp cannot see that someone deleted the image behind its back.
ensure_ros_images() {
    local signature
    signature="uid=${HOST_UID}:${HOST_GID} $(hash_files docker/Dockerfile.ros-humble docker-compose.yml)"

    if [ "$BUILD" -eq 0 ] && stamp_valid ros-image "$signature" \
        && docker image inspect deepstream-work:ros-humble >/dev/null 2>&1; then
        skip "ROS profile images"
        return 0
    fi

    step "Building ROS profile images"
    docker compose --profile ros build
    stamp_write ros-image "$signature"
}

ensure_ros_images

# Fail before starting anything if a port is taken; these are all host-network
# services, so a leftover from another run would bind silently in the wrong place.
require_port_free "RTSP server" "$RTSP_PORT"
require_port_free "ROS image endpoint" 5609
require_port_free "ROS detect endpoint" 5610
require_port_free "ROS assess endpoint" 5611
require_port_free "Foxglove Bridge" "$FOXGLOVE_PORT"

start_service() {
    local name="$1" label="$2"; shift 2
    log "  starting ${label}..."
    track_container "$name"
    docker compose --profile ros run --rm -T \
        --name "$name" --label "$DSW_LABEL=$DSW_RUN_ID" "$@" &
    track_pid "$!"
}

step "Starting the ROS stack"
log "  RTSP:      $RTSP_URL"
log "  Foxglove:  ws://localhost:${FOXGLOVE_PORT}"
log "  ROS msgs:  $CDCL_ROS_WS"
# Worth showing when they differ: it is the non-obvious part of this setup.
[ "$CDCL_ROS_WS_MOUNT" != "$CDCL_ROS_WS" ] \
    && log "             mounted in-container at $CDCL_ROS_WS_MOUNT"
[ "$BAG" -eq 1 ] && log "  Bag:       $BAG_OUTPUT"
log ""

start_service "deepstream-rtsp-${DSW_RUN_ID}" "RTSP server" \
    deepstream-dev \
    python3 scripts/rtsp_server.py "$VIDEO" --port "$RTSP_PORT" --mount "$MOUNT"
wait_for_port "RTSP server" "$RTSP_PORT" 60

start_service "ros-humble-publisher-${DSW_RUN_ID}" "ROS publisher" \
    ros-humble-publisher
wait_for_port "ROS image endpoint" 5609 60
wait_for_port "ROS detect endpoint" 5610 60
wait_for_port "ROS assess endpoint" 5611 60

start_service "ros-foxglove-bridge-${DSW_RUN_ID}" "Foxglove Bridge" \
    -e FOXGLOVE_PORT="$FOXGLOVE_PORT" ros-foxglove-bridge
wait_for_port "Foxglove Bridge" "$FOXGLOVE_PORT" 60

if [ "$BAG" -eq 1 ]; then
    mkdir -p "$(dirname "$BAG_OUTPUT")"
    start_service "rosbag-${DSW_RUN_ID}" "bag recorder" \
        ros-humble-publisher scripts/ros_service.sh bag "$BAG_OUTPUT"
    # ros2 bag needs to have discovered the topics before frames start flowing.
    sleep 2
fi

start_service "deepstream-ros-source-${DSW_RUN_ID}" "DeepStream source" \
    deepstream-ros-source python3 src/ros_source.py \
    --stream "$RTSP_URL" ${SOURCE_ARGS[@]+"${SOURCE_ARGS[@]}"}

log ""
step "Stack is up. Connect Foxglove to ws://localhost:${FOXGLOVE_PORT}"
log "  /uas4/image"
log "  /uas4/target_detections"
log "  /casualty_image/compressed/annotated"
log ""
log "Press Ctrl-C to stop everything."

# Return as soon as any component exits, so one crashed piece tears down the
# rest instead of leaving a half-running stack that looks healthy.
set +e
wait -n ${DSW_PIDS[@]+"${DSW_PIDS[@]}"}
status=$?
set -e

log ""
report_exit "$status" "a stack component"
warn "shutting the rest of the stack down."
exit "$status"
