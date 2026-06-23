#!/usr/bin/env bash
# Start the local RTSP + ROS + Foxglove workflow.
#
# The stack runs four pieces with host networking: an RTSP server for the sample
# video, a ROS bridge that publishes messages, Foxglove Bridge for visualization,
# and the DeepStream ROS pub app. Ctrl-C stops every container this script starts.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_stack.sh [options] [-- ros-pub-args...]

Starts the local RTSP video server, ROS Humble publisher, Foxglove Bridge, and
DeepStream ROS pub app. Press Ctrl-C to stop and remove the containers started by
this script.

Options:
  --video PATH          Video served by RTSP. Default: streams/dtc-d4-trimmed.mp4
  --rtsp-port PORT     RTSP server port. Default: 8555
  --rtsp-mount NAME    RTSP mount name. Default: dtc-d4-trimmed
  --foxglove-port PORT Foxglove Bridge websocket port. Default: 8765
  --bag                Record all ROS topics to an MCAP bag under outputs/rosbags.
  --build              Build ROS profile images before starting.
  -h, --help           Show this help.

Environment:
  BAG_OUTPUT           Bag output path. Default: outputs/rosbags/deepstream-<run-id>
  CDCL_ROS_WS          Host ROS workspace for cdcl_umd_msgs. Default: /home/user/ros2_ws
  ROS_DOMAIN_ID        ROS domain ID. Default: 0

Examples:
  scripts/run_stack.sh
  scripts/run_stack.sh --bag
  scripts/run_stack.sh --video streams/demo.mp4 --rtsp-mount demo
  scripts/run_stack.sh -- --rtsp-latency-ms 0 --jpeg-quality 90
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VIDEO="${RTSP_VIDEO:-streams/dtc-d4-trimmed.mp4}"
RTSP_PORT="${RTSP_PORT:-8555}"
RTSP_MOUNT="${RTSP_MOUNT:-dtc-d4-trimmed}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"
BUILD=0
BAG=0
PUB_ARGS=()

# Parse stack options first; anything after "--" is passed to ros_pub.py.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      VIDEO="$2"
      shift 2
      ;;
    --rtsp-port)
      RTSP_PORT="$2"
      shift 2
      ;;
    --rtsp-mount)
      RTSP_MOUNT="$2"
      shift 2
      ;;
    --foxglove-port)
      FOXGLOVE_PORT="$2"
      shift 2
      ;;
    --bag)
      BAG=1
      shift
      ;;
    --build)
      BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PUB_ARGS+=("$@")
      break
      ;;
    *)
      PUB_ARGS+=("$1")
      shift
      ;;
  esac
done

RTSP_MOUNT="${RTSP_MOUNT#/}"
RTSP_MOUNT="${RTSP_MOUNT:-stream}"
RTSP_URL="rtsp://127.0.0.1:${RTSP_PORT}/${RTSP_MOUNT}"
RUN_ID="${ROS_STACK_RUN_ID:-$(date +%Y%m%d%H%M%S)-$$}"
BAG_OUTPUT="${BAG_OUTPUT:-outputs/rosbags/deepstream-${RUN_ID}}"
CONTAINERS=()
PIDS=()

# Track all launched containers/processes so Ctrl-C leaves no stack leftovers.
cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if [[ ${#CONTAINERS[@]} -eq 0 && ${#PIDS[@]} -eq 0 ]]; then
    exit "$status"
  fi

  echo
  echo "Stopping RTSP/ROS/Foxglove stack..."

  if [[ ${#CONTAINERS[@]} -gt 0 ]]; then
    docker stop --time 10 "${CONTAINERS[@]}" >/dev/null 2>&1 || true
    docker rm -f "${CONTAINERS[@]}" >/dev/null 2>&1 || true
  fi

  for pid in "${PIDS[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  wait >/dev/null 2>&1 || true

  echo "Stopped."
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_port() {
  local name="$1"
  local port="$2"
  local timeout="${3:-30}"

  for _ in $(seq 1 "$timeout"); do
    if bash -c ":</dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
      echo "${name} is listening on port ${port}."
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for ${name} on port ${port}." >&2
  return 1
}

require_port_free() {
  local name="$1"
  local port="$2"

  if bash -c ":</dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
    echo "${name} port ${port} is already in use. Stop the existing service or choose another port." >&2
    return 1
  fi
}

start_compose_run() {
  local name="$1"
  local label="$2"
  shift 2

  echo "Starting ${label} (${name})..."
  CONTAINERS+=("$name")
  docker compose --profile ros run --rm -T --name "$name" "$@" &
  PIDS+=("$!")
}

if [[ "$BUILD" -eq 1 ]]; then
  docker compose --profile ros build
fi

# Fail early if a previous run still owns one of the fixed host-network ports.
require_port_free "RTSP server" "$RTSP_PORT"
require_port_free "ROS image publisher endpoint" 5609
require_port_free "ROS detect publisher endpoint" 5610
require_port_free "ROS assess publisher endpoint" 5611
require_port_free "Foxglove Bridge" "$FOXGLOVE_PORT"

echo "RTSP URL: ${RTSP_URL}"
echo "Foxglove: ws://localhost:${FOXGLOVE_PORT}"
if [[ "$BAG" -eq 1 ]]; then
  echo "Bag: ${BAG_OUTPUT}"
fi
echo "Topics:"
echo "  /uas4/image"
echo "  /uas4/target_detections"
echo "  /casualty_image/compressed/annotated"
echo

# Launch in dependency order: stream, ROS publishers, Foxglove, optional bag, then ros_pub.
start_compose_run \
  "deepstream-rtsp-${RUN_ID}" \
  deepstream-dev \
  -e RTSP_PORT="$RTSP_PORT" \
  -e RTSP_MOUNT="$RTSP_MOUNT" \
  deepstream-dev \
  scripts/start_rtsp_stream.sh "$VIDEO"
wait_for_port "RTSP server" "$RTSP_PORT" 30

start_compose_run \
  "ros-humble-publisher-${RUN_ID}" \
  ros-humble-publisher \
  ros-humble-publisher
wait_for_port "ROS image publisher endpoint" 5609 30
wait_for_port "ROS detect publisher endpoint" 5610 30
wait_for_port "ROS assess publisher endpoint" 5611 30

start_compose_run \
  "ros-foxglove-bridge-${RUN_ID}" \
  ros-foxglove-bridge \
  -e FOXGLOVE_PORT="$FOXGLOVE_PORT" \
  ros-foxglove-bridge
wait_for_port "Foxglove Bridge" "$FOXGLOVE_PORT" 30

if [[ "$BAG" -eq 1 ]]; then
  mkdir -p "$(dirname "$BAG_OUTPUT")"
  start_compose_run \
    "rosbag-${RUN_ID}" \
    "rosbag recorder" \
    ros-humble-publisher \
    scripts/record_bag.sh "$BAG_OUTPUT"
  sleep 2
fi

start_compose_run \
  "deepstream-ros-pub-${RUN_ID}" \
  deepstream-ros-pub \
  deepstream-ros-pub \
  scripts/run_pub.sh \
  --stream "$RTSP_URL" \
  "${PUB_ARGS[@]}"

echo
echo "Stack is running. Press Ctrl-C to stop everything cleanly."

set +e
wait -n "${PIDS[@]}"
status=$?
set -e

echo "A stack process exited with status ${status}."
exit "$status"
