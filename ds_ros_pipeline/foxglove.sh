#!/usr/bin/env bash
# Container entrypoint for the ds-ros-foxglove service: serves this pipeline's
# ROS graph to Foxglove Studio over a websocket (DESIGN.md §4 lists what is on
# that graph).
#
# Runs in the existing deepstream-work:ros-humble image rather than the
# DeepStream one: foxglove_bridge is already installed there, and the bridge
# needs nothing from DeepStream (no GPU, no pyds). Layering it onto the 21.6 GB
# DS image instead would rebuild that image for a pure-visualization add-on.
# Cross-container discovery works because both services share the host network
# and host IPC, exactly as the root compose's ROS services already do.
set -eo pipefail

source /opt/ros/humble/setup.bash

# Same stale-segment hazard as run.sh: with ipc:host, Fast DDS segments left in
# the host's /dev/shm by a SIGKILLed predecessor can segfault the next
# participant at construction. Only dead-owner segments are removed.
fastdds shm clean || true

# Unlike run.sh this is a warning, not a fatal error. Without the overlay the
# bridge still serves every standard-message topic (preview, mosaic, vlm,
# status) and every service; it just cannot build schemas for the two
# cdcl_umd_msgs topics. Refusing to start a debugging tool over a partial graph
# would be the worse failure.
if [ -n "${CDCL_ROS_SETUP:-}" ] && [ -f "$CDCL_ROS_SETUP" ]; then
    source "$CDCL_ROS_SETUP"
else
    echo "ds_ros_pipeline/foxglove.sh: CDCL_ROS_SETUP unset or missing" >&2
    echo "  (${CDCL_ROS_SETUP:-<unset>})" >&2
    echo "Continuing without cdcl_umd_msgs: /ds/detections and" >&2
    echo "/ds/assessments will be advertised but cannot be deserialized." >&2
fi

# Stock launch arguments only. /vlm_raw messages (11,059,256 B of raw rgb8)
# sit just above foxglove_bridge's 10 MB send_buffer_limit default, so raising
# that looked necessary -- but an A/B against a stock bridge on a second port
# delivered every raw frame either way, including a 6-deep burst against a
# deliberately non-reading client: the limit caps a per-client *backlog*, and
# on localhost the socket drains faster than one accumulates. The knob is left
# at its default rather than tuned on a guess; a genuinely slow or remote
# viewer that drops raw frames is the case to revisit it, via FOXGLOVE_ARGS:
#
#   FOXGLOVE_ARGS="send_buffer_limit:=67108864" docker compose ... up
exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml \
    address:=0.0.0.0 \
    port:="${FOXGLOVE_PORT:-8765}" \
    ${FOXGLOVE_ARGS:-} \
    "$@"
