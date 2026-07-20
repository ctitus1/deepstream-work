#!/usr/bin/env bash
# Serve the ROS graph to Foxglove Studio over a websocket.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/ros_env.sh"

exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml \
  address:=0.0.0.0 \
  port:="${FOXGLOVE_PORT:-8765}"
