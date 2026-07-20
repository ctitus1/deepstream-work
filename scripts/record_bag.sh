#!/usr/bin/env bash
# Record the active ROS graph to an MCAP bag.
#
# Launched by ros.sh when --bag is supplied; also runnable directly inside the
# ROS Humble container.
set -euo pipefail

OUTPUT="${1:-outputs/rosbags/deepstream-$(date +%Y%m%d%H%M%S)}"

source "$(dirname "${BASH_SOURCE[0]}")/lib/ros_env.sh"

mkdir -p "$(dirname "$OUTPUT")"
exec ros2 bag record -s mcap -a -o "$OUTPUT"
