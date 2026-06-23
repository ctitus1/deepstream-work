#!/usr/bin/env bash
# Record the active ROS graph to an MCAP bag.
#
# This is launched by run_stack.sh when --bag is supplied, but it can also be
# run directly inside the ROS Humble container.
set -euo pipefail

OUTPUT="${1:-outputs/rosbags/deepstream-$(date +%Y%m%d%H%M%S)}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

if [[ -n "${CDCL_ROS_SETUP:-}" && -f "$CDCL_ROS_SETUP" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "$CDCL_ROS_SETUP"
  set -u
fi

mkdir -p "$(dirname "$OUTPUT")"
exec ros2 bag record -s mcap -a -o "$OUTPUT"
