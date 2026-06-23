#!/usr/bin/env bash
# Start the ROS Humble bridge.
#
# The bridge needs both ROS Humble and the mounted cdcl_umd_msgs workspace
# sourced before it can publish the custom message types used by Foxglove.
set -euo pipefail

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

python3 src/ros_bridge.py "$@"
