#!/usr/bin/env bash
# Start the ROS Humble bridge.
#
# The bridge needs both ROS Humble and the mounted cdcl_umd_msgs workspace
# sourced before it can publish the custom message types used by Foxglove.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/ros_env.sh"

python3 src/ros_bridge.py "$@"
