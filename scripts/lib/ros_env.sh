#!/usr/bin/env bash
# Source ROS Humble and the mounted cdcl_umd_msgs workspace.
#
# Source it, do not execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/ros_env.sh"
#
# Every caller runs under `set -euo pipefail`. ROS 2's setup.bash and colcon's
# local_setup files read unset variables (AMENT_TRACE_SETUP_FILES, COLCON_TRACE,
# _colcon_prefix_*), so `set -u` has to be lifted around each source or the
# script aborts before its entrypoint is ever reached.

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
