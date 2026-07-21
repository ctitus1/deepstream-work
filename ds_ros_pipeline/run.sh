#!/usr/bin/env bash
# Container entrypoint for the ds-ros-pipeline service (DESIGN.md §2).
# Sources the ROS2 environment plus the cdcl_umd_msgs overlay, puts the
# existing deepstream_yolo helpers on PYTHONPATH, and execs the node.
# (ds_ros_pipeline/ itself lands on sys.path automatically as the script
# directory of ds_node.py -- flat sibling imports need nothing extra.)
set -eo pipefail

# --prebuild (DESIGN.md §11 risk 10): write the batch nvinfer config and warm
# the engine cache, then exit. Needs no ROS environment at all, so it runs
# before the setup sourcing — usable in a bare container without the
# cdcl_umd_msgs workspace mount (e.g. an image-build warmup step).
if [ "${1:-}" = "--prebuild" ]; then
    shift
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 ds_ros_pipeline/ds_node.py --prebuild "$@"
fi

# The image bakes its ROS distro into DS_ROS_DISTRO (Dockerfile ARG
# ROS_DISTRO: humble on the 7.1/jammy base, jazzy on 9.0/noble). Fall back to
# the last-sorted installed distro so hand-built containers still work.
if [ -z "${DS_ROS_DISTRO:-}" ] || [ ! -f "/opt/ros/${DS_ROS_DISTRO}/setup.bash" ]; then
    for candidate in /opt/ros/*/setup.bash; do
        DS_ROS_DISTRO="$(basename "$(dirname "$candidate")")"
    done
fi
source "/opt/ros/${DS_ROS_DISTRO}/setup.bash"

# With ipc:host, Fast DDS shared-memory segments left in the host's /dev/shm
# by a SIGKILLed predecessor (docker kill, §10 test 8) intermittently segfault
# the next participant during rclpy Node construction. `fastdds shm clean`
# removes only dead-owner ports/segments -- liveness is checked via file
# locks, so live participants anywhere on the host are untouched. Best
# effort: never block startup on it.
fastdds shm clean || true

if [ -z "${CDCL_ROS_SETUP:-}" ]; then
    echo "ds_ros_pipeline/run.sh: CDCL_ROS_SETUP is not set." >&2
    echo "It must point at the cdcl_umd_msgs workspace overlay, e.g." >&2
    echo "  /home/user/ros2_ws/install/setup.bash" >&2
    echo "compose.yaml sets it from CDCL_ROS_WS; refusing to start without" >&2
    echo "cdcl_umd_msgs rather than failing later on the first publish." >&2
    exit 1
fi
if [ ! -f "$CDCL_ROS_SETUP" ]; then
    echo "ds_ros_pipeline/run.sh: CDCL_ROS_SETUP=$CDCL_ROS_SETUP does not exist." >&2
    echo "Is the colcon workspace mounted at the path it was built at?" >&2
    echo "(compose.yaml mounts \$CDCL_ROS_WS, default /home/user/ros2_ws," >&2
    echo "read-only at that same path.)" >&2
    exit 1
fi
source "$CDCL_ROS_SETUP"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 ds_ros_pipeline/ds_node.py "$@"
