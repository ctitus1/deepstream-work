#!/usr/bin/env bash
# The three entrypoints of the ROS Humble container, in one dispatcher.
#
#   bridge    publish DeepStream frames onto the ROS graph
#   foxglove  serve that graph to Foxglove Studio over a websocket
#   bag       record the whole graph to an MCAP bag
#
# They live together because they share the part that is easy to get wrong:
# every one of them must source ROS Humble and the mounted cdcl_umd_msgs
# workspace before it can name a custom message type, and they differ only in
# what they exec afterwards.
#
# Launched by scripts/ros.sh and by the compose service commands; also runnable
# by hand inside the container.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    cat <<'EOF'
Usage:
  scripts/ros_service.sh bridge [args...]     -> src/ros_bridge.py
  scripts/ros_service.sh foxglove             -> foxglove_bridge, $FOXGLOVE_PORT
  scripts/ros_service.sh bag [OUTPUT]         -> ros2 bag record -s mcap -a
EOF
}

# ROS 2's setup.bash and colcon's local_setup files read unset variables
# (AMENT_TRACE_SETUP_FILES, COLCON_TRACE, _colcon_prefix_*), so `set -u` has to
# be lifted around each source or this aborts before reaching any entrypoint.
if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    set +u; source /opt/ros/humble/setup.bash; set -u
fi

# Absent on a workspace that was never built; the bridge then fails on `import
# cdcl_umd_msgs`, which scripts/ros.sh already guards against on the host side.
if [[ -n "${CDCL_ROS_SETUP:-}" && -f "$CDCL_ROS_SETUP" ]]; then
    # shellcheck disable=SC1090
    set +u; source "$CDCL_ROS_SETUP"; set -u
fi

COMMAND="${1:-}"
[ "$#" -gt 0 ] && shift

# Every branch execs, so the real process replaces this shell and receives
# docker stop's SIGTERM directly rather than through a bash that would not
# forward it -- which is what lets a recording close its bag cleanly.
case "$COMMAND" in
    bridge)
        exec python3 src/ros_bridge.py "$@"
        ;;
    foxglove)
        exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml \
            address:=0.0.0.0 \
            port:="${FOXGLOVE_PORT:-8765}"
        ;;
    bag)
        OUTPUT="${1:-outputs/rosbags/deepstream-$(date +%Y%m%d%H%M%S)}"
        mkdir -p "$(dirname "$OUTPUT")"
        exec ros2 bag record -s mcap -a -o "$OUTPUT"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        [ -n "$COMMAND" ] && printf 'error: unknown service %s\n\n' "$COMMAND" >&2
        usage >&2
        exit 1
        ;;
esac
