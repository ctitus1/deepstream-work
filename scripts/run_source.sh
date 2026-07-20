#!/usr/bin/env bash
# Run the DeepStream side of the ROS pipeline: it forks raw, detection, and
# assessment frame streams to the bridge's TCP endpoints.
set -euo pipefail

python3 src/ros_source.py "$@"
