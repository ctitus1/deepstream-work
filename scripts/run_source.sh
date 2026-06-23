#!/usr/bin/env bash
# Run the DeepStream side of the ROS pipeline.
#
# This wrapper keeps Docker Compose commands short: it forwards all arguments to
# the Python app that forks raw, detection, and assessment frame streams to TCP.
set -euo pipefail

python3 src/ros_source.py "$@"
