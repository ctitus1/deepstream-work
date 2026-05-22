#!/usr/bin/env bash
set -euo pipefail

xhost +local:docker >/dev/null

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

docker compose run --rm deepstream-dev
