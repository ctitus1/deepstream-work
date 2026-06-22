#!/usr/bin/env bash
set -euo pipefail

xhost +local:docker >/dev/null
cleanup() {
  xhost -local:docker >/dev/null 2>&1 || true
}
trap cleanup EXIT

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

docker compose run --rm deepstream-dev
