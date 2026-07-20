#!/usr/bin/env bash
# Shared helpers for the three top-level entrypoints: setup.sh, parser.sh, ros.sh.
#
# Three concerns live here because all three commands need them and they are
# easy to get subtly wrong per-script:
#
#   1. Where am I?      Host or DeepStream container; the commands work from both.
#   2. Is this done?    Stamp files that make expensive setup steps skippable.
#   3. Clean teardown.  Traps that leave no orphaned container or process behind,
#                       including after a previous run was hard-killed.
#
# Source it, do not execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

# Guard against double-sourcing, which would reset the lifecycle arrays
# mid-run and orphan whatever was already tracked.
[ -n "${DSW_COMMON_SOURCED:-}" ] && return 0
DSW_COMMON_SOURCED=1

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_DIR

STATE_DIR="${DSW_STATE_DIR:-$PROJECT_DIR/.setup-state}"

# Every container these scripts start is labelled with the id of the run that
# started it. That gives cleanup a way to find its own containers by property
# rather than by name, which matters because `docker compose run` creates the
# container asynchronously: a stop issued while it is still being created is a
# no-op against a container that then comes up anyway. Sweeping the label after
# the stop closes that window.
DSW_LABEL="com.deepstream-work.managed"
DSW_RUN_ID="${DSW_RUN_ID:-$$-$(date +%s)}"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'; C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RESET=$'\033[0m'
else
    C_BOLD=''; C_DIM=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_RESET=''
fi

log()  { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s%s%s\n' "$C_GREEN" "$C_RESET" "$C_BOLD" "$*" "$C_RESET"; }
skip() { printf '%s==>%s %s %s(cached)%s\n' "$C_DIM" "$C_RESET" "$*" "$C_DIM" "$C_RESET"; }
warn() { printf '%swarning:%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Where am I?
# ---------------------------------------------------------------------------

in_deepstream_container() {
    [ -d /opt/nvidia/deepstream ]
}

require_docker() {
    command -v docker >/dev/null 2>&1 \
        || die "docker is not installed or not on PATH."
    docker info >/dev/null 2>&1 \
        || die "cannot talk to the Docker daemon. Is it running, and are you in the 'docker' group?"
}

# Compose needs the host UID/GID to build an image whose user matches the
# caller, otherwise bind-mounted files come back root-owned.
export_host_ids() {
    HOST_UID="$(id -u)"; export HOST_UID
    HOST_GID="$(id -g)"; export HOST_GID
}

# ---------------------------------------------------------------------------
# Media paths
# ---------------------------------------------------------------------------

# Anything handed to a container has to be project-relative. The repo is
# bind-mounted at a different absolute path inside (/workspace/deepstream-work
# by default), so a host absolute path simply does not exist there -- and the
# failure surfaces several elements deep in GStreamer rather than here.
project_relative() {
    local path="$1"
    case "$path" in
        "$PROJECT_DIR"/*) printf '%s' "${path#"$PROJECT_DIR"/}" ;;
        *)                printf '%s' "$path" ;;
    esac
}

# Project-relative default media, so the same string works on both sides of the
# container boundary. Mirrors paths.default_media().
default_media() {
    project_relative "$(PYTHONPATH="$PROJECT_DIR/src" python3 -c \
        'from deepstream_yolo.paths import DEFAULT_MEDIA; print(DEFAULT_MEDIA)')"
}

# paths.py owns the default model; the shell reads it rather than repeating it.
default_model() {
    PYTHONPATH="$PROJECT_DIR/src" python3 -c \
        'from deepstream_yolo.paths import DEFAULT_MODEL; print(DEFAULT_MODEL)'
}

# ---------------------------------------------------------------------------
# The ROS workspace supplying cdcl_umd_msgs
# ---------------------------------------------------------------------------

# A workspace is only usable once it has been colcon-built and actually contains
# the custom messages this repo publishes. Checking for the built package rather
# than just the directory matters: a workspace that was cloned but never built
# passes a bare -d test and then fails inside the bridge container as an opaque
# import error seconds later.
ros_workspace_is_valid() {
    local ws="${1:-}"
    [ -n "$ws" ] || return 1
    [ -f "$ws/install/setup.bash" ] || return 1
    [ -d "$ws/install/cdcl_umd_msgs" ] || return 1
}

# Locate the workspace without requiring the caller to know where it lives.
# ROS 2 Humble has no Ubuntu 24.04 packages, so on Noble hosts the workspace is
# built inside a Humble container and left wherever that project happens to sit
# -- rarely the historical ~/ros2_ws. An explicit CDCL_ROS_WS always wins.
find_ros_workspace() {
    if [ -n "${CDCL_ROS_WS:-}" ]; then
        printf '%s' "$CDCL_ROS_WS"
        return 0
    fi

    local candidate
    for candidate in "$HOME/ros2_ws" "$HOME"/*/ros2_ws "$HOME"/*/*/ros2_ws; do
        if ros_workspace_is_valid "$candidate"; then
            printf '%s' "$candidate"
            return 0
        fi
    done

    return 1
}

# Recover the absolute path a workspace was built at.
#
# `colcon build --symlink-install` bakes absolute paths into install/: the
# local_setup files are symlinks into <root>/build/... Mounting such a workspace
# at any other path leaves those symlinks dangling, and the failure is nasty --
# sourcing setup.bash prints "not found" for each one, exits 0 anyway, and sets
# PYTHONPATH to nothing, so the bridge dies on `import cdcl_umd_msgs` with no
# hint that the mount path was the cause. Reproducing the original path inside
# the container makes every symlink resolve.
ros_workspace_build_prefix() {
    local ws="${1:-}" link target
    link="$(find "$ws/install" -maxdepth 4 -name 'local_setup.bash' -type l -print -quit 2>/dev/null)"
    [ -n "$link" ] || return 1

    target="$(readlink "$link")"
    case "$target" in
        */build/*) printf '%s' "${target%%/build/*}" ;;
        *) return 1 ;;
    esac
}

# Path the workspace must appear at inside the container. For a workspace built
# in place this is just its own path; for one built elsewhere (the norm on hosts
# with no ROS, where it is built in a container) it is that container's path.
ros_workspace_mount() {
    local prefix
    prefix="$(ros_workspace_build_prefix "${1:-}" || true)"
    printf '%s' "${prefix:-/ros2_ws}"
}

# Resolve, validate, and export. Dies with something actionable rather than
# letting the bridge fail on the import.
require_ros_workspace() {
    local ws
    ws="$(find_ros_workspace || true)"

    if [ -z "$ws" ]; then
        die "no built ROS workspace with cdcl_umd_msgs found.
Looked in: ~/ros2_ws, ~/*/ros2_ws, ~/*/*/ros2_ws
It must be colcon-built (install/setup.bash and install/cdcl_umd_msgs present).
Point at it explicitly with: CDCL_ROS_WS=/path/to/ros2_ws"
    fi

    if ! ros_workspace_is_valid "$ws"; then
        die "CDCL_ROS_WS=$ws is not a built workspace.
Expected $ws/install/setup.bash and $ws/install/cdcl_umd_msgs.
Build it with colcon inside a ROS Humble container, then retry."
    fi

    CDCL_ROS_WS="$ws"
    # Mount point and setup path are derived together so the compose services
    # cannot disagree about where the workspace lives inside the container.
    CDCL_ROS_WS_MOUNT="${CDCL_ROS_WS_MOUNT:-$(ros_workspace_mount "$ws")}"
    CDCL_ROS_SETUP="${CDCL_ROS_SETUP:-$CDCL_ROS_WS_MOUNT/install/setup.bash}"
    export CDCL_ROS_WS CDCL_ROS_WS_MOUNT CDCL_ROS_SETUP

    printf '%s' "$ws"
}

# ---------------------------------------------------------------------------
# The YOLO export virtualenv
# ---------------------------------------------------------------------------

python_tag() {
    python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])'
}

# Mirrors paths.yolo_venv_dir(). Keep the two in step: model_cache shells out to
# the interpreter this resolves to. The path is version-keyed because the host
# and the container see the same bind-mounted directory but run different
# Pythons, and a venv is only usable by the version that built it.
yolo_venv_dir() {
    local tag; tag="$(python_tag)"
    if [ -d "$PROJECT_DIR/.venv-yolo/lib/python${tag}" ]; then
        printf '%s/.venv-yolo' "$PROJECT_DIR"
    else
        printf '%s/.venv-yolo-%s' "$PROJECT_DIR" "$tag"
    fi
}

# ---------------------------------------------------------------------------
# Stamps: skip expensive work whose inputs have not changed.
#
# A stamp stores a signature describing the step's real inputs. The step is
# skipped only when the stored signature matches the current one AND the
# artifact it was supposed to produce is still on disk, so deleting an output
# correctly forces a rebuild.
# ---------------------------------------------------------------------------

stamp_file() { printf '%s/%s.stamp' "$STATE_DIR" "$1"; }

stamp_valid() {
    local name="$1" signature="$2"
    shift 2

    local file; file="$(stamp_file "$name")"
    [ -f "$file" ] || return 1
    [ "$(cat "$file" 2>/dev/null)" = "$signature" ] || return 1

    # Remaining args are artifacts that must still exist.
    local artifact
    for artifact in "$@"; do
        [ -e "$artifact" ] || return 1
    done
    return 0
}

stamp_write() {
    local name="$1" signature="$2"
    mkdir -p "$STATE_DIR"
    printf '%s' "$signature" > "$(stamp_file "$name")"
}

# Hash whatever files exist, so a signature can track a step's real inputs.
hash_files() {
    local path
    for path in "$@"; do
        [ -f "$path" ] && cat "$path"
    done | sha256sum | cut -d' ' -f1
}

# ---------------------------------------------------------------------------
# Lifecycle: never leave a container or child process behind.
# ---------------------------------------------------------------------------

DSW_CONTAINERS=()
DSW_PIDS=()
DSW_CLEANED=0

track_container() { DSW_CONTAINERS+=("$1"); }
track_pid()       { DSW_PIDS+=("$1"); }

# Remove containers this project started that outlived their script, which is
# what a SIGKILL or a host reboot leaves behind. Without this, the port checks
# fail on every subsequent run with no hint about why.
#
# Deliberately limited to containers that are no longer running: a *running*
# managed container may belong to a concurrent session, and killing someone
# else's stack would be a far worse failure than the port conflict that a real
# collision reports a moment later.
sweep_stale_containers() {
    command -v docker >/dev/null 2>&1 || return 0

    local stale
    stale="$(docker ps -aq --filter "label=$DSW_LABEL" --filter "status=exited" \
             --filter "status=created" --filter "status=dead" 2>/dev/null || true)"
    [ -z "$stale" ] && return 0

    warn "removing containers left over from a previous run"
    # shellcheck disable=SC2086
    docker rm -f $stale >/dev/null 2>&1 || true
}

# Backstop for the create/stop race described at DSW_LABEL: catches containers
# this run started that the name-based stop above missed.
sweep_own_containers() {
    command -v docker >/dev/null 2>&1 || return 0

    local mine
    mine="$(docker ps -aq --filter "label=$DSW_LABEL=$DSW_RUN_ID" 2>/dev/null || true)"
    [ -z "$mine" ] && return 0

    # shellcheck disable=SC2086
    docker rm -f $mine >/dev/null 2>&1 || true
}

# Ask nicely, then insist. The grace period matters: the parser finalizes its
# mp4 (writes the moov atom) on SIGINT/SIGTERM, and killing it sooner leaves an
# unplayable file behind.
dsw_cleanup() {
    local status=$?

    # EXIT fires after INT/TERM have already run this; do the work once.
    [ "$DSW_CLEANED" -eq 1 ] && exit "$status"
    DSW_CLEANED=1
    trap - EXIT INT TERM HUP

    if [ ${#DSW_CONTAINERS[@]} -eq 0 ] && [ ${#DSW_PIDS[@]} -eq 0 ]; then
        exit "$status"
    fi

    printf '\n'
    step "Shutting down..."

    local pid
    for pid in ${DSW_PIDS[@]+"${DSW_PIDS[@]}"}; do
        kill -INT "$pid" 2>/dev/null || true
    done

    if [ ${#DSW_CONTAINERS[@]} -gt 0 ]; then
        # docker stop sends SIGTERM first, which the apps now handle, so an
        # in-progress recording is finalized rather than truncated.
        docker stop --time "${DSW_STOP_TIMEOUT:-15}" \
            ${DSW_CONTAINERS[@]+"${DSW_CONTAINERS[@]}"} >/dev/null 2>&1 || true
        docker rm -f ${DSW_CONTAINERS[@]+"${DSW_CONTAINERS[@]}"} >/dev/null 2>&1 || true
    fi

    # Give the signalled children a moment to finish flushing, then insist.
    local waited=0
    while [ "$waited" -lt "${DSW_STOP_TIMEOUT:-15}" ]; do
        local alive=0
        for pid in ${DSW_PIDS[@]+"${DSW_PIDS[@]}"}; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        [ "$alive" -eq 0 ] && break
        sleep 1
        waited=$((waited + 1))
    done

    for pid in ${DSW_PIDS[@]+"${DSW_PIDS[@]}"}; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    wait >/dev/null 2>&1 || true

    # Last, once the compose clients are gone and can no longer create anything.
    sweep_own_containers

    log "Stopped."
    exit "$status"
}

# HUP is included so closing the terminal tears the stack down too.
install_lifecycle_traps() {
    trap dsw_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
}

# ---------------------------------------------------------------------------
# Exit reporting
# ---------------------------------------------------------------------------

# Explain a non-zero exit, especially one caused by a signal.
#
# A process killed by a signal exits as 128+N and, in the SIGSEGV case, has
# printed nothing at all -- the crash takes it down before any handler or
# buffered output escapes. That is the single hardest failure to diagnose from
# the outside, so name it rather than letting the caller show a bare "Stopped."
report_exit() {
    local status="$1" what="$2"

    [ "$status" -eq 0 ] && return 0

    if [ "$status" -le 128 ]; then
        warn "$what exited with status ${status}."
        return 0
    fi

    local signum=$((status - 128))
    # INT and TERM are how these scripts stop things on purpose.
    case "$signum" in
        2|15) return 0 ;;
    esac

    local name; name="$(kill -l "$signum" 2>/dev/null || echo "$signum")"
    warn "$what was killed by SIG${name} (exit ${status})."

    if [ "$signum" -eq 11 ]; then
        log "  A segfault inside a GStreamer or GIO plugin leaves no message of its own."
        log "  To see where it died:"
        log "    docker compose run --rm -T deepstream-dev \\"
        log "      gdb -batch -ex run -ex 'bt 25' --args python3 src/parser_app.py --stream <url>"
    fi
}

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

port_is_open() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

wait_for_port() {
    local name="$1" port="$2" timeout="${3:-60}" elapsed=0

    while [ "$elapsed" -lt "$timeout" ]; do
        port_is_open "$port" && { log "  ${name} is up on port ${port}."; return 0; }

        # A container that died will never open its port; fail immediately with
        # the real reason rather than after the full timeout.
        local pid
        for pid in ${DSW_PIDS[@]+"${DSW_PIDS[@]}"}; do
            kill -0 "$pid" 2>/dev/null || {
                die "${name} exited before opening port ${port}. See the output above."
            }
        done

        sleep 1
        elapsed=$((elapsed + 1))
    done

    die "timed out after ${timeout}s waiting for ${name} on port ${port}."
}

require_port_free() {
    local name="$1" port="$2"
    port_is_open "$port" && die \
        "${name} port ${port} is already in use. Stop that service, or pick another port."
    return 0
}
