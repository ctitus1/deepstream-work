#!/usr/bin/env bash
# One-command setup for this repo.
#
# Runs from the host (building the image and re-entering the container as
# needed) or from inside an already-running DeepStream container. Every stage is
# stamped, so re-running is cheap and only what actually changed is redone:
# pointing --model at a new checkpoint rebuilds that model and nothing else.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh

ALL_STAGES=(image parser env models verify)

# 'env' is deliberately not in the default sequence. It installs torch and the
# nvidia-cu12 wheel set (several GB), and the export path already builds it on
# demand the moment a model actually needs exporting. Running it eagerly would
# make every setup pay for a step most runs never need. Ask for it by name to
# pre-warm: scripts/setup.sh --only env
DEFAULT_STAGES=(image parser models verify)

usage() {
    cat <<'EOF'
Usage:
  scripts/setup.sh [options]

Prepares everything needed to run the pipeline. Safe to re-run: each stage is
skipped when its inputs have not changed.

Stages, in order:
  image    Build the DeepStream docker image          (host only)
  parser   Compile the custom YOLO bbox parser        -> lib/
  models   Export the detector + assessment models    -> models/, configs/generated/
  verify   Headless smoke test; builds the TensorRT engine

On demand only (several GB; the models stage triggers it automatically when a
model actually needs exporting, so it is rarely worth running by hand):
  env      Create the YOLO export virtualenv          -> .venv-yolo-<pyver>/

Options:
  --only STAGE[,STAGE]  Run only these stages.
  --skip STAGE[,STAGE]  Skip these stages.
  --force [STAGE]       Ignore stamps, for every stage or just the named one.
  --model NAME          Detector checkpoint to export. Default: yolo12n.pt
  --long-side N         Inference long side. Default: 640
  --stream PATH         Media used for geometry and verification.
                        Default: the video in streams/
  --no-assessment       Skip the injury assessment model.
  --frames N            Frames for the verify stage. Default: 60
  -h, --help            Show this help.

Examples:
  scripts/setup.sh                              # full setup, or a fast no-op
  scripts/setup.sh --model best.pt              # export one new model
  scripts/setup.sh --only verify                # just re-check the pipeline
  scripts/setup.sh --force parser               # rebuild the parser library
EOF
}

MODEL=""
LONG_SIDE=640
STREAM=""
FRAMES=60
ASSESSMENT=1
FORCE_ALL=0
FORCE_STAGE=""
ONLY=""
SKIP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)      ONLY="$2"; shift 2 ;;
        --skip)      SKIP="$2"; shift 2 ;;
        --force)
            # Optional argument: a bare --force means every stage.
            if [[ -n "${2:-}" && "$2" != -* ]]; then
                FORCE_STAGE="$2"; shift 2
            else
                FORCE_ALL=1; shift
            fi
            ;;
        --model)      MODEL="$2"; shift 2 ;;
        --long-side)  LONG_SIDE="$2"; shift 2 ;;
        --stream)     STREAM="$2"; shift 2 ;;
        --frames)     FRAMES="$2"; shift 2 ;;
        --no-assessment) ASSESSMENT=0; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            die "unknown option: $1 (try --help)" ;;
    esac
done

is_stage() {
    local candidate="$1" known
    for known in "${ALL_STAGES[@]}"; do
        [ "$candidate" = "$known" ] && return 0
    done
    return 1
}

# Validate stage names up front: a typo silently doing nothing is worse than a
# short error, especially for --skip.
validate_stages() {
    local list="$1" label="$2" name
    [ -z "$list" ] && return 0
    for name in ${list//,/ }; do
        is_stage "$name" || die "$label: unknown stage '$name'. Valid: ${ALL_STAGES[*]}"
    done
}

validate_stages "$ONLY" "--only"
validate_stages "$SKIP" "--skip"
[ -n "$FORCE_STAGE" ] && { is_stage "$FORCE_STAGE" \
    || die "--force: unknown stage '$FORCE_STAGE'. Valid: ${ALL_STAGES[*]}"; }

is_default_stage() {
    local candidate="$1" known
    for known in "${DEFAULT_STAGES[@]}"; do
        [ "$candidate" = "$known" ] && return 0
    done
    return 1
}

wants() {
    local stage="$1" name

    if [ -n "$ONLY" ]; then
        for name in ${ONLY//,/ }; do [ "$name" = "$stage" ] && return 0; done
        return 1
    fi

    for name in ${SKIP//,/ }; do [ "$name" = "$stage" ] && return 1; done

    # Naming a non-default stage with --force is a request to run it.
    [ "$FORCE_STAGE" = "$stage" ] && return 0

    is_default_stage "$stage"
}

forced() {
    [ "$FORCE_ALL" -eq 1 ] && return 0
    [ "$FORCE_STAGE" = "$1" ] && return 0
    return 1
}

# Options are rebuilt rather than forwarded verbatim so the inner invocation is
# explicit about what it was asked to do.
inner_args() {
    # The image stage is host-only, so the inner run always skips it -- but the
    # caller's own --skip has to be carried through as well, or asking the host
    # to skip a stage silently ran it inside the container anyway.
    local skip="image"
    [ -n "$SKIP" ] && skip="image,$SKIP"

    local args=(--skip "$skip")
    [ -n "$MODEL" ]        && args+=(--model "$MODEL")
    [ -n "$STREAM" ]       && args+=(--stream "$(project_relative "$STREAM")")
    [ "$ASSESSMENT" -eq 0 ] && args+=(--no-assessment)
    [ "$FORCE_ALL" -eq 1 ] && args+=(--force)
    [ -n "$FORCE_STAGE" ]  && args+=(--force "$FORCE_STAGE")
    [ -n "$ONLY" ]         && args+=(--only "$ONLY")
    args+=(--long-side "$LONG_SIDE" --frames "$FRAMES")
    printf '%s\n' "${args[@]}"
}

# ---------------------------------------------------------------------------
# Stage: image (host only)
# ---------------------------------------------------------------------------

setup_image() {
    wants image || return 0
    in_deepstream_container && return 0

    require_docker
    export_host_ids

    local ds_version="${DS_VERSION:-7.1}"
    local signature; signature="ds=${ds_version} $(hash_files docker/Dockerfile docker-compose.yml)"

    # docker's own layer cache makes a rebuild cheap, but not free: skipping the
    # call entirely keeps a warm setup at roughly zero cost.
    if ! forced image && stamp_valid docker-image "$signature"; then
        skip "DeepStream image (deepstream-work:${ds_version})"
        return 0
    fi

    step "Building the DeepStream image (deepstream-work:${ds_version})"
    docker compose build
    stamp_write docker-image "$signature"
}

# ---------------------------------------------------------------------------
# Stages that need the DeepStream runtime
# ---------------------------------------------------------------------------

setup_parser() {
    wants parser || return 0
    if forced parser; then
        scripts/build_yolo_parser.sh --force
    else
        scripts/build_yolo_parser.sh
    fi
}

setup_env() {
    wants env || return 0
    if forced env; then
        scripts/setup_yolo_export_env.sh --force
    else
        scripts/setup_yolo_export_env.sh
    fi
}

setup_models() {
    wants models || return 0

    # The export cache lives in models/*.meta.json rather than in a stamp, since
    # it is keyed by model, resolution, and source geometry all at once.
    # --force invalidates it by removing this model's tagged artifacts.
    if forced models; then
        # Read the default from paths.py rather than repeating it here, so the
        # two cannot drift and force-clear the wrong model's artifacts.
        local stem; stem="$(basename "${MODEL:-$(default_model)}" .pt)"
        warn "discarding cached exports for ${stem}"
        rm -f "models/${stem}"_*.onnx "models/${stem}"_*.meta.json
    fi

    step "Exporting models"
    local args=(--long-side "$LONG_SIDE")
    [ -n "$MODEL" ]  && args+=(--model "$MODEL")
    [ -n "$STREAM" ] && args+=(--stream "$STREAM")
    [ "$ASSESSMENT" -eq 0 ] && args+=(--no-assessment)
    python3 scripts/prepare_models.py "${args[@]}"
}

setup_verify() {
    wants verify || return 0

    step "Verifying the pipeline (${FRAMES} frames; first run builds the TensorRT engine)"
    local args=(--frames "$FRAMES" --long-side "$LONG_SIDE")
    [ -n "$MODEL" ]  && args+=(--model "$MODEL")
    # Deliberately the local file, not the RTSP default: no server is running
    # during setup, and the smoke test only needs frames from somewhere.
    args+=(--stream "${STREAM:-$(default_media)}")
    python3 validation/smoke_pipeline.py "${args[@]}"
}

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

main() {
    setup_image

    # The remaining stages need DeepStream, CUDA and pyds. From the host, hand
    # the rest of the run to the container in a single hop rather than paying
    # container startup per stage.
    if ! in_deepstream_container; then
        wants parser || wants env || wants models || wants verify || {
            log "Setup complete."
            return 0
        }

        require_docker
        export_host_ids
        step "Entering the DeepStream container for the remaining stages"

        local args=(); mapfile -t args < <(inner_args)
        # DSW_SETUP_INNER keeps the nested run from printing its own summary,
        # which would otherwise appear just above this one.
        docker compose run --rm -T \
            --label "$DSW_LABEL=$DSW_RUN_ID" \
            -e DSW_SETUP_INNER=1 \
            deepstream-dev scripts/setup.sh "${args[@]}"

        log ""
        log "Setup complete. Next:"
        log "  scripts/parser.sh     # detections in a window"
        log "  scripts/ros.sh        # RTSP + ROS + Foxglove stack"
        return 0
    fi

    setup_parser
    setup_env
    setup_models
    setup_verify

    [ -n "${DSW_SETUP_INNER:-}" ] && return 0

    log ""
    log "Setup complete."
}

main
