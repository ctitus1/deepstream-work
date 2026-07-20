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

Cleanup (the inverse of the stages above):
  --clean               List every generated artifact that would be removed.
  --yes                 With --clean, actually remove them.
  --include-models      With --clean, also remove models/. streams/ is user
                        media and is never removed.

Examples:
  scripts/setup.sh                              # full setup, or a fast no-op
  scripts/setup.sh --model best.pt              # export one new model
  scripts/setup.sh --only verify                # just re-check the pipeline
  scripts/setup.sh --force parser               # rebuild the parser library
  scripts/setup.sh --clean                      # show what cleanup would remove
  scripts/setup.sh --clean --yes                # remove it
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
CLEAN=0
CLEAN_YES=0
CLEAN_MODELS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)     CLEAN=1; shift ;;
        --yes)       CLEAN_YES=1; shift ;;
        --include-models) CLEAN_MODELS=1; shift ;;
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

in_list() {
    local needle="$1" item
    shift
    for item in "$@"; do
        [ "$item" = "$needle" ] && return 0
    done
    return 1
}

# Validate stage names up front: a typo silently doing nothing is worse than a
# short error, especially for --skip.
validate_stages() {
    local list="$1" label="$2" name
    [ -z "$list" ] && return 0
    for name in ${list//,/ }; do
        in_list "$name" "${ALL_STAGES[@]}" \
            || die "$label: unknown stage '$name'. Valid: ${ALL_STAGES[*]}"
    done
}

validate_stages "$ONLY" "--only"
validate_stages "$SKIP" "--skip"
[ -n "$FORCE_STAGE" ] && { in_list "$FORCE_STAGE" "${ALL_STAGES[@]}" \
    || die "--force: unknown stage '$FORCE_STAGE'. Valid: ${ALL_STAGES[*]}"; }

wants() {
    local stage="$1" name

    if [ -n "$ONLY" ]; then
        for name in ${ONLY//,/ }; do [ "$name" = "$stage" ] && return 0; done
        return 1
    fi

    for name in ${SKIP//,/ }; do [ "$name" = "$stage" ] && return 1; done

    # Naming a non-default stage with --force is a request to run it.
    [ "$FORCE_STAGE" = "$stage" ] && return 0

    in_list "$stage" "${DEFAULT_STAGES[@]}"
}

forced() {
    [ "$FORCE_ALL" -eq 1 ] || [ "$FORCE_STAGE" = "$1" ]
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

# Both of these delegate to a script that keeps its own stamp, so the stage is
# just "run it, and pass --force through".
run_stamped_stage() {
    local stage="$1" script="$2"
    wants "$stage" || return 0
    if forced "$stage"; then
        "$script" --force
    else
        "$script"
    fi
}

setup_parser() { run_stamped_stage parser scripts/setup/yolo_parser.sh; }
setup_env()    { run_stamped_stage env scripts/setup/yolo_env.sh; }

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
        # The .engine glob matters as much as the .onnx one: nvinfer loads an
        # existing engine without validating it against the ONNX, so clearing
        # only the export would re-run the whole stage and still infer with the
        # previous model's weights.
        rm -f "models/${stem}"_*.onnx "models/${stem}"_*.onnx_*.engine \
              "models/${stem}"_*.meta.json
    fi

    step "Exporting models"
    local args=(--long-side "$LONG_SIDE")
    [ -n "$MODEL" ]  && args+=(--model "$MODEL")
    [ -n "$STREAM" ] && args+=(--stream "$STREAM")
    [ "$ASSESSMENT" -eq 0 ] && args+=(--no-assessment)
    python3 scripts/setup/prepare_models.py "${args[@]}"
}

setup_verify() {
    wants verify || return 0

    step "Verifying the pipeline (${FRAMES} frames; first run builds the TensorRT engine)"
    local args=(--frames "$FRAMES" --long-side "$LONG_SIDE")
    [ -n "$MODEL" ]  && args+=(--model "$MODEL")
    # Deliberately the local file, not the RTSP default: no server is running
    # during setup, and the smoke test only needs frames from somewhere.
    args+=(--stream "${STREAM:-$(default_media)}")
    python3 scripts/smoke_pipeline.py "${args[@]}"
}

# ---------------------------------------------------------------------------
# Clean: the inverse of everything above
# ---------------------------------------------------------------------------

# Undoing setup belongs with setup rather than in a script of its own -- the two
# have to agree on exactly which paths are generated, and split across files
# they drifted.
#
# Dry run unless --yes. streams/ is never touched: it is user media, not an
# artifact, and there is no way to get it back.
clean_artifacts() {
    local targets=(.setup-state configs/generated external lib bus.jpg labels.txt yolo12n.pt yolo12x.pt)

    # Version-keyed export venvs (.venv-yolo-3.10, .venv-yolo-3.12); a bare
    # .venv-yolo may also survive from an older checkout.
    local entry
    while IFS= read -r entry; do targets+=("$entry"); done \
        < <(find . -maxdepth 1 -type d -name '.venv-yolo*' -printf '%P\n')

    # outputs/ mixes run output with the checked-in diagram sources that
    # .gitignore whitelists, so queue its entries individually rather than
    # removing the directory.
    if [ -d outputs ]; then
        while IFS= read -r entry; do targets+=("$entry"); done \
            < <(find outputs -mindepth 1 -maxdepth 1 -not -name diagrams)
    fi

    while IFS= read -r entry; do targets+=("$entry"); done \
        < <(find . \( -path './.git' -o -path './.venv-yolo*' -o -path './external' \) -prune \
                 -o -type d -name __pycache__ -print)

    [ "$CLEAN_MODELS" -eq 1 ] && targets+=(models)

    log "Cleanup targets:"
    local target found=0
    for target in "${targets[@]}"; do
        [ -e "$target" ] && { du -sh "$target"; found=1; }
    done
    [ "$found" -eq 0 ] && { log "  nothing to remove."; return 0; }

    if [ "$CLEAN_YES" -ne 1 ]; then
        log ""
        log "Dry run. Re-run with --clean --yes to remove these."
        return 0
    fi

    for target in "${targets[@]}"; do
        [ -e "$target" ] || continue

        # Cleanup only ever removes generated artifacts, so anything git tracks
        # is by definition source and is refused. .gitkeep markers are excepted:
        # they only pin otherwise-empty generated directories, and are restored
        # below.
        local tracked
        tracked="$(git ls-files -- "$target" 2>/dev/null | grep -Ev '(^|/)\.gitkeep$' || true)"
        if [ -n "$tracked" ]; then
            warn "keeping $target: it holds git-tracked files"
            continue
        fi

        rm -rf "$target"
    done

    mkdir -p models configs/generated
    touch models/.gitkeep configs/generated/.gitkeep
    log ""
    log "Artifacts removed."
}

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

main() {
    [ "$CLEAN" -eq 1 ] && { clean_artifacts; return 0; }

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
