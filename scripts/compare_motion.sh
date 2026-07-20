#!/usr/bin/env bash
# Compare motion approaches on a video, end to end.
#
#   scripts/compare_motion.sh streams/other.mp4
#
# Video in, side-by-side comparison video out. Runs each motion approach over
# the whole file and renders a labelled grid.
#
# Motion only: no detector, no assessment, no scoring. This answers "what do
# these approaches see", which is a different question from "which one wins" --
# eval/score.py answers that one, against ground truth, in a table.
#
# Every stage caches, so re-running is cheap and a failed run resumes rather
# than starting over. --force redoes the lot.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh

usage() {
    cat <<'EOF'
Usage:
  scripts/compare_motion.sh [VIDEO] [options]

Runs each motion approach over a video and renders a labelled grid comparison.
Motion only -- no detector is run. Artifacts are keyed by the video's name, so
comparing two videos never mixes their results.

Options:
  --approaches LIST   Comma-separated approach modules, in grid order. Default:
                      klt_homography,gradient_diff,bgsub_compensated,baseline
  --start N           First frame to render. Default: 0
  --frames N          Frames to render. Default: 0 (to the end)
  --out PATH          Output video. Default: outputs/<video>_comparison.mp4
  --crf N             x264 quality, lower is better. Default: 20
  --force             Redo every stage, ignoring cached artifacts.
  -h, --help          Show this help.

Examples:
  scripts/compare_motion.sh streams/other.mp4
  scripts/compare_motion.sh streams/other.mp4 --start 2160 --frames 150
  scripts/compare_motion.sh --approaches klt_homography,baseline
EOF
}

VIDEO=""
APPROACHES="klt_homography,gradient_diff,bgsub_compensated,baseline"
START=0
FRAMES=0
OUT=""
CRF=20
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)      VIDEO="$2"; shift 2 ;;
        --approaches) APPROACHES="$2"; shift 2 ;;
        --start)      START="$2"; shift 2 ;;
        --frames)     FRAMES="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        --crf)        CRF="$2"; shift 2 ;;
        --force)      FORCE=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        -*)           die "unknown option: $1 (try --help)" ;;
        # A bare path is the video, so the common case reads as
        # "compare this video" rather than "--video this video".
        *)            [ -z "$VIDEO" ] || die "more than one video given: $VIDEO and $1"
                      VIDEO="$1"; shift ;;
    esac
done

require_docker
export_host_ids
command -v ffmpeg >/dev/null || die "ffmpeg is not installed on the host"

VIDEO="$(project_relative "${VIDEO:-$(default_media)}")"
[ -f "$VIDEO" ] || die "video not found: $VIDEO
streams/ holds: $(ls streams/ 2>/dev/null | tr '\n' ' ')"

# Artifacts are keyed by the video's stem so comparing two videos never mixes
# one's runs into the other's grid.
SLUG="$(basename "${VIDEO%.*}")"
RUN_DIR="eval/runs/${SLUG}"
OUT="${OUT:-outputs/${SLUG}_comparison.mp4}"
mkdir -p "$RUN_DIR" "$(dirname "$OUT")"

# The GPU container, for the stages that need DeepStream and pyds.
gpu_run() {
    docker compose run --rm -T deepstream-dev "$@"
}

# The rendering container. Deliberately NOT `docker compose run`, and
# deliberately bypassing the entrypoint: both write to stdout, and this stage
# pipes raw video frames through stdout. A single stray banner byte shifts every
# frame and rotates the colour channels. See eval/make_comparison_video.py.
render_run() {
    docker run --rm -i --user "${HOST_UID}:${HOST_GID}" --entrypoint python3 \
        -v "$PWD":/workspace/deepstream-work -w /workspace/deepstream-work \
        "deepstream-work:${DS_VERSION:-7.1}" "$@"
}

# --- the approaches, in a single pass ---------------------------------------
# Decoding 4K H.265 dominates, and it does not depend on which approach is
# asking, so every approach that still needs running goes through one pass
# instead of one each. Four approaches used to mean four decodes.
RUNS=()
MISSING=()
for approach in ${APPROACHES//,/ }; do
    run_json="${RUN_DIR}/${approach}.json"
    RUNS+=("$run_json")
    if [ "$FORCE" -eq 1 ] || [ ! -f "$run_json" ]; then
        MISSING+=("$approach")
    else
        skip "${approach} ($run_json)"
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    batch="$(IFS=,; echo "${MISSING[*]}")"
    step "Running ${#MISSING[@]} approach(es) in one pass: ${batch}"
    gpu_run python3 eval/run_motion.py \
        --approach "$batch" --stream "$VIDEO" --out "${RUN_DIR}/"
fi

# --- render -----------------------------------------------------------------
# Panels are labelled with each run's own approach name; the renderer reads it
# out of the run JSON.
RUNS_CSV="$(IFS=,; echo "${RUNS[*]}")"
GEOMETRY="$(render_run eval/make_comparison_video.py \
    --video "$VIDEO" --runs "$RUNS_CSV" --geometry)"
FPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
    -of csv=p=0 "$VIDEO" | head -1)"

step "Rendering ${GEOMETRY} at ${FPS} fps -> ${OUT}"
# The renderer emits MJPEG rather than raw frames: docker's stdout proxy is the
# slowest thing in this whole script at ~55 MB/s, and raw 1080p is 6.2 MB a
# frame. See eval/make_comparison_video.py.
render_run eval/make_comparison_video.py \
    --video "$VIDEO" --runs "$RUNS_CSV" \
    --start "$START" --frames "$FRAMES" \
    | ffmpeg -y -loglevel error \
        -f image2pipe -vcodec mjpeg -r "$FPS" -i - \
        -c:v libx264 -preset veryfast -crf "$CRF" -pix_fmt yuv420p \
        -movflags +faststart "$OUT"

log ""
log "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
