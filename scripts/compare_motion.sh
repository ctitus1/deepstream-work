#!/usr/bin/env bash
# Compare motion approaches on a video, end to end.
#
#   scripts/compare_motion.sh streams/other.mp4
#
# Video in, side-by-side comparison video out. Runs each motion approach over
# the whole file and renders a labelled grid.
#
# Motion only: no detector, no assessment. Detection is needed to *score* an
# approach, not to run one, so it is opt-in behind --score rather than a cost
# every comparison pays.
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
  --score             Also run the detector and put each approach's F1 on its
                      panel. Costs a full detection pass over the video.
  --force             Redo every stage, ignoring cached artifacts.
  -h, --help          Show this help.

Examples:
  scripts/compare_motion.sh streams/other.mp4
  scripts/compare_motion.sh streams/other.mp4 --start 2160 --frames 150
  scripts/compare_motion.sh --approaches klt_homography,baseline --score
EOF
}

VIDEO=""
APPROACHES="klt_homography,gradient_diff,bgsub_compensated,baseline"
START=0
FRAMES=0
OUT=""
CRF=20
SCORE=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)      VIDEO="$2"; shift 2 ;;
        --approaches) APPROACHES="$2"; shift 2 ;;
        --start)      START="$2"; shift 2 ;;
        --frames)     FRAMES="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        --crf)        CRF="$2"; shift 2 ;;
        --score)      SCORE=1; shift ;;
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

# Artifacts are keyed by the video's stem so two videos can be compared without
# one silently scoring against the other's detections.
SLUG="$(basename "${VIDEO%.*}")"
RUN_DIR="eval/runs/${SLUG}"
DETECTIONS="eval/detections_${SLUG}.json"
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

# --- detections, only when scores were asked for ----------------------------
if [ "$SCORE" -eq 1 ]; then
    if [ "$FORCE" -eq 1 ] || [ ! -f "$DETECTIONS" ]; then
        step "Detecting on every frame (once per video, for the score labels)"
        gpu_run python3 eval/dump_detections.py --stream "$VIDEO" --out "$DETECTIONS"
    else
        skip "detections ($DETECTIONS)"
    fi
fi

# --- one run per approach ---------------------------------------------------
RUNS=()
for approach in ${APPROACHES//,/ }; do
    run_json="${RUN_DIR}/${approach}.json"
    if [ "$FORCE" -eq 1 ] || [ ! -f "$run_json" ]; then
        step "Running ${approach}"
        gpu_run python3 eval/run_motion.py \
            --approach "$approach" --stream "$VIDEO" --out "$run_json"
    else
        skip "${approach} ($run_json)"
    fi
    RUNS+=("$run_json")
done

# --- labels -----------------------------------------------------------------
LABELS=""
if [ "$SCORE" -eq 1 ]; then
    step "Scoring"
    LABELS="$(python3 - "$DETECTIONS" "${RUNS[@]}" <<'PY'
import json, sys
sys.path.insert(0, "eval")
from score import score, stationary_boxes_by_frame
from tracks import mover_boxes_by_frame, resolve

detections, runs = sys.argv[1], sys.argv[2:]
data, _, _, stationary = resolve(detections)
movers = mover_boxes_by_frame(data, stationary)
statics = stationary_boxes_by_frame(data, stationary)

labels = []
for path in runs:
    run = json.loads(open(path).read())
    result = score(run, movers, statics)
    labels.append(f"{result['approach']}  F1 {result['f1']:.3f}")
# Pipe-separated: an approach name may contain a comma, a label never a pipe.
print("|".join(labels))
PY
)"
    printf '%s\n' "$LABELS" | tr '|' '\n' | sed 's/^/  /'
fi

# --- render -----------------------------------------------------------------
RUNS_CSV="$(IFS=,; echo "${RUNS[*]}")"
GEOMETRY="$(render_run eval/make_comparison_video.py \
    --video "$VIDEO" --runs "$RUNS_CSV" --geometry)"
FPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
    -of csv=p=0 "$VIDEO" | head -1)"

step "Rendering ${GEOMETRY} at ${FPS} fps -> ${OUT}"
render_run eval/make_comparison_video.py \
    --video "$VIDEO" --runs "$RUNS_CSV" --labels "$LABELS" \
    --start "$START" --frames "$FRAMES" \
    | ffmpeg -y -loglevel error \
        -f rawvideo -pix_fmt bgr24 -s "$GEOMETRY" -r "$FPS" -i - \
        -c:v libx264 -preset slow -crf "$CRF" -pix_fmt yuv420p \
        -movflags +faststart "$OUT"

log ""
log "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
