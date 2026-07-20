#!/usr/bin/env bash
# Sweep two klt-homography parameters and render the grid.
#
#   scripts/compare_klt.sh streams/dtc_c2-d3-rgb.mp4
#
# Every cell is the same approach on the same frames under a different pair of
# values, so anything that differs between panels is the parameters and nothing
# else.
#
# The default grid is laid out so both axes run the same direction -- toward
# more sensitivity -- and each axis buys it a different way:
#
#   ACROSS (x), residual_floor DESCENDING 24 -> 3
#       Branch pixels of unexplained displacement a point needs before it counts
#       as moving: the literal "how much movement" dial. Rightward asks for less
#       movement, so smaller motion is detected. The values descend so that
#       rightward always means more sensitive, whatever the numbers say.
#
#   DOWN (y), lag ASCENDING 4 -> 24
#       Frames the camera model is fitted over. This does not lower the bar, it
#       raises the signal: real displacement accumulates over the window while
#       tracking jitter does not, so a slow target clears the same threshold
#       given longer. It should catch more of the true slow movement -- and it
#       gives background misregistration the same time to accumulate, so it
#       should cost false positives too.
#
# So: rightward, smaller movements register. Downward, more of the real slow
# motion is caught at the price of more spurious boxes. Top-left is the most
# conservative cell, bottom-right the most permissive, and the useful reading is
# where along each axis the boxes stop being the target and start being noise.
#
# Whether the FP cost actually behaves that way is the point of looking, not an
# assumption baked into the layout.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh

usage() {
    cat <<'EOF'
Usage:
  scripts/compare_klt.sh [VIDEO] [options]

Sweeps two klt-homography parameters over a video and renders a labelled grid,
one panel per combination. All variants run in a single pass over the video.

Options:
  --param-a NAME      Parameter varied DOWN the grid. Default: lag
  --values-a LIST     Its values, top to bottom. Default: 4,8,16,24
  --param-b NAME      Parameter varied ACROSS the grid. Default: residual_floor
  --values-b LIST     Its values, left to right. Default: 24,12,6,3

Order the values so both axes run toward more sensitivity: rightward should
detect smaller movement, downward should catch more of it at the cost of more
false positives.
  --cfg JSON          Extra config applied to every variant. Default: {}
  --width N           Branch resolution width. Default: 960
  --height N          Branch resolution height. Default: 540
  --start N           First frame to render. Default: 0
  --frames N          Frames to render. Default: 0 (to the end)
  --tile-width N      Panel width. Default: 480 (4 across = 1920)
  --tile-height N     Panel height. Default: 270
  --crf N             x264 quality, lower is better. Default: 20
  --out PATH          Output video. Default names itself after what it swept.
  --force             Re-run variants that are already cached.
  -h, --help          Show this help.

Examples:
  scripts/compare_klt.sh streams/dtc_c2-d3-rgb.mp4
  scripts/compare_klt.sh --param-b min_travel --values-b 4,10,20,40
  scripts/compare_klt.sh --param-a ransac_threshold --values-a 0.5,1,2,4
EOF
}

VIDEO=""
PARAM_A="lag";            VALUES_A="2,4,8,16"
PARAM_B="residual_floor"; VALUES_B="24,12,6,3"
EXTRA_CFG="{}"
WIDTH=960; HEIGHT=540
START=0; FRAMES=0
TILE_W=480; TILE_H=270
CRF=28
OUT=""
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --param-a)     PARAM_A="$2"; shift 2 ;;
        --values-a)    VALUES_A="$2"; shift 2 ;;
        --param-b)     PARAM_B="$2"; shift 2 ;;
        --values-b)    VALUES_B="$2"; shift 2 ;;
        --cfg)         EXTRA_CFG="$2"; shift 2 ;;
        --width)       WIDTH="$2"; shift 2 ;;
        --height)      HEIGHT="$2"; shift 2 ;;
        --start)       START="$2"; shift 2 ;;
        --frames)      FRAMES="$2"; shift 2 ;;
        --tile-width)  TILE_W="$2"; shift 2 ;;
        --tile-height) TILE_H="$2"; shift 2 ;;
        --crf)         CRF="$2"; shift 2 ;;
        --out)         OUT="$2"; shift 2 ;;
        --force)       FORCE=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        -*)            die "unknown option: $1 (try --help)" ;;
        *)             [ -z "$VIDEO" ] || die "more than one video given: $VIDEO and $1"
                       VIDEO="$1"; shift ;;
    esac
done

require_docker
export_host_ids
command -v ffmpeg >/dev/null || die "ffmpeg is not installed on the host"

VIDEO="$(project_relative "${VIDEO:-$(default_media)}")"
[ -f "$VIDEO" ] || die "video not found: $VIDEO
streams/ holds: $(ls streams/ 2>/dev/null | tr '\n' ' ')"

SLUG="$(basename "${VIDEO%.*}")"
SWEEP="${PARAM_A}-${PARAM_B}"
RUN_DIR="eval/runs/${SLUG}/klt-${SWEEP}"
OUT="${OUT:-outputs/${SLUG}_klt_${SWEEP}.mp4}"
mkdir -p "$RUN_DIR" "$(dirname "$OUT")"

# Variant list, and the run files in the same order, so the grid reads
# left-to-right as B varies and top-to-bottom as A varies.
# Read line by line rather than with one whitespace-splitting `read`: the panel
# labels contain spaces, which would cut the JSON mid-string.
mapfile -t SWEEP_FIELDS < <(python3 - "$PARAM_A" "$VALUES_A" "$PARAM_B" "$VALUES_B" "$RUN_DIR" "$EXTRA_CFG" <<'PY'
import json, sys

param_a, values_a, param_b, values_b, run_dir, extra = sys.argv[1:7]
extra = json.loads(extra)


def parse(text):
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(int(raw) if raw.lstrip("-").isdigit() else float(raw))
    return out


def tag(value):
    return str(value).replace(".", "p")


variants, runs = [], []
for a in parse(values_a):
    for b in parse(values_b):
        name = f"{param_a}{tag(a)}_{param_b}{tag(b)}"
        variants.append(
            {
                "name": name,
                "label": f"{param_a}={a}  {param_b}={b}",
                "cfg": {**extra, param_a: a, param_b: b},
            }
        )
        runs.append(f"{run_dir}/{name}.json")

# One field per line; labels contain spaces so they cannot share a line.
print(json.dumps(variants, separators=(",", ":")))
print(",".join(runs))
print(len(parse(values_b)))
PY
)
VARIANTS="${SWEEP_FIELDS[0]}"
RUNS_CSV="${SWEEP_FIELDS[1]}"
COLS="${SWEEP_FIELDS[2]}"

COUNT="$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$VARIANTS")"

# --- run whichever variants are not already on disk --------------------------
MISSING="$(python3 - "$VARIANTS" "$RUN_DIR" "$FORCE" <<'PY'
import json, os, sys
variants, run_dir, force = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
missing = [v for v in variants if force or not os.path.isfile(f"{run_dir}/{v['name']}.json")]
print(json.dumps(missing, separators=(",", ":")))
PY
)"
MISSING_COUNT="$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$MISSING")"

if [ "$MISSING_COUNT" -gt 0 ]; then
    step "Running ${MISSING_COUNT}/${COUNT} variants in one pass over ${VIDEO}"
    log "  sweeping ${PARAM_A} x ${PARAM_B} at ${WIDTH}x${HEIGHT}"
    docker compose run --rm -T deepstream-dev python3 eval/run_motion.py \
        --approach klt_homography --variants "$MISSING" \
        --stream "$VIDEO" --width "$WIDTH" --height "$HEIGHT" \
        --out "${RUN_DIR}/"
else
    skip "all ${COUNT} variants"
fi

# --- render ------------------------------------------------------------------
render_run() {
    docker run --rm -i --user "${HOST_UID}:${HOST_GID}" --entrypoint python3 \
        -v "$PWD":/workspace/deepstream-work -w /workspace/deepstream-work \
        "deepstream-work:${DS_VERSION:-7.1}" "$@"
}

GEOMETRY="$(render_run eval/make_comparison_video.py \
    --video "$VIDEO" --runs "$RUNS_CSV" --cols "$COLS" \
    --tile-width "$TILE_W" --tile-height "$TILE_H" --geometry)"
FPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
    -of csv=p=0 "$VIDEO" | head -1)"

step "Rendering ${COUNT} panels at ${GEOMETRY}, ${FPS} fps -> ${OUT}"
render_run eval/make_comparison_video.py \
    --video "$VIDEO" --runs "$RUNS_CSV" --cols "$COLS" \
    --tile-width "$TILE_W" --tile-height "$TILE_H" \
    --start "$START" --frames "$FRAMES" \
    | ffmpeg -y -loglevel error \
        -f image2pipe -vcodec mjpeg -r "$FPS" -i - \
        -c:v libx264 -preset veryfast -crf "$CRF" -pix_fmt yuv420p \
        -movflags +faststart "$OUT"

log ""
log "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
