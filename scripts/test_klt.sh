#!/usr/bin/env bash
# Run the motion detector on one or more videos and produce a video of it.
#
#   scripts/test_klt.sh streams/lorton-d4-thermal-nano.mp4
#   scripts/test_klt.sh streams/*.mp4 --target-height 100
#
# The short answer to "what does this do on that clip". Runs klt-homography at
# its defaults and writes outputs/<video>_klt.mp4, one per input. For varying a
# parameter and comparing side by side, use scripts/compare_klt.sh.
#
# `scripts/compare_klt.sh VIDEO` with no sweep options does exactly what this
# does for a single video; this exists to take several at once and to be the
# obviously-named thing.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh

usage() {
    cat <<'EOF'
Usage:
  scripts/test_klt.sh [VIDEO...] [options]

Runs klt-homography at its defaults over each video and writes a labelled
video of what it found to outputs/<video>_klt.mp4.

Options:
  --target-height N   How tall a target is expected to be, in SOURCE pixels.
                      This is what every pixel threshold is scaled against, so
                      it is the one setting worth getting right per camera.
                      Default: the approach's own (440, measured on 4K aerial).
  --start N           First frame to render. Default: 0
  --frames N          Frames to render. Default: 0 (to the end)
  --crf N             x264 quality, lower is better. Default: 20
  --force             Re-run even if the analysis is already cached.
  -h, --help          Show this help.

With no video, the one in streams/ is used.

Examples:
  scripts/test_klt.sh
  scripts/test_klt.sh streams/lorton-d4-thermal-nano.mp4 --target-height 100
  scripts/test_klt.sh streams/*-nano*.mp4 --target-height 100
EOF
}

VIDEOS=()
TARGET_HEIGHT=""
PASS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-height) TARGET_HEIGHT="$2"; shift 2 ;;
        --start|--frames|--crf) PASS+=("$1" "$2"); shift 2 ;;
        --force)         PASS+=("$1"); shift ;;
        -h|--help)       usage; exit 0 ;;
        -*)              die "unknown option: $1 (try --help)" ;;
        *)               VIDEOS+=("$1"); shift ;;
    esac
done

[ ${#VIDEOS[@]} -gt 0 ] || VIDEOS=("$(default_media)")
[ -n "$TARGET_HEIGHT" ] && PASS+=(--cfg "{\"target_height\": ${TARGET_HEIGHT}}")

failed=0
for video in "${VIDEOS[@]}"; do
    log ""
    step "=== $(basename "$video") ==="
    # compare_klt.sh with no sweep options runs one panel at the defaults,
    # derives the tile size from the source, and names the output itself.
    if ! scripts/compare_klt.sh "$video" ${PASS[@]+"${PASS[@]}"}; then
        warn "failed: $video"
        failed=$((failed + 1))
    fi
done

log ""
if [ "$failed" -gt 0 ]; then
    die "$failed of ${#VIDEOS[@]} video(s) failed"
fi
log "Done: ${#VIDEOS[@]} video(s)."
