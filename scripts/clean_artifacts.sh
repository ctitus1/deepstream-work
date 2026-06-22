#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TARGETS=(
  ".venv-yolo"
  "configs/generated"
  "external"
  "lib"
  "outputs"
  "bus.jpg"
  "labels.txt"
  "yolo12n.pt"
  "yolo12l.pt"
  "yolo12x.pt"
  "yolo26n.pt"
)

while IFS= read -r cache_dir; do
  TARGETS+=("$cache_dir")
done < <(
  find . \
    \( -path './.git' -o -path './.venv-yolo' -o -path './external' \) -prune \
    -o -type d -name __pycache__ -print
)

usage() {
  echo "Usage: $0 [--force] [--include-models]"
  echo
  echo "Without --force, this prints what would be removed."
  echo "The streams/ directory is user-provided local media and is never removed."
}

FORCE=0
INCLUDE_MODELS=0

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --include-models) INCLUDE_MODELS=1 ;;
    --include-streams)
      echo "Refusing to remove streams/: user-provided videos are not cleanup artifacts."
      exit 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [ "$INCLUDE_MODELS" -eq 1 ]; then
  TARGETS+=("models")
fi

echo "Artifact cleanup targets:"
for target in "${TARGETS[@]}"; do
  if [ -e "$target" ]; then
    du -sh "$target"
  fi
done

if [ "$FORCE" -ne 1 ]; then
  echo
  echo "Dry run only. Re-run with --force to remove these targets."
  exit 0
fi

for target in "${TARGETS[@]}"; do
  if [ -e "$target" ]; then
    rm -rf "$target"
  fi
done

mkdir -p models configs/generated
touch models/.gitkeep configs/generated/.gitkeep

echo "Artifacts removed."
