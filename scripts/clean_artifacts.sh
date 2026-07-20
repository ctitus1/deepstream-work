#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TARGETS=(
  ".setup-state"
  "configs/generated"
  "external"
  "lib"
  "bus.jpg"
  "labels.txt"
  "yolo12n.pt"
  "yolo12x.pt"
)

# The export environment is version-keyed (.venv-yolo-3.10, .venv-yolo-3.12) so
# the host and the container can each keep their own; a bare .venv-yolo may also
# exist from an older checkout. Collect whichever are present.
while IFS= read -r venv_dir; do
  TARGETS+=("$venv_dir")
done < <(find . -maxdepth 1 -type d -name '.venv-yolo*' -printf '%P\n')

# outputs/ mixes generated run output with checked-in diagram sources, which
# .gitignore deliberately whitelists (!outputs/diagrams/). Removing the whole
# directory deleted tracked files, so queue its generated entries individually.
if [ -d outputs ]; then
  while IFS= read -r out_entry; do
    TARGETS+=("$out_entry")
  done < <(find outputs -mindepth 1 -maxdepth 1 -not -name diagrams)
fi

while IFS= read -r cache_dir; do
  TARGETS+=("$cache_dir")
done < <(
  find . \
    \( -path './.git' -o -path './.venv-yolo*' -o -path './external' \) -prune \
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
  if [ ! -e "$target" ]; then
    continue
  fi

  # Cleanup only ever removes generated artifacts. Anything git tracks is by
  # definition source, so refuse rather than delete it. .gitkeep placeholders
  # are excepted: they only mark otherwise-empty generated directories, and are
  # recreated below.
  tracked="$(git ls-files -- "$target" 2>/dev/null | grep -Ev '(^|/)\.gitkeep$' || true)"
  if [ -n "$tracked" ]; then
    echo "Refusing to remove $target: it contains git-tracked files:"
    printf '  %s\n' $tracked
    continue
  fi

  rm -rf "$target"
done

mkdir -p models configs/generated
touch models/.gitkeep configs/generated/.gitkeep

echo "Artifacts removed."
