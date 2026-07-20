#!/usr/bin/env bash
# Build the DeepStream-Yolo custom bbox parser into lib/.
#
# Skipped entirely when the existing .so was built from the same toolchain and
# upstream ref, since nothing here changes between runs of the pipeline.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/common.sh

ROOT_DIR="$(pwd)"
FORCE="${FORCE:-0}"

[ "${1:-}" = "--force" ] && FORCE=1

# DeepStream-Yolo's Makefile interpolates CUDA_VER straight into
# /usr/local/cuda-$(CUDA_VER)/{include,lib64}, so it must name a real directory
# in this image, not just any installed CUDA. Each DeepStream release ships a
# specific toolkit (see DeepStream-Yolo's README compatibility table).
detect_cuda_version() {
    local ds
    ds="${DS_VERSION:-}"
    if [ -z "$ds" ] && [ -r /opt/nvidia/deepstream/deepstream/version ]; then
        ds="$(sed -n 's/^Version:[[:space:]]*//p' /opt/nvidia/deepstream/deepstream/version | head -1)"
    fi

    case "$ds" in
        7.1*) echo "12.6" ; return ;;
        8.0*) echo "12.8" ; return ;;
        9.0*) echo "13.1" ; return ;;
    esac

    # Unknown release: prefer whatever /usr/local/cuda resolves to, else the
    # highest versioned toolkit directory present.
    local linked
    linked="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
    if [ -n "$linked" ] && [ "${linked##*/cuda-}" != "$linked" ]; then
        echo "${linked##*/cuda-}"
        return
    fi
    ls -d /usr/local/cuda-*/ 2>/dev/null \
        | sed 's|.*/cuda-||; s|/$||' \
        | sort -V | tail -1
}

CUDA_VERSION="${CUDA_VERSION:-$(detect_cuda_version)}"
if [ -z "$CUDA_VERSION" ]; then
    echo "Could not determine a CUDA version; set CUDA_VERSION explicitly." >&2
    exit 1
fi
CUDA_MAJOR_MINOR="$(printf '%s\n' "$CUDA_VERSION" | awk -F. '{print $1 "." $2}')"
CUDA_PACKAGE_VERSION="${CUDA_MAJOR_MINOR/./-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${CUDA_MAJOR_MINOR}}"
DEEPSTREAM_YOLO_REF="${DEEPSTREAM_YOLO_REF:-2894babce8e75c49115dbe0c7b516289ed853565}"

# The build inputs are the CUDA toolchain and the upstream source revision;
# neither changes between pipeline runs, so a matching stamp means the existing
# .so is already correct. This skips an apt transaction and a full nvcc rebuild.
SIGNATURE="cuda=${CUDA_MAJOR_MINOR} ref=${DEEPSTREAM_YOLO_REF}"
TARGET_LIB="$ROOT_DIR/lib/libnvdsinfer_custom_impl_Yolo.so"

if [ "$FORCE" -eq 0 ] && stamp_valid yolo-parser "$SIGNATURE" "$TARGET_LIB"; then
    skip "YOLO parser library"
    exit 0
fi

# nvcc is the one tool the Makefile cannot do without, so its presence stands in
# for the whole toolchain: apt is an expensive no-op once it is installed.
if [ "$FORCE" -eq 1 ] || ! command -v nvcc >/dev/null 2>&1; then
    step "Installing build toolchain for CUDA ${CUDA_MAJOR_MINOR}"
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
      git \
      build-essential \
      make \
      g++ \
      "cuda-cudart-dev-${CUDA_PACKAGE_VERSION}" \
      "cuda-compiler-${CUDA_PACKAGE_VERSION}" \
      "cuda-nvcc-${CUDA_PACKAGE_VERSION}"
fi

# DeepStream-Yolo Makefile expects /usr/local/cuda-$CUDA_MAJOR_MINOR/lib64.
# CUDA packages in this image place libraries under targets/x86_64-linux/lib.
# Instead of patching interactively, make the expected layout reproducible here.
sudo mkdir -p "${CUDA_HOME}/lib64"

REAL_CUBLAS="$(find "${CUDA_HOME}" -name 'libcublas.so*' -print -quit 2>/dev/null || true)"
if [ -n "$REAL_CUBLAS" ]; then
  sudo ln -sfn "$REAL_CUBLAS" "${CUDA_HOME}/lib64/libcublas.so"
fi

REAL_CUDART="$(find "${CUDA_HOME}" -name 'libcudart.so*' -print -quit 2>/dev/null || true)"
if [ -z "$REAL_CUDART" ]; then
  echo "Could not find libcudart.so"
  exit 1
fi

sudo ln -sfn "$REAL_CUDART" "${CUDA_HOME}/lib64/libcudart.so"
sudo ln -sfn "${CUDA_HOME}" /usr/local/cuda

mkdir -p external lib

if [ ! -d external/DeepStream-Yolo ]; then
  git clone https://github.com/marcoslucianops/DeepStream-Yolo.git external/DeepStream-Yolo
fi

git -C external/DeepStream-Yolo checkout "$DEEPSTREAM_YOLO_REF" >/dev/null

cd external/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo

# Reaching this point means the stamp did not match, so the inputs really did
# change. A clean build is what guarantees the .so matches the checked-out ref.
rm -f ./*.o ./*.so layers/*.o

step "Compiling the parser against CUDA ${CUDA_MAJOR_MINOR}"
CUDA_VER="$CUDA_MAJOR_MINOR" make

cp libnvdsinfer_custom_impl_Yolo.so "$TARGET_LIB"

cd "$ROOT_DIR"
stamp_write yolo-parser "$SIGNATURE"

log "Built: $(ls -lh "$TARGET_LIB" | awk '{print $9, $5}')"
