#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT_DIR="$(pwd)"

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
echo "Building YOLO parser against CUDA ${CUDA_MAJOR_MINOR}"
CUDA_PACKAGE_VERSION="${CUDA_MAJOR_MINOR/./-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${CUDA_MAJOR_MINOR}}"
DEEPSTREAM_YOLO_REF="${DEEPSTREAM_YOLO_REF:-2894babce8e75c49115dbe0c7b516289ed853565}"

sudo apt update

sudo apt install -y \
  git \
  build-essential \
  make \
  g++ \
  "cuda-cudart-dev-${CUDA_PACKAGE_VERSION}" \
  "cuda-compiler-${CUDA_PACKAGE_VERSION}" \
  "cuda-nvcc-${CUDA_PACKAGE_VERSION}"

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

rm -f *.o *.so layers/*.o

CUDA_VER="$CUDA_MAJOR_MINOR" make

cp libnvdsinfer_custom_impl_Yolo.so "$ROOT_DIR/lib/"

echo "Built:"
ls -lh "$ROOT_DIR/lib/libnvdsinfer_custom_impl_Yolo.so"
