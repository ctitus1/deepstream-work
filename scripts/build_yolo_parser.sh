#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT_DIR="$(pwd)"
CUDA_VERSION="${CUDA_VERSION:-13.1}"
CUDA_MAJOR_MINOR="$(printf '%s\n' "$CUDA_VERSION" | awk -F. '{print $1 "." $2}')"
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
