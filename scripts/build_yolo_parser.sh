#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sudo apt update

sudo apt install -y \
  git \
  build-essential \
  make \
  g++ \
  cuda-cudart-dev-13-1 \
  cuda-compiler-13-1 \
  cuda-nvcc-13-1

# DeepStream-Yolo Makefile expects /usr/local/cuda-13.1/lib64.
# CUDA 13.1 packages in this image place libraries under targets/x86_64-linux/lib.
# Instead of patching interactively, make the expected layout reproducible here.
sudo mkdir -p /usr/local/cuda-13.1/lib64

if [ -f /usr/local/cuda-13.1/targets/x86_64-linux/lib/libcublas.so.13 ]; then
  sudo ln -sfn /usr/local/cuda-13.1/targets/x86_64-linux/lib/libcublas.so.13 \
    /usr/local/cuda-13.1/lib64/libcublas.so
fi

REAL_CUDART="$(find /usr/local/cuda-13.1 -name 'libcudart.so*' | head -n1)"
if [ -z "$REAL_CUDART" ]; then
  echo "Could not find libcudart.so"
  exit 1
fi

sudo ln -sfn "$REAL_CUDART" /usr/local/cuda-13.1/lib64/libcudart.so
sudo ln -sfn /usr/local/cuda-13.1 /usr/local/cuda

mkdir -p external lib

if [ ! -d external/DeepStream-Yolo ]; then
  git clone https://github.com/marcoslucianops/DeepStream-Yolo.git external/DeepStream-Yolo
fi

cd external/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo

rm -f *.o *.so layers/*.o

CUDA_VER=13.1 make

cp libnvdsinfer_custom_impl_Yolo.so /home/user/deepstream-work/lib/

echo "Built:"
ls -lh /home/user/deepstream-work/lib/libnvdsinfer_custom_impl_Yolo.so
