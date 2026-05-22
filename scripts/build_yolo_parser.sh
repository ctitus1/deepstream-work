#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p lib

CUDA_VER=12.5 make -C external/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo clean
CUDA_VER=12.5 make -C external/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo

cp external/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \
  lib/libnvdsinfer_custom_impl_Yolo.so

ls -lh lib/libnvdsinfer_custom_impl_Yolo.so
