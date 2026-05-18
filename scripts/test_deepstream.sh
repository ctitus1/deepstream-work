#!/usr/bin/env bash
set -euo pipefail

deepstream-app --version-all

python3 - <<'PY'
import ctypes
lib = ctypes.CDLL("libcudart.so")
count = ctypes.c_int()
print("cudaGetDeviceCount =", lib.cudaGetDeviceCount(ctypes.byref(count)), count.value)
print("cudaSetDevice =", lib.cudaSetDevice(0))
PY

gst-launch-1.0 -v videotestsrc num-buffers=300 ! \
  video/x-raw,format=RGBA,width=1280,height=720 ! \
  nveglglessink sync=false
