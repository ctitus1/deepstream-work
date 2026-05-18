#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="yolo12n.pt"
LONG_SIDE="${1:-640}"
STREAM="streams/dtc-d4.ts"
STRIDE=32

mkdir -p models external outputs

detect_dims() {
    if command -v gst-discoverer-1.0 >/dev/null 2>&1 && [ -f "$STREAM" ]; then
        local out w h
        out="$(gst-discoverer-1.0 "$STREAM" 2>/dev/null || true)"
        w="$(printf '%s\n' "$out" | awk '/Width:/ {print $2; exit}')"
        h="$(printf '%s\n' "$out" | awk '/Height:/ {print $2; exit}')"

        if [ -n "$w" ] && [ -n "$h" ]; then
            echo "$w $h"
            return
        fi
    fi

    # Safe default for 1080p-like video.
    echo "1920 1080"
}

read SRC_W SRC_H < <(detect_dims)

read MODEL_W MODEL_H < <(python3 - <<PY
src_w = int("$SRC_W")
src_h = int("$SRC_H")
long_side = int("$LONG_SIDE")
stride = int("$STRIDE")

def ceil_to_stride(x):
    x = int(round(x))
    return max(stride, ((x + stride - 1) // stride) * stride)

if src_w >= src_h:
    w = ceil_to_stride(long_side)
    h = ceil_to_stride(long_side * src_h / src_w)
else:
    h = ceil_to_stride(long_side)
    w = ceil_to_stride(long_side * src_w / src_h)

print(w, h)
PY
)

echo "Source: ${SRC_W}x${SRC_H}"
echo "Model:  ${MODEL_W}x${MODEL_H}"

if [ ! -d .venv-yolo ]; then
    python3 -m venv .venv-yolo
fi

source .venv-yolo/bin/activate

python -m pip install --upgrade 'pip' 'setuptools<82' wheel

python -m pip install \
    ultralytics \
    onnx \
    onnxsim \
    onnxruntime \
    onnxscript \
    open_clip_torch \
    timm \
    einops

if [ ! -d external/DeepStream-Yolo ]; then
    git clone https://github.com/marcoslucianops/DeepStream-Yolo.git external/DeepStream-Yolo
fi

if [ ! -f "models/$MODEL" ]; then
    echo "Downloading $MODEL with Ultralytics..."
    yolo predict model="$MODEL" source='https://ultralytics.com/images/bus.jpg' imgsz="$MODEL_W" save=False >/dev/null

    FOUND="$(find . "$HOME" -name "$MODEL" 2>/dev/null | head -n1 || true)"
    if [ -z "$FOUND" ]; then
        echo "Could not find downloaded $MODEL"
        exit 1
    fi

    cp -f "$FOUND" "models/$MODEL"
fi

python external/DeepStream-Yolo/utils/export_yolov12.py \
    -w "/home/user/deepstream-work/models/$MODEL" \
    -s "$MODEL_H" "$MODEL_W" \
    --opset 18 \
    --simplify

rm -f labels.txt

cat > models/coco_labels.txt <<'LABELS'
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
LABELS

python - <<PY
import json
from pathlib import Path

meta = {
    "model": "yolo12n",
    "source_stream": "$STREAM",
    "source_width": int("$SRC_W"),
    "source_height": int("$SRC_H"),
    "model_width": int("$MODEL_W"),
    "model_height": int("$MODEL_H"),
    "onnx": "models/yolo12n.onnx",
    "engine": "models/yolo12n.onnx_b1_gpu0_fp16.engine",
    "labels": "models/coco_labels.txt",
}

Path("models/yolo12n.meta.json").write_text(json.dumps(meta, indent=2) + "\\n")
print(json.dumps(meta, indent=2))
PY

python - <<'PY'
import onnx

path = "models/yolo12n.onnx"
m = onnx.load(path)
onnx.checker.check_model(m)

print("ONNX OK:", path)

for x in m.graph.input:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in x.type.tensor_type.shape.dim]
    print("INPUT ", x.name, dims)

for x in m.graph.output:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in x.type.tensor_type.shape.dim]
    print("OUTPUT", x.name, dims)
PY

rm -f models/yolo12n*.engine models/yolo12n.onnx*.engine

echo
echo "Done."
echo "Run:"
echo "  python3 src/deepstream_yolo12n_app.py"
