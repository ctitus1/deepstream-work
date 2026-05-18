#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"

# Single model size parameter.
# The script detects the input stream aspect ratio and converts this one dimension
# into a stride-safe WIDTH x HEIGHT for YOLO/DeepStream.
#
# Examples:
#   ./scripts/setup_and_export_yolo.sh yolo11n.pt 640
#   ./scripts/setup_and_export_yolo.sh yolo12x.pt 640
#   ./scripts/setup_and_export_yolo.sh yolo12x.pt 1920
#
# For a 1920x1080 stream:
#   640  -> 640x352
#   1920 -> 1920x1088
MODEL_SIZE="${2:-1920}"

# Stream used to derive aspect ratio.
# Override as third arg if needed:
#   ./scripts/setup_and_export_yolo.sh yolo12x.pt 640 streams/other.ts
STREAM="${3:-streams/dtc-d4.ts}"

STRIDE=32

if [ -z "$MODEL" ]; then
    echo "Usage:"
    echo "  $0 <model.pt|model-name> [model_size] [stream]"
    echo
    echo "Examples:"
    echo "  $0 yolo11n.pt 640"
    echo "  $0 yolo12x.pt 640 streams/dtc-d4.ts"
    echo "  $0 yolo12x.pt 1920 streams/dtc-d4.ts"
    exit 1
fi

cd "$(dirname "$0")/.."

mkdir -p models external configs lib outputs

detect_dims() {
    local stream="$1"

    if command -v gst-discoverer-1.0 >/dev/null 2>&1 && [ -f "$stream" ]; then
        local out
        out="$(gst-discoverer-1.0 "$stream" 2>/dev/null || true)"
        local w h
        w="$(printf '%s\n' "$out" | awk '/Width:/ {print $2; exit}')"
        h="$(printf '%s\n' "$out" | awk '/Height:/ {print $2; exit}')"
        if [ -n "$w" ] && [ -n "$h" ]; then
            echo "$w $h"
            return
        fi
    fi

    # Best 1080p default.
    echo "1920 1080"
}

read SRC_W SRC_H < <(detect_dims "$STREAM")

read INFER_W INFER_H < <(python3 - <<PY
src_w = int("$SRC_W")
src_h = int("$SRC_H")
long_edge = int("$MODEL_SIZE")
stride = int("$STRIDE")

def round_stride(x):
    x = int(x)
    return max(stride, ((x + stride - 1) // stride) * stride)

if src_w >= src_h:
    w = round_stride(long_edge)
    h = round_stride(long_edge * src_h / src_w)
else:
    h = round_stride(long_edge)
    w = round_stride(long_edge * src_w / src_h)

print(w, h)
PY
)

echo "Source stream: $STREAM"
echo "Source size:   ${SRC_W}x${SRC_H}"
echo "YOLO size:     ${INFER_W}x${INFER_H}"

if [ ! -d .venv-yolo ]; then
    echo "Creating YOLO export virtual environment..."
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
    echo "Cloning DeepStream-Yolo..."
    git clone https://github.com/marcoslucianops/DeepStream-Yolo.git external/DeepStream-Yolo
fi

MODEL_BASENAME="$(basename "$MODEL")"
MODEL_STEM="${MODEL_BASENAME%.pt}"

if [ -f "$MODEL" ]; then
    cp -f "$MODEL" "models/$MODEL_BASENAME"
elif [ -f "models/$MODEL_BASENAME" ]; then
    :
else
    echo "Downloading model with Ultralytics: $MODEL"
    yolo predict model="$MODEL" source='https://ultralytics.com/images/bus.jpg' imgsz="$INFER_W" save=False >/dev/null

    FOUND="$(find . -maxdepth 4 -name "$MODEL_BASENAME" | head -n1 || true)"
    if [ -z "$FOUND" ]; then
        FOUND="$(find "$HOME" -name "$MODEL_BASENAME" 2>/dev/null | head -n1 || true)"
    fi
    if [ -z "$FOUND" ]; then
        echo "Could not find downloaded model: $MODEL_BASENAME"
        exit 1
    fi
    cp -f "$FOUND" "models/$MODEL_BASENAME"
fi

echo "Exporting for DeepStream-Yolo:"
echo "  model: models/$MODEL_BASENAME"
echo "  size:  ${INFER_H}x${INFER_W}"

case "$MODEL_STEM" in
    yolo11*)
        python external/DeepStream-Yolo/utils/export_yolo11.py \
            -w "/home/user/deepstream-work/models/$MODEL_BASENAME" \
            -s "$INFER_H" "$INFER_W" \
            --opset 18 \
            --simplify
        ;;
    yolov12*|yolo12*)
        python external/DeepStream-Yolo/utils/export_yolov12.py \
            -w "/home/user/deepstream-work/models/$MODEL_BASENAME" \
            -s "$INFER_H" "$INFER_W" \
            --opset 18 \
            --simplify
        ;;
    yolov8*|yolo8*)
        python external/DeepStream-Yolo/utils/export_yoloV8.py \
            -w "/home/user/deepstream-work/models/$MODEL_BASENAME" \
            -s "$INFER_H" "$INFER_W" \
            --opset 18 \
            --simplify
        ;;
    *)
        echo "Unsupported model family for DeepStream-Yolo exporter: $MODEL_STEM"
        echo "Supported by this script: yolo11*, yolo12*/yolov12*, yolov8*/yolo8*"
        exit 1
        ;;
esac

ONNX="models/${MODEL_STEM}.onnx"

if [ ! -f "$ONNX" ]; then
    FOUND_ONNX="$(find models . -maxdepth 4 -name "${MODEL_STEM}.onnx" | head -n1 || true)"
    if [ -n "$FOUND_ONNX" ]; then
        cp -f "$FOUND_ONNX" "$ONNX"
    fi
fi

if [ ! -f "$ONNX" ]; then
    echo "Export failed: missing $ONNX"
    exit 1
fi

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

rm -f labels.txt

echo
echo "ONNX check:"
python - <<PY
import onnx

path = "$ONNX"
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

rm -f "models/${MODEL_STEM}"*.engine "models/${MODEL_STEM}.onnx"*.engine

ENGINE="${ONNX}_b1_gpu0_fp16.engine"
INFER_CONFIG="configs/config_infer_primary_${MODEL_STEM}.txt"
APP_CONFIG="configs/dtc_d4_${MODEL_STEM}.txt"

cat > "$INFER_CONFIG" <<EOF_INFER
[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=/home/user/deepstream-work/${ONNX}
model-engine-file=/home/user/deepstream-work/${ENGINE}
labelfile-path=/home/user/deepstream-work/models/coco_labels.txt
batch-size=1
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0

# DeepStream-Yolo parser.
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=/home/user/deepstream-work/lib/libnvdsinfer_custom_impl_Yolo.so
output-blob-names=output

# Robust bbox geometry:
# streammux size == ONNX input size, so no letterbox/pad transform is needed.
maintain-aspect-ratio=0
symmetric-padding=0

cluster-mode=2

[class-attrs-all]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
topk=300
EOF_INFER

cat > "$APP_CONFIG" <<EOF_APP
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[tiled-display]
enable=0
rows=1
columns=1
width=${INFER_W}
height=${INFER_H}
gpu-id=0
nvbuf-memory-type=0

[source0]
enable=1
type=3
uri=file:///home/user/deepstream-work/${STREAM}
num-sources=1
gpu-id=0
cudadec-memtype=0

[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=40000
width=${INFER_W}
height=${INFER_H}
enable-padding=0
nvbuf-memory-type=0
live-source=0

[primary-gie]
enable=1
gpu-id=0
batch-size=1
gie-unique-id=1
nvbuf-memory-type=0
config-file=$(basename "$INFER_CONFIG")

[osd]
enable=1
gpu-id=0
border-width=3
text-size=15
text-color=1;1;1;1
text-bg-color=0.3;0.3;0.3;1
font=Serif
show-clock=0
nvbuf-memory-type=0

[sink0]
enable=1
type=2
sync=0
gpu-id=0
nvbuf-memory-type=0

[sink1]
enable=0
type=1
sync=0

[tests]
file-loop=0
EOF_APP

echo
echo "Done."
echo "PT:           models/$MODEL_BASENAME"
echo "ONNX:         $ONNX"
echo "Engine:       $ENGINE"
echo "Infer config: $INFER_CONFIG"
echo "App config:   $APP_CONFIG"
echo "Labels:       models/coco_labels.txt"
echo
echo "Run:"
echo "  deepstream-app -c $APP_CONFIG"
