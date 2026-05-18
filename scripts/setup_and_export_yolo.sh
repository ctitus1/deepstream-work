#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
SIZE="${2:-640}"

if [ -z "$MODEL" ]; then
    echo "Usage:"
    echo "  $0 <model.pt|model-name> [size]"
    echo
    echo "Examples:"
    echo "  $0 yolo11n.pt 640"
    echo "  $0 yolo11s.pt 640"
    echo "  $0 /path/to/custom.pt 640"
    exit 1
fi

cd "$(dirname "$0")/.."

mkdir -p models external

if [ ! -d .venv-yolo ]; then
    echo "Creating YOLO export virtual environment..."
    python3 -m venv .venv-yolo
fi

source .venv-yolo/bin/activate

python -m pip install --upgrade pip setuptools wheel

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

# If user passed a local path, copy it into models/.
# If user passed yolo11n.pt etc. and it does not exist, let Ultralytics download it first.
if [ -f "$MODEL" ]; then
    cp -f "$MODEL" "models/$MODEL_BASENAME"
elif [ -f "models/$MODEL_BASENAME" ]; then
    :
else
    echo "Downloading model with Ultralytics: $MODEL"
    yolo predict model="$MODEL" source='https://ultralytics.com/images/bus.jpg' imgsz="$SIZE" save=False >/dev/null
    FOUND="$(find . -maxdepth 3 -name "$MODEL_BASENAME" | head -n1 || true)"
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
echo "  size:  $SIZE"

case "$MODEL_STEM" in
    yolo11*)
        python external/DeepStream-Yolo/utils/export_yolo11.py \
            -w "/home/user/deepstream-work/models/$MODEL_BASENAME" \
            -s "$SIZE" \
            --opset 18 \
            --simplify
        ;;
    yolov12*|yolo12*)
        python external/DeepStream-Yolo/utils/export_yolov12.py \
            -w "/home/user/deepstream-work/models/$MODEL_BASENAME" \
            -s "$SIZE" \
            --opset 18 \
            --simplify
        ;;
    yolov8*|yolo8*)
        python external/DeepStream-Yolo/utils/export_yoloV8.py \
            -w "/home/user/deepstream-work/models/$MODEL_BASENAME" \
            -s "$SIZE" \
            --opset 18 \
            --simplify
        ;;
    *)
        echo "Unsupported model family for DeepStream-Yolo exporter: $MODEL_STEM"
        echo "Supported by this script: yolo11*, yolo12*/yolov12*, yolov8*/yolo8*"
        exit 1
        ;;
esac

# Export scripts write ONNX beside the .pt file.
ONNX="models/${MODEL_STEM}.onnx"

if [ ! -f "$ONNX" ]; then
    FOUND_ONNX="$(find models . -maxdepth 3 -name "${MODEL_STEM}.onnx" | head -n1 || true)"
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

echo
echo "Done."
echo "PT:     models/$MODEL_BASENAME"
echo "ONNX:   $ONNX"
echo "Engine: $ENGINE"
echo "Labels: models/coco_labels.txt"
echo
echo "For nvinfer, use:"
echo "  onnx-file=/home/user/deepstream-work/$ONNX"
echo "  model-engine-file=/home/user/deepstream-work/$ENGINE"
echo "  output-blob-names=output"
