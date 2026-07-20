#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Single model size parameter.
# The script detects the input stream aspect ratio and converts this one dimension
# into a stride-safe WIDTH x HEIGHT for YOLO/DeepStream.
#
# Examples:
#   ./scripts/setup_and_export_yolo.sh yolo11n.pt 640
#   ./scripts/setup_and_export_yolo.sh yolo12x.pt 640
#   ./scripts/setup_and_export_yolo.sh yolo12x.pt 1920
#
# The long edge is rounded UP to the stride, so the result does not preserve
# the source aspect exactly; nvstreammux letterboxes to make up the difference.
# For a 1920x1080 stream:
#   640  -> 640x384
#   1920 -> 1920x1088
MODEL_SIZE="${2:-1920}"

# Stream used to derive aspect ratio.
# Override as third arg if needed:
#   ./scripts/setup_and_export_yolo.sh yolo12x.pt 640 streams/other.mp4
STREAM="${3:-streams/dtc-d4-trimmed.mp4}"

STRIDE=32
DEEPSTREAM_YOLO_REF="${DEEPSTREAM_YOLO_REF:-2894babce8e75c49115dbe0c7b516289ed853565}"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-configs/generated}"
VENV_DIR=".venv-yolo"

if [ -z "$MODEL" ]; then
    echo "Usage:"
    echo "  $0 <model.pt|model-name> [model_size] [stream]"
    echo
    echo "Examples:"
    echo "  $0 yolo11n.pt 640"
    echo "  $0 yolo12x.pt 640 streams/dtc-d4-trimmed.mp4"
    echo "  $0 yolo12x.pt 1920 streams/dtc-d4-trimmed.mp4"
    exit 1
fi

cd "$ROOT_DIR"

mkdir -p models external "$GENERATED_CONFIG_DIR" lib outputs
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ROOT_DIR/outputs/ultralytics-config}"
mkdir -p "$YOLO_CONFIG_DIR"

safe_copy() {
    local src="$1"
    local dst="$2"

    mkdir -p "$(dirname "$dst")"

    if [ -e "$src" ] && [ -e "$dst" ] && [ "$(readlink -f "$src")" = "$(readlink -f "$dst")" ]; then
        return 0
    fi

    cp -f "$src" "$dst"
}

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$ROOT_DIR" "$1" ;;
    esac
}

clone_deepstream_yolo() {
    if [ ! -d external/DeepStream-Yolo ]; then
        echo "Cloning DeepStream-Yolo..."
        git clone https://github.com/marcoslucianops/DeepStream-Yolo.git external/DeepStream-Yolo
    fi

    git -C external/DeepStream-Yolo checkout "$DEEPSTREAM_YOLO_REF" >/dev/null
}

install_labels() {
    # The DeepStream-Yolo exporter writes labels.txt from the class names baked
    # into the checkpoint, so it is the only source that is right for a
    # fine-tuned model. For a stock COCO checkpoint it is identical to the
    # checked-in list, which stays the fallback when the exporter emits nothing.
    if [ -s labels.txt ]; then
        cp -f labels.txt models/coco_labels.txt
        return
    fi

    if [ ! -f labels/coco_labels.txt ]; then
        echo "Missing labels/coco_labels.txt and the exporter produced no labels.txt"
        exit 1
    fi

    cp -f labels/coco_labels.txt models/coco_labels.txt
}

ensure_export_venv() {
    if [ ! -x "$VENV_DIR/bin/python3" ] || ! "$VENV_DIR/bin/python3" -m pip --version >/dev/null 2>&1; then
        echo "Creating YOLO export virtual environment..."
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi

    PYTHON_BIN="$VENV_DIR/bin/python3"
    YOLO_BIN="$VENV_DIR/bin/yolo"
}


detect_dims() {
    local stream="$1"
    local stream_path
    stream_path="$(absolute_path "$stream")"

    if [ -n "${SOURCE_WIDTH:-}" ] && [ -n "${SOURCE_HEIGHT:-}" ]; then
        echo "$SOURCE_WIDTH $SOURCE_HEIGHT"
        return
    fi

    if command -v gst-discoverer-1.0 >/dev/null 2>&1; then
        local discover_target=""
        case "$stream" in
            *://*) discover_target="$stream" ;;
            *) [ -f "$stream_path" ] && discover_target="$stream_path" ;;
        esac

        if [ -n "$discover_target" ]; then
            local out
            out="$(gst-discoverer-1.0 "$discover_target" 2>/dev/null || true)"
            local w h
            w="$(printf '%s\n' "$out" | awk '/Width:/ {print $2; exit}')"
            h="$(printf '%s\n' "$out" | awk '/Height:/ {print $2; exit}')"
            if [ -n "$w" ] && [ -n "$h" ]; then
                echo "$w $h"
                return
            fi
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

ensure_export_venv

# Retried because torch pulls several 150-500 MB nvidia-*-cu12 wheels whose
# downloads are intermittently corrupted on some networks; pip surfaces that as
# a hash mismatch even though nothing here pins hashes. See the longer note in
# scripts/setup_yolo_export_env.sh.
pip_install_retry() {
    local attempt=1
    while true; do
        if "$PYTHON_BIN" -m pip install --timeout 60 --retries 5 "$@"; then
            return 0
        fi
        if [ "$attempt" -ge "${PIP_ATTEMPTS:-4}" ]; then
            echo "pip install failed after ${attempt} attempts: $*" >&2
            return 1
        fi
        echo "pip install attempt ${attempt} failed; retrying..." >&2
        attempt=$((attempt + 1))
    done
}

pip_install_retry --upgrade 'pip' 'setuptools<82' wheel

pip_install_retry -r requirements/yolo-export.txt
if [ ! -x "$YOLO_BIN" ]; then
    echo "Missing YOLO CLI after dependency install: $YOLO_BIN"
    exit 1
fi

clone_deepstream_yolo

MODEL_BASENAME="$(basename "$MODEL")"
MODEL_STEM="${MODEL_BASENAME%.pt}"

if [ -f "$MODEL" ]; then
    safe_copy "$MODEL" "models/$MODEL_BASENAME"
elif [ -f "models/$MODEL_BASENAME" ]; then
    :
else
    echo "Downloading model with Ultralytics: $MODEL"
    "$YOLO_BIN" predict model="$MODEL" source='https://ultralytics.com/images/bus.jpg' imgsz="$INFER_W" save=False >/dev/null

    FOUND="$(find . -maxdepth 4 -name "$MODEL_BASENAME" | head -n1 || true)"
    if [ -z "$FOUND" ]; then
        FOUND="$(find "$HOME" -name "$MODEL_BASENAME" 2>/dev/null | head -n1 || true)"
    fi
    if [ -z "$FOUND" ]; then
        echo "Could not find downloaded model: $MODEL_BASENAME"
        exit 1
    fi
    safe_copy "$FOUND" "models/$MODEL_BASENAME"
fi

echo "Exporting for DeepStream-Yolo:"
echo "  model: models/$MODEL_BASENAME"
echo "  size:  ${INFER_H}x${INFER_W}"

case "$MODEL_STEM" in
    yolo11*)
        "$PYTHON_BIN" external/DeepStream-Yolo/utils/export_yolo11.py \
            -w "$ROOT_DIR/models/$MODEL_BASENAME" \
            -s "$INFER_H" "$INFER_W" \
            --opset 18 \
            --simplify
        ;;
    yolov12*|yolo12*)
        "$PYTHON_BIN" external/DeepStream-Yolo/utils/export_yolov12.py \
            -w "$ROOT_DIR/models/$MODEL_BASENAME" \
            -s "$INFER_H" "$INFER_W" \
            --opset 18 \
            --simplify
        ;;
    yolov8*|yolo8*)
        "$PYTHON_BIN" external/DeepStream-Yolo/utils/export_yoloV8.py \
            -w "$ROOT_DIR/models/$MODEL_BASENAME" \
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
        safe_copy "$FOUND_ONNX" "$ONNX"
    fi
fi

if [ ! -f "$ONNX" ]; then
    echo "Export failed: missing $ONNX"
    exit 1
fi

install_labels

# install_labels has already consumed it; this only clears the stray copy the
# exporter leaves in the repo root.
rm -f labels.txt

echo
echo "ONNX check:"
"$PYTHON_BIN" - <<PY
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

# Only the engine for the artifact being regenerated. A bare
# "models/${MODEL_STEM}"*.engine also matches the resolution-tagged engines
# model_cache builds (yolo12x_640_640x384...engine, yolo12x_1280_1280x736...),
# so alternating --long-side values forced a full TensorRT rebuild every run.
rm -f "models/${MODEL_STEM}.onnx"*.engine

ENGINE="${ONNX}_b1_gpu0_fp16.engine"
INFER_CONFIG="${GENERATED_CONFIG_DIR}/config_infer_primary_${MODEL_STEM}.txt"
APP_CONFIG="${GENERATED_CONFIG_DIR}/deepstream_${MODEL_STEM}.txt"
INFER_CONFIG_ABS="$(absolute_path "$INFER_CONFIG")"
STREAM_ABS="$(absolute_path "$STREAM")"
ONNX_ABS="$(absolute_path "$ONNX")"
ENGINE_ABS="$(absolute_path "$ENGINE")"
# models/coco_labels.txt always holds the most recent export, so a config that
# points at it starts describing a different model as soon as another one is
# exported. Keep a per-model copy and point this config at that instead.
MODEL_LABELS="models/${MODEL_STEM}.labels.txt"
cp -f models/coco_labels.txt "$MODEL_LABELS"

LABELS_ABS="$(absolute_path "$MODEL_LABELS")"
CUSTOM_LIB_ABS="$(absolute_path "lib/libnvdsinfer_custom_impl_Yolo.so")"

# nvinfer's num-detected-classes has to match the labels just installed, or a
# fine-tuned model announces a class count its own engine does not produce.
NUM_CLASSES="$(grep -c '[^[:space:]]' "$MODEL_LABELS" || true)"
if [ "$NUM_CLASSES" -lt 1 ]; then
    echo "No class names in $MODEL_LABELS"
    exit 1
fi

cat > "$INFER_CONFIG" <<EOF_INFER
[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=${ONNX_ABS}
model-engine-file=${ENGINE_ABS}
labelfile-path=${LABELS_ABS}
batch-size=1
network-mode=2
num-detected-classes=${NUM_CLASSES}
interval=0
gie-unique-id=1
process-mode=1
network-type=0

# DeepStream-Yolo parser.
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=${CUSTOM_LIB_ABS}
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
uri=file://${STREAM_ABS}
num-sources=1
gpu-id=0
cudadec-memtype=0

[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=40000
width=${INFER_W}
height=${INFER_H}
# Letterbox instead of stretching. The model size is the long edge rounded UP
# to the stride, so it does not keep the source aspect: a 3840x2160 source at
# long-side 640 gives 640x384 (5:3), not 640x360 (16:9). With padding disabled
# nvstreammux squeezed the frame ~6.7% vertically before inference, against a
# model trained on letterboxed images. This matches the Python runtime path,
# which letterboxes via maintain-aspect-ratio=1/symmetric-padding=1.
enable-padding=1
nvbuf-memory-type=0
live-source=0

[primary-gie]
enable=1
gpu-id=0
batch-size=1
gie-unique-id=1
nvbuf-memory-type=0
config-file=${INFER_CONFIG_ABS}

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
echo "Labels:       ${MODEL_LABELS} (${NUM_CLASSES} classes)"
echo
echo "Run:"
echo "  deepstream-app -c $APP_CONFIG"
