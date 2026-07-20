import shutil
from pathlib import Path

from .paths import CUSTOM_LIB_PATH, GENERATED_CONFIG_DIR, LABELS_PATH, LABELS_SOURCE_PATH

CLIP_INPUT_SIZE = 336
INJURY_HEADS = (
    "severe_hemorrhage",
    "respiratory_distress",
    "trauma_head",
    "trauma_torso",
    "trauma_upper_ext",
    "trauma_lower_ext",
    "alertness_ocular",
    "person_type",
)
INJURY_CLASS_COUNTS = {
    "severe_hemorrhage": 2,
    "respiratory_distress": 2,
    "trauma_head": 2,
    "trauma_torso": 2,
    "trauma_upper_ext": 3,
    "trauma_lower_ext": 3,
    "alertness_ocular": 3,
    "person_type": 2,
}


DEFAULT_NUM_DETECTED_CLASSES = 80


def ensure_labels_file() -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LABELS_PATH.exists():
        # scripts/setup_and_export_yolo.sh installs the exported model's own
        # class names here. Re-copying the checked-in COCO list on every run
        # would put 80 wrong names back on top of a fine-tuned model.
        return
    if LABELS_SOURCE_PATH.exists():
        shutil.copy2(LABELS_SOURCE_PATH, LABELS_PATH)


def num_detected_classes(labels_path: Path | None = None) -> int:
    """Class count for nvinfer, read from the labels this model will deploy with.

    Hardcoding 80 makes ``num-detected-classes`` disagree with the engine's
    real output tensor for any model that is not stock COCO.
    """
    path = labels_path or LABELS_PATH
    if path.is_file():
        names = [line for line in path.read_text().splitlines() if line.strip()]
        if names:
            return len(names)
    return DEFAULT_NUM_DETECTED_CLASSES


def write_infer_config(
    config_path: Path,
    onnx_path: Path,
    engine_path: Path,
    conf: float,
    *,
    labels_path: Path | None = None,
    maintain_aspect_ratio: int = 1,
    symmetric_padding: int = 1,
) -> None:
    # models/coco_labels.txt holds whatever the most recent export installed, so
    # it is not safe to point a config at it: exporting a 3-class model and then
    # running a stock 80-class one would give the second model the first one's
    # names and class count. Callers pass the per-model snapshot instead.
    ensure_labels_file()
    labels = labels_path or LABELS_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file={onnx_path}
model-engine-file={engine_path}
labelfile-path={labels}
batch-size=1
network-mode=2
num-detected-classes={num_detected_classes(labels)}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path={CUSTOM_LIB_PATH}
output-blob-names=output
maintain-aspect-ratio={maintain_aspect_ratio}
symmetric-padding={symmetric_padding}
cluster-mode=2

[class-attrs-all]
pre-cluster-threshold={conf}
nms-iou-threshold=0.45
topk=300
"""
    )


def generated_config_path(name: str) -> Path:
    return GENERATED_CONFIG_DIR / name


def write_assessment_config(
    config_path: Path,
    onnx_path: Path,
    engine_path: Path,
    batch_size: int,
    *,
    gie_id: int = 2,
    operate_on_gie_id: int = 1,
    operate_on_class_ids: str = "0",
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    output_blob_names = ";".join(INJURY_HEADS)
    config_path.write_text(
        f"""[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file={onnx_path}
model-engine-file={engine_path}
batch-size={batch_size}
network-mode=2
gie-unique-id={gie_id}
process-mode=2
network-type=100
output-tensor-meta=1
operate-on-gie-id={operate_on_gie_id}
operate-on-class-ids={operate_on_class_ids}
classifier-async-mode=0
infer-dims=3;{CLIP_INPUT_SIZE};{CLIP_INPUT_SIZE}
output-blob-names={output_blob_names}
maintain-aspect-ratio=1
symmetric-padding=1
input-object-min-width=8
input-object-min-height=8
"""
    )
