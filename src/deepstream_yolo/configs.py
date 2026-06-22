import shutil
from pathlib import Path

from .paths import CUSTOM_LIB_PATH, GENERATED_CONFIG_DIR, LABELS_PATH, LABELS_SOURCE_PATH


def ensure_labels_file() -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LABELS_SOURCE_PATH.exists():
        shutil.copy2(LABELS_SOURCE_PATH, LABELS_PATH)


def write_infer_config(
    config_path: Path,
    onnx_path: Path,
    engine_path: Path,
    conf: float,
    *,
    maintain_aspect_ratio: int = 1,
    symmetric_padding: int = 1,
) -> None:
    ensure_labels_file()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file={onnx_path}
model-engine-file={engine_path}
labelfile-path={LABELS_PATH}
batch-size=1
network-mode=2
num-detected-classes=80
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
