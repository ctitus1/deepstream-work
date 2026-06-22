from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_DIR / "models"
CONFIGS_DIR = PROJECT_DIR / "configs"
GENERATED_CONFIG_DIR = CONFIGS_DIR / "generated"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
LIB_DIR = PROJECT_DIR / "lib"
LABELS_SOURCE_PATH = PROJECT_DIR / "labels" / "coco_labels.txt"
LABELS_PATH = MODELS_DIR / "coco_labels.txt"
CUSTOM_LIB_PATH = LIB_DIR / "libnvdsinfer_custom_impl_Yolo.so"
YOLO_PYTHON = PROJECT_DIR / ".venv-yolo" / "bin" / "python3"

DEFAULT_STREAM = PROJECT_DIR / "streams" / "dtc-d4-trimmed.mp4"
SETUP_SCRIPT = SCRIPTS_DIR / "setup_and_export_yolo.sh"
INJURY_SETUP_SCRIPT = SCRIPTS_DIR / "setup_injury_model.sh"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate
