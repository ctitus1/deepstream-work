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

STREAMS_DIR = PROJECT_DIR / "streams"
DEFAULT_MEDIA = STREAMS_DIR / "dtc-d4-trimmed.mp4"
DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8555/dtc-d4-trimmed"
DEFAULT_STREAM = DEFAULT_RTSP_URL
SETUP_SCRIPT = SCRIPTS_DIR / "setup_and_export_yolo.sh"
INJURY_SETUP_SCRIPT = SCRIPTS_DIR / "setup_injury_model.sh"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def available_media() -> list[str]:
    """File names currently sitting in ``streams/``."""
    if not STREAMS_DIR.is_dir():
        return []
    return sorted(
        entry.name
        for entry in STREAMS_DIR.iterdir()
        if entry.is_file() and not entry.name.startswith(".")
    )


def missing_media_message(path: str | Path) -> str:
    """Explain a missing stream in terms of what this checkout actually holds.

    ``streams/`` is gitignored user media, so the documented default is simply
    absent on any machine that is not the author's. Naming the missing file and
    listing the alternatives beats letting it fall through to an opaque
    GStreamer failure several elements later.
    """
    if not STREAMS_DIR.is_dir():
        detail = f"{STREAMS_DIR} does not exist"
    else:
        names = available_media()
        detail = (
            f"{STREAMS_DIR} contains: {', '.join(names)}" if names else f"{STREAMS_DIR} is empty"
        )

    return (
        f"Stream file not found: {path}\n"
        f"{detail}\n"
        "streams/ is gitignored, so its contents differ per checkout; "
        "point at one of the files above."
    )
