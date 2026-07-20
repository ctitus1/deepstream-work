import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

LEGACY_YOLO_VENV = PROJECT_DIR / ".venv-yolo"


def yolo_venv_dir() -> Path:
    """Export virtualenv for the interpreter that is asking.

    The venv lives in the bind-mounted project directory, so the host and the
    DeepStream container both see the same path -- but they run different
    interpreters (3.12 here, 3.10 in the DS 7.1 image), and a venv only works
    with the version that built it. Sharing one directory meant each side found
    the other's unusable, deleted it, and re-downloaded torch and the whole
    nvidia-cu12 wheel set. Keying the directory by version lets both coexist.

    A pre-existing ``.venv-yolo`` is still honored when it happens to match the
    running interpreter, so an already-working setup is not thrown away.
    """
    if (LEGACY_YOLO_VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}").is_dir():
        return LEGACY_YOLO_VENV
    return PROJECT_DIR / f".venv-yolo-{sys.version_info.major}.{sys.version_info.minor}"

MODELS_DIR = PROJECT_DIR / "models"
CONFIGS_DIR = PROJECT_DIR / "configs"
GENERATED_CONFIG_DIR = CONFIGS_DIR / "generated"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
SETUP_DIR = SCRIPTS_DIR / "setup"
LIB_DIR = PROJECT_DIR / "lib"
LABELS_SOURCE_PATH = PROJECT_DIR / "labels" / "coco_labels.txt"
LABELS_PATH = MODELS_DIR / "coco_labels.txt"
CUSTOM_LIB_PATH = LIB_DIR / "libnvdsinfer_custom_impl_Yolo.so"
YOLO_VENV_DIR = yolo_venv_dir()
YOLO_PYTHON = YOLO_VENV_DIR / "bin" / "python3"

STREAMS_DIR = PROJECT_DIR / "streams"
SETUP_SCRIPT = SETUP_DIR / "yolo_export.sh"
INJURY_SETUP_SCRIPT = SETUP_DIR / "injury_model.sh"

# Shipped in models/ and small enough that its engine builds in seconds, so the
# zero-argument path works on a fresh checkout. Override with --model.
DEFAULT_MODEL = "yolo12n.pt"
DEFAULT_ASSESSMENT_MODEL = "models/injury.pt"

DEFAULT_RTSP_HOST = "127.0.0.1"
DEFAULT_RTSP_PORT = 8555
# Fallback only. Every default below prefers whatever media this checkout
# actually has; see default_media().
FALLBACK_MEDIA_NAME = "dtc-d4-trimmed.mp4"


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


def default_media() -> Path:
    """The video the zero-argument path should use.

    ``streams/`` is gitignored user media, so no hardcoded name is right on more
    than one machine. When the directory holds exactly one video that is
    unambiguously the one meant; anything else keeps the historical name so the
    failure is reported by ``missing_media_message`` rather than by silently
    picking an arbitrary file out of several.
    """
    names = available_media()
    if len(names) == 1:
        return STREAMS_DIR / names[0]
    return STREAMS_DIR / FALLBACK_MEDIA_NAME


def default_rtsp_mount() -> str:
    """RTSP mount name derived from the default media.

    The server and every client default derive the mount from the same file, so
    they agree without anyone passing a name.
    """
    return default_media().stem


def default_rtsp_url() -> str:
    return f"rtsp://{DEFAULT_RTSP_HOST}:{DEFAULT_RTSP_PORT}/{default_rtsp_mount()}"


# Resolved once at import so argparse defaults stay plain strings. RTSP is the
# preferred input path for both apps, matching the live pipeline's pacing and
# drop behavior.
DEFAULT_MEDIA = default_media()
DEFAULT_RTSP_URL = default_rtsp_url()
DEFAULT_STREAM = DEFAULT_RTSP_URL


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
