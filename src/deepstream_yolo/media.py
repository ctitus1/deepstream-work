"""What the pipeline reads and what it writes.

Two halves of one concern, kept together because every caller that resolves an
input also decides where output goes:

  * **Input** -- ``resolve_stream_source`` turns whatever came off the command
    line (an RTSP URL, a ``file://`` URI, a relative path) into a
    ``StreamSource`` the graph builder can use without re-parsing it.
  * **Output** -- ``resolve_record_path`` names the annotated mp4, and
    ``select_encoder`` picks the encoder that writes it.

``pipeline.build_pipeline()`` builds the recording branch itself; the naming and
encoder policy live here so they stay out of the graph builder.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from .paths import PROJECT_DIR, missing_media_message, resolve_project_path

OUTPUTS_DIR = PROJECT_DIR / "outputs"
MAX_TAG_LEN = 24


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamSource:
    raw: str
    uri: str
    path: Path | None

    @property
    def is_rtsp(self) -> bool:
        return urlparse(self.uri).scheme.lower() in {"rtsp", "rtsps"}

    @property
    def display(self) -> str:
        if self.path is None:
            return self.uri
        try:
            return str(self.path.relative_to(PROJECT_DIR))
        except ValueError:
            return str(self.path)


def local_source(raw: str, path: Path) -> StreamSource:
    """Build a file-backed source, failing early if the media is not here."""
    if not path.is_file():
        raise FileNotFoundError(missing_media_message(path))

    return StreamSource(raw=raw, uri=path.resolve().as_uri(), path=path)


def resolve_stream_source(stream: str | Path) -> StreamSource:
    raw = str(stream)
    parsed = urlparse(raw)

    if parsed.scheme == "file":
        return local_source(raw, Path(unquote(parsed.path)))

    if parsed.scheme:
        return StreamSource(raw=raw, uri=raw, path=None)

    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    return local_source(raw, path)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderChoice:
    """An H.264 encoder plus the caps and properties the record branch needs."""

    factory: str
    caps: str
    properties: dict = field(default_factory=dict)


# Ordered best-first. Only nvv4l2h264enc ships in the DeepStream image, and it
# keeps the annotated frame in NVMM; software fallbacks (nvh264enc, x264enc,
# openh264enc) are all absent, so select_encoder's error is the real other path.
ENCODER_PREFERENCES = (
    EncoderChoice(
        factory="nvv4l2h264enc",
        caps="video/x-raw(memory:NVMM), format=NV12",
        properties={
            "bitrate": 8000000,
            "iframeinterval": 30,
            "idrinterval": 30,
            "insert-sps-pps": True,
        },
    ),
)


def select_encoder() -> EncoderChoice:
    """Return the first installed encoder from ``ENCODER_PREFERENCES``."""
    for choice in ENCODER_PREFERENCES:
        if Gst.ElementFactory.find(choice.factory):
            return choice

    tried = ", ".join(choice.factory for choice in ENCODER_PREFERENCES)
    raise RuntimeError(f"No H.264 encoder available for recording; tried: {tried}")


def stream_tag(uri: str) -> str:
    """Filesystem-safe short name for a stream URI."""
    parsed = urlparse(uri)
    if parsed.scheme:
        source = Path(parsed.path)
        name = source.stem or source.name or parsed.netloc or parsed.scheme
    else:
        name = Path(uri).stem or Path(uri).name

    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe or "stream")[:MAX_TAG_LEN]


def default_record_path(stream_uri: str) -> Path:
    """Timestamped ``record_<UTC>_<streamtag>.mp4`` under ``outputs/``."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return OUTPUTS_DIR / f"record_{stamp}_{stream_tag(stream_uri)}.mp4"


def resolve_record_path(path: str | Path | None, stream_uri: str) -> Path:
    """Resolve an explicit ``--record PATH`` or auto-name one, creating the parent dir."""
    target = resolve_project_path(path) if path else default_record_path(stream_uri)
    if target.suffix.lower() != ".mp4":
        target = target.with_suffix(".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
