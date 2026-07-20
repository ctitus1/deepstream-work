"""Annotated-video recording helpers.

``pipeline.build_pipeline()`` uses these to name mp4 outputs and to pick the
best available H.264 encoder. The recording branch itself is built in
``pipeline.py``; this module keeps naming and encoder policy out of the graph
builder.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from .paths import PROJECT_DIR, resolve_project_path

OUTPUTS_DIR = PROJECT_DIR / "outputs"
MAX_TAG_LEN = 24


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
