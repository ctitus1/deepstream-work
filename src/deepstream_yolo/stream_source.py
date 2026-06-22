from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .paths import PROJECT_DIR


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


def resolve_stream_source(stream: str | Path) -> StreamSource:
    raw = str(stream)
    parsed = urlparse(raw)

    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        return StreamSource(raw=raw, uri=path.resolve().as_uri(), path=path)

    if parsed.scheme:
        return StreamSource(raw=raw, uri=raw, path=None)

    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    return StreamSource(raw=raw, uri=path.resolve().as_uri(), path=path)
