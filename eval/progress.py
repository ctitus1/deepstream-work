"""A progress bar with an ETA, for the long passes over a video.

Written to stderr, which is where these tools already put their diagnostics and
which is never the raw-frame stream stdout carries. The bar redraws in place
with a carriage return; that is a byte the terminal interprets, so it still
animates when the work is happening inside a container whose own stderr is a
pipe rather than a tty -- which is the normal case here and the one where
gating on ``isatty()`` would switch the bar off exactly when it is wanted.

Total unknown is a supported state: a GStreamer pipeline does not have to know
how many frames it will be handed. Without a total it reports count and rate
and omits the bar and the estimate rather than inventing either.
"""

from __future__ import annotations

import sys
import time


def _clock(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds > 359999:  # NaN-safe
        return "--:--"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class Progress:
    """Count work as it happens and estimate what is left.

    The rate used for the estimate is measured over a trailing window rather
    than over the whole run. These passes do not have a uniform speed -- a
    tracker's first frames are cheap because nothing has been detected yet, and
    a lag-24 variant does nothing at all until it has 24 frames of history -- so
    an average-since-start ETA stays wrong for a long time after the work
    settles.
    """

    def __init__(self, total: int = 0, label: str = "", width: int = 28, every: float = 0.25):
        self.total = int(total or 0)
        self.label = label
        self.width = width
        self.every = every
        self.count = 0
        self.start = time.perf_counter()
        self._last_draw = 0.0
        self._window: list[tuple[float, int]] = [(self.start, 0)]
        self._drawn = False

    def update(self, step: int = 1) -> None:
        self.count += step
        now = time.perf_counter()
        if now - self._last_draw < self.every:
            return
        self._last_draw = now

        self._window.append((now, self.count))
        # Roughly the last five seconds, and always at least two samples.
        while len(self._window) > 2 and now - self._window[0][0] > 5.0:
            self._window.pop(0)

        self._draw(now)

    def _rate(self, now: float) -> float:
        first_time, first_count = self._window[0]
        span = now - first_time
        return (self.count - first_count) / span if span > 1e-6 else 0.0

    def _draw(self, now: float) -> None:
        rate = self._rate(now)
        parts = []
        if self.label:
            parts.append(self.label)

        if self.total > 0:
            done = min(self.count, self.total)
            frac = done / self.total
            filled = int(self.width * frac)
            parts.append("[" + "#" * filled + "-" * (self.width - filled) + "]")
            parts.append(f"{frac * 100:5.1f}%")
            parts.append(f"{done}/{self.total}")
            remaining = (self.total - done) / rate if rate > 1e-6 else -1.0
            parts.append(f"{rate:5.1f} fps")
            parts.append(f"eta {_clock(remaining)}")
        else:
            parts.append(f"{self.count} frames")
            parts.append(f"{rate:5.1f} fps")

        line = "  " + "  ".join(parts)
        # Pad to overwrite whatever the previous, possibly longer, line left.
        sys.stderr.write("\r" + line.ljust(96)[:96])
        sys.stderr.flush()
        self._drawn = True

    def close(self, note: str = "") -> None:
        elapsed = time.perf_counter() - self.start
        rate = self.count / elapsed if elapsed > 1e-6 else 0.0
        parts = [self.label] if self.label else []
        parts.append(f"{self.count} frames")
        parts.append(f"in {_clock(elapsed)}")
        parts.append(f"{rate:5.1f} fps")
        if note:
            parts.append(note)
        line = "  " + "  ".join(parts)
        if self._drawn:
            sys.stderr.write("\r" + line.ljust(96)[:96] + "\n")
        else:
            sys.stderr.write(line + "\n")
        sys.stderr.flush()


def frame_count(path) -> int:
    """Frames in a video, or 0 when it cannot be established cheaply.

    Deliberately tolerant: the count is only used to draw a bar, so a container
    that lies about it or a backend that will not answer costs a nicer display
    and nothing else.
    """
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return 0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        return max(0, total)
    except Exception:
        return 0
