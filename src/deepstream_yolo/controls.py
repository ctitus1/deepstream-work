"""Parser-app keyboard controls and local-file playback pacing.

``parser_app.py`` uses ``KeyboardControls`` for pause/quit/rate keys and
``RateLimiter`` only for local-file playback. Live RTSP streams remain paced by
the stream clock and late-frame dropping in the GStreamer pipeline.
"""

import sys
import termios
import time
import tty

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst


class RateLimiter:
    def __init__(self, base_fps: float = 30.0, rate: float = 1.0, enabled: bool = True):
        self.base_fps = base_fps
        self.rate = rate
        self.enabled = enabled
        self.last_time = 0.0

    @property
    def max_fps(self) -> float:
        return self.base_fps * self.rate

    def set_rate(self, rate: float) -> None:
        if not self.enabled:
            print("rate controls disabled for live RTSP streams", flush=True)
            return

        self.rate = round(max(0.05, min(8.0, rate)), 2)
        print(f"rate={self.rate:.2f}x max_fps={self.max_fps:.1f}", flush=True)

    def probe(self, _pad, _info, _data):
        if not self.enabled:
            return Gst.PadProbeReturn.OK

        now = time.perf_counter()
        period = 1.0 / self.max_fps

        if self.last_time:
            wait = period - (now - self.last_time)
            if wait > 0:
                time.sleep(wait)

        self.last_time = time.perf_counter()
        return Gst.PadProbeReturn.OK


class KeyboardControls:
    def __init__(self, pipeline, loop, limiter: RateLimiter):
        self.pipeline = pipeline
        self.loop = loop
        self.limiter = limiter
        self.paused = False
        self.old_term = None

    def start(self):
        self.old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        GLib.io_add_watch(sys.stdin, GLib.IO_IN, self.on_key)
        if self.limiter.enabled:
            print(
                "keys: right/up speed up | left/down slow down | r reset 1x | "
                "space pause/play | q quit",
                flush=True,
            )
        else:
            print("keys: space pause/play | q quit", flush=True)

    def stop(self):
        if self.old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_term)

    def on_key(self, _source, _condition):
        key = sys.stdin.read(1)
        if key == "\x1b":
            key += sys.stdin.read(2)

        if key == "q":
            self.loop.quit()
            return False

        if key == " ":
            self.toggle_pause()
        elif key == "r" and self.limiter.enabled:
            self.limiter.set_rate(1.0)
        elif key == "\x1b[C" and self.limiter.enabled:
            self.limiter.set_rate(self.limiter.rate + 0.5)
        elif key == "\x1b[D" and self.limiter.enabled:
            self.limiter.set_rate(self.limiter.rate - 0.5)
        elif key == "\x1b[A" and self.limiter.enabled:
            self.limiter.set_rate(self.limiter.rate + 0.05)
        elif key == "\x1b[B" and self.limiter.enabled:
            self.limiter.set_rate(self.limiter.rate - 0.05)

        return True

    def toggle_pause(self):
        self.paused = not self.paused
        self.pipeline.set_state(Gst.State.PAUSED if self.paused else Gst.State.PLAYING)
        print("paused" if self.paused else "playing", flush=True)
