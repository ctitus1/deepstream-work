import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds


# Frames that never reach the sink are never completed, so their entries would
# otherwise accumulate for the life of the process. That is reachable in normal
# operation: the RTSP display queue is leaky, and --show-assessed-only drops
# frames at the SGIE source pad.
MAX_PENDING_FRAMES = 512

# Stage marks in pipeline order, each paired with the name of the interval that
# ends at it. "assessment" is absent under --no-assessment, so the row is built
# from whichever marks are present rather than from two hand-written variants.
TIMED_STAGES = (
    ("mux", None),
    ("infer", "detect"),
    ("assessment", "assess"),
    ("convert", "convert"),
    ("osd", "osd"),
    ("sink", "sink"),
)
REQUIRED_STAGES = tuple(stage for stage, _ in TIMED_STAGES if stage != "assessment")


def compute_fps(seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return 1.0 / seconds


def stage_row(marks: dict) -> dict | None:
    """Successive stage durations for one frame, or None if it never finished."""
    if not all(stage in marks for stage in REQUIRED_STAGES):
        return None

    present = [(stage, name) for stage, name in TIMED_STAGES if stage in marks]
    row = {
        name: marks[stage] - marks[previous]
        for (previous, _), (stage, name) in zip(present, present[1:])
    }
    row["total"] = marks["sink"] - marks["mux"]
    return row


class TimeLog:
    def __init__(self, fps_interval: float = 1.0, timing_interval: float = 10.0):
        self.fps_interval = fps_interval
        self.timing_interval = timing_interval
        self.last_fps_time = time.perf_counter()
        self.last_timing_time = self.last_fps_time
        self.frames = 0
        self.last_frames = 0
        self.times = {}

    def fps_probe(self, _pad, _info, _data):
        self.frames += 1
        now = time.perf_counter()
        elapsed = now - self.last_fps_time

        if elapsed >= self.fps_interval:
            fps = (self.frames - self.last_frames) / elapsed
            print(f"OUTPUT_FPS {fps:.2f}", flush=True)
            self.last_fps_time = now
            self.last_frames = self.frames

        return Gst.PadProbeReturn.OK

    def mark(self, stage: str):
        def _probe(_pad, info, _data):
            now = time.perf_counter()
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))

            if batch_meta:
                frame_list = batch_meta.frame_meta_list
                while frame_list:
                    frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                    self.times.setdefault(int(frame_meta.frame_num), {})[stage] = now
                    frame_list = frame_list.next

                self._trim()

            if stage == "sink":
                self._print_timing(now)

            return Gst.PadProbeReturn.OK

        return _probe

    def _trim(self) -> None:
        # Frame numbers increase, so the lowest pending entries are the ones
        # that were dropped upstream and will never complete.
        excess = len(self.times) - MAX_PENDING_FRAMES
        if excess <= 0:
            return
        for frame_num in sorted(self.times)[:excess]:
            del self.times[frame_num]

    def _print_timing(self, now: float) -> None:
        if now - self.last_timing_time < self.timing_interval:
            return

        rows = []
        for frame_num, marks in list(self.times.items()):
            row = stage_row(marks)
            if row is not None:
                rows.append(row)
                del self.times[frame_num]

        if not rows:
            # No frame completed the full mux..sink path in this window.
            # avg_seconds() returns 0.0 for an empty set, so printing anyway
            # emitted a plausible-looking "detect=0.00ms detect_fps=0.00" line
            # that reads as a measurement rather than an absence of data.
            self.last_timing_time = now
            return

        def avg_seconds(name: str) -> float:
            values = [row[name] for row in rows if name in row]
            return sum(values) / len(values) if values else 0.0

        detect_seconds = avg_seconds("detect")
        fields = [
            "TIME",
            f"n={len(rows)}",
            f"detect={detect_seconds * 1000.0:.2f}ms",
            f"detect_fps={compute_fps(detect_seconds):.2f}",
        ]
        if any("assess" in row for row in rows):
            assess_seconds = avg_seconds("assess")
            fields.extend(
                [
                    f"assess={assess_seconds * 1000.0:.2f}ms",
                    f"assess_fps={compute_fps(assess_seconds):.2f}",
                ]
            )
        fields.extend(
            [
                f"convert={avg_seconds('convert') * 1000.0:.2f}ms",
                f"osd={avg_seconds('osd') * 1000.0:.2f}ms",
                f"sink={avg_seconds('sink') * 1000.0:.2f}ms",
                f"total={avg_seconds('total') * 1000.0:.2f}ms",
            ]
        )

        print(" ".join(fields), flush=True)

        self.last_timing_time = now
