import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds


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
            print(f"FPS {fps:.2f}", flush=True)
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

            if stage == "sink":
                self._print_timing(now)

            return Gst.PadProbeReturn.OK

        return _probe

    def _print_timing(self, now: float) -> None:
        if now - self.last_timing_time < self.timing_interval:
            return

        rows = []
        for frame_num, t in list(self.times.items()):
            if all(k in t for k in ("mux", "infer", "convert", "osd", "sink")):
                rows.append(
                    (
                        t["infer"] - t["mux"],
                        t["convert"] - t["infer"],
                        t["osd"] - t["convert"],
                        t["sink"] - t["osd"],
                        t["sink"] - t["mux"],
                    )
                )
                del self.times[frame_num]

        def avg_ms(i: int) -> float:
            return 1000.0 * sum(row[i] for row in rows) / len(rows) if rows else 0.0

        print(
            "TIME "
            f"n={len(rows)} "
            f"infer={avg_ms(0):.2f}ms "
            f"convert={avg_ms(1):.2f}ms "
            f"osd={avg_ms(2):.2f}ms "
            f"sink={avg_ms(3):.2f}ms "
            f"total={avg_ms(4):.2f}ms",
            flush=True,
        )

        self.last_timing_time = now
