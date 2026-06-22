#!/usr/bin/env python3
import argparse
import os
import sys
import termios
import threading
import time
import tty


class StderrLineFilter:
    def __init__(self, suppress):
        self.suppress = suppress
        self.read_fd = None
        self.saved_stderr_fd = None
        self.thread = None
        self.skip_blank = False

    def start(self) -> None:
        read_fd, write_fd = os.pipe()
        self.read_fd = read_fd
        self.saved_stderr_fd = os.dup(2)
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        os.dup2(write_fd, 2)
        os.close(write_fd)

    def stop(self) -> None:
        if self.saved_stderr_fd is None:
            return

        sys.stderr.flush()
        os.dup2(self.saved_stderr_fd, 2)
        if self.thread:
            self.thread.join(timeout=1.0)
        os.close(self.saved_stderr_fd)
        self.saved_stderr_fd = None

    def _emit(self, line: bytes) -> None:
        if self.suppress(line):
            self.skip_blank = True
            return

        if self.skip_blank and not line.strip():
            return

        self.skip_blank = False
        os.write(self.saved_stderr_fd, line)

    def _pump(self) -> None:
        pending = b""
        try:
            while True:
                chunk = os.read(self.read_fd, 4096)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._emit(line + b"\n")
            if pending:
                self._emit(pending)
        finally:
            os.close(self.read_fd)


def is_gst_plugin_scan_warning(line: bytes) -> bool:
    return b"gst-plugin-scanner" in line and b"Failed to load plugin" in line


GST_SCAN_WARNING_FILTER = None
if "--show-gst-scan-warnings" not in sys.argv:
    GST_SCAN_WARNING_FILTER = StderrLineFilter(is_gst_plugin_scan_warning)
    GST_SCAN_WARNING_FILTER.start()

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import GLib, Gst

import pyds

from deepstream_yolo.model_cache import discover_size, ensure_model
from deepstream_yolo.paths import DEFAULT_STREAM, resolve_project_path

PERSON_CLASS_ID = 0


def stop_gst_scan_warning_filter() -> None:
    if GST_SCAN_WARNING_FILTER:
        GST_SCAN_WARNING_FILTER.stop()


def conf_color(confidence: float, lower_conf: float):
    mid_conf = (lower_conf + 0.8) / 2.0

    if confidence <= lower_conf:
        return 1.0, 0.0, 0.0, 1.0

    if confidence < mid_conf:
        t = (confidence - lower_conf) / max(mid_conf - lower_conf, 1e-6)
        return 1.0, t, 0.0, 1.0

    if confidence < 0.8:
        t = (confidence - mid_conf) / max(0.8 - mid_conf, 1e-6)
        return 1.0 - t, 1.0, 0.0, 1.0

    return 0.0, 1.0, 0.0, 1.0


def add_line_box(batch_meta, frame_meta, left, top, width, height, label, color) -> None:
    x1, y1 = round(left), round(top)
    x2, y2 = round(left + width), round(top + height)

    frame_h = int(getattr(frame_meta, "source_frame_height", 0) or 1080)
    font_size = max(1, round(frame_h * 0.001))
    line_width = 3

    meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    meta.num_lines = 4
    meta.num_labels = 1

    for line, coords in zip(
        meta.line_params,
        (
            (x1, y1, x2, y1),
            (x2, y1, x2, y2),
            (x2, y2, x1, y2),
            (x1, y2, x1, y1),
        ),
    ):
        line.x1, line.y1, line.x2, line.y2 = coords
        line.line_width = line_width
        line.line_color.set(*color)

    text = meta.text_params[0]
    text.display_text = label
    text.x_offset = max(0, x1 - line_width // 2)
    text.y_offset = max(0, y1 - 2 * font_size - 100)
    text.font_params.font_name = "Serif"
    text.font_params.font_size = font_size
    text.font_params.font_color.set(*color)
    text.set_bg_clr = 1
    text.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

    pyds.nvds_add_display_meta_to_frame(frame_meta, meta)


def bbox_probe(conf: float):
    def _probe(_pad, info, _data):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list

        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            obj_list = frame_meta.obj_meta_list

            while obj_list:
                obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                rect = obj.rect_params
                rect.border_width = 0
                obj.text_params.display_text = ""

                if obj.class_id == PERSON_CLASS_ID:
                    add_line_box(
                        batch_meta,
                        frame_meta,
                        rect.left,
                        rect.top,
                        rect.width,
                        rect.height,
                        f"person {obj.confidence:.2f}",
                        conf_color(float(obj.confidence), conf),
                    )

                obj_list = obj_list.next

            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe


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


class RateLimiter:
    def __init__(self, base_fps: float = 30.0, rate: float = 1.0):
        self.base_fps = base_fps
        self.rate = rate
        self.last_time = 0.0

    @property
    def max_fps(self) -> float:
        return self.base_fps * self.rate

    def set_rate(self, rate: float) -> None:
        self.rate = round(max(0.05, min(8.0, rate)), 2)
        print(f"rate={self.rate:.2f}x max_fps={self.max_fps:.1f}", flush=True)

    def probe(self, _pad, _info, _data):
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
        print(
            "keys: right/up speed up | left/down slow down | r reset 1x | space pause/play | q quit",
            flush=True,
        )

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
        elif key == "r":
            self.limiter.set_rate(1.0)
        elif key == "\x1b[C":
            self.limiter.set_rate(self.limiter.rate + 0.5)
        elif key == "\x1b[D":
            self.limiter.set_rate(self.limiter.rate - 0.5)
        elif key == "\x1b[A":
            self.limiter.set_rate(self.limiter.rate + 0.05)
        elif key == "\x1b[B":
            self.limiter.set_rate(self.limiter.rate - 0.05)

        return True

    def toggle_pause(self):
        self.paused = not self.paused
        self.pipeline.set_state(Gst.State.PAUSED if self.paused else Gst.State.PLAYING)
        print("paused" if self.paused else "playing", flush=True)


def element(factory: str, name: str):
    elem = Gst.ElementFactory.make(factory, name)
    if elem is None:
        raise RuntimeError(f"Missing GStreamer element: {factory}")
    return elem


def on_pad_added(_demux, pad, parsers):
    caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()

    if "video/x-h265" in caps:
        pad.link(parsers["h265"].get_static_pad("sink"))
    elif "video/x-h264" in caps:
        pad.link(parsers["h264"].get_static_pad("sink"))


def on_message(_bus, msg, loop):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"ERROR: {err}\nDEBUG: {dbg}", file=sys.stderr)
        loop.quit()
    elif msg.type == Gst.MessageType.EOS:
        loop.quit()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo12x.pt")
    ap.add_argument("--long-side", type=int, default=640)
    ap.add_argument("--stream", default=str(DEFAULT_STREAM))
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--base-fps", type=float, default=30.0)
    ap.add_argument("--show-gst-scan-warnings", action="store_true")
    args = ap.parse_args()

    stream = resolve_project_path(args.stream)

    try:
        Gst.init(None)
        src_w, src_h = discover_size(stream)
    finally:
        stop_gst_scan_warning_filter()

    model_w, model_h, config = ensure_model(
        args.model,
        stream,
        args.long_side,
        src_w,
        src_h,
        args.conf,
    )

    print(f"video={src_w}x{src_h} model={model_w}x{model_h} conf={args.conf} config={config}")

    pipeline = Gst.Pipeline.new("yolo-parser")

    source = element("filesrc", "source")
    demux = element("qtdemux", "demux")
    h265_parser = element("h265parse", "h265-parser")
    h264_parser = element("h264parse", "h264-parser")
    decoder = element("nvv4l2decoder", "decoder")
    queue = element("queue", "queue")
    streammux = element("nvstreammux", "streammux")
    pgie = element("nvinfer", "pgie")
    convert = element("nvvideoconvert", "convert")
    caps = element("capsfilter", "caps")
    osd = element("nvdsosd", "osd")
    sink = element("nveglglessink", "sink")

    source.set_property("location", str(stream))
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", src_w)
    streammux.set_property("height", src_h)
    streammux.set_property("batched-push-timeout", 40000)
    pgie.set_property("config-file-path", str(config))
    caps.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), format=RGBA, width={src_w}, height={src_h}"
        ),
    )
    osd.set_property("process-mode", 1)
    osd.set_property("display-bbox", 1)
    osd.set_property("display-text", 1)
    sink.set_property("sync", False)
    sink.set_property("qos", False)

    for elem in (
        source,
        demux,
        h265_parser,
        h264_parser,
        decoder,
        queue,
        streammux,
        pgie,
        convert,
        caps,
        osd,
        sink,
    ):
        pipeline.add(elem)

    source.link(demux)
    demux.connect("pad-added", on_pad_added, {"h265": h265_parser, "h264": h264_parser})
    h265_parser.link(decoder)
    h264_parser.link(decoder)
    decoder.link(queue)
    queue.get_static_pad("src").link(streammux.request_pad_simple("sink_0"))
    streammux.link(pgie)
    pgie.link(convert)
    convert.link(caps)
    caps.link(osd)
    osd.link(sink)

    pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        bbox_probe(args.conf),
        None,
    )

    limiter = RateLimiter(base_fps=args.base_fps)
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, limiter.probe, None)

    if args.debug:
        timer = TimeLog()
        sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, timer.fps_probe, None)
        for pad, stage in (
            (streammux.get_static_pad("src"), "mux"),
            (pgie.get_static_pad("src"), "infer"),
            (caps.get_static_pad("src"), "convert"),
            (osd.get_static_pad("src"), "osd"),
            (sink.get_static_pad("sink"), "sink"),
        ):
            pad.add_probe(Gst.PadProbeType.BUFFER, timer.mark(stage), None)

    loop = GLib.MainLoop()
    controls = KeyboardControls(pipeline, loop, limiter) if sys.stdin.isatty() else None
    if controls:
        controls.start()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        if controls:
            controls.stop()
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
