#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import GLib, Gst, GstPbutils

import pyds


PROJECT_DIR = Path("/home/user/deepstream-work")
DEFAULT_STREAM = PROJECT_DIR / "streams/dtc-d4.ts"
SETUP_SCRIPT = PROJECT_DIR / "scripts/setup_and_export_yolo.sh"

PERSON_CLASS_ID = 0
CACHE_POLICY = "parser_line_osd_timing_v1"


def discover_size(path: Path) -> tuple[int, int]:
    info = GstPbutils.Discoverer.new(5 * Gst.SECOND).discover_uri(path.resolve().as_uri())
    stream = info.get_video_streams()[0]
    return int(stream.get_width()), int(stream.get_height())


def model_stem(model: str) -> str:
    return Path(model).stem


def onnx_size(path: Path) -> tuple[int, int]:
    code = (
        "import onnx;"
        f"m=onnx.load({str(path)!r});"
        "d=[x.dim_value or x.dim_param for x in m.graph.input[0].type.tensor_type.shape.dim];"
        "print(int(d[3]), int(d[2]))"
    )
    out = subprocess.check_output(
        [str(PROJECT_DIR / ".venv-yolo/bin/python"), "-c", code],
        text=True,
    )
    return tuple(map(int, out.split()))


def paths(stem: str, long_side: int, w: int, h: int) -> dict[str, Path]:
    tag = f"{stem}_{long_side}_{w}x{h}"
    return {
        "onnx": PROJECT_DIR / "models" / f"{tag}.onnx",
        "meta": PROJECT_DIR / "models" / f"{tag}.meta.json",
        "engine": PROJECT_DIR / "models" / f"{tag}.onnx_b1_gpu0_fp16.engine",
        "config": PROJECT_DIR / "configs" / f"config_infer_primary_{tag}.txt",
    }


def write_config(p: dict[str, Path]) -> None:
    p["config"].write_text(
        f"""[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file={p["onnx"]}
model-engine-file={p["engine"]}
labelfile-path=/home/user/deepstream-work/models/coco_labels.txt
batch-size=1
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=/home/user/deepstream-work/lib/libnvdsinfer_custom_impl_Yolo.so
output-blob-names=output
maintain-aspect-ratio=1
symmetric-padding=1
cluster-mode=2

[class-attrs-all]
pre-cluster-threshold=0.2
nms-iou-threshold=0.45
topk=300
"""
    )


def ensure_model(model: str, stream: Path, long_side: int, src_w: int, src_h: int) -> tuple[int, int, Path]:
    stem = model_stem(model)
    cached = sorted((PROJECT_DIR / "models").glob(f"{stem}_{long_side}_*.onnx"))

    if cached:
        w, h = onnx_size(cached[0])
        p = paths(stem, long_side, w, h)
        write_config(p)
        return w, h, p["config"]

    subprocess.run(
        [str(SETUP_SCRIPT), model, str(long_side), str(stream.relative_to(PROJECT_DIR))],
        cwd=PROJECT_DIR,
        check=True,
    )

    base = PROJECT_DIR / "models" / f"{stem}.onnx"
    w, h = onnx_size(base)
    p = paths(stem, long_side, w, h)

    shutil.copy2(base, p["onnx"])
    p["meta"].write_text(
        json.dumps(
            {
                "model": stem,
                "source_width": src_w,
                "source_height": src_h,
                "model_width": w,
                "model_height": h,
                "cache_policy": CACHE_POLICY,
            },
            indent=2,
        )
        + "\n"
    )
    write_config(p)
    return w, h, p["config"]


def conf_color(conf: float):
    # Piecewise linear interpolation:
    #   0.2 -> red
    #   0.5 -> yellow
    #   0.8+ -> green
    if conf <= 0.2:
        return 1.0, 0.0, 0.0, 1.0

    if conf < 0.5:
        u = (conf - 0.2) / 0.3
        return 1.0, u, 0.0, 1.0

    if conf < 0.8:
        u = (conf - 0.5) / 0.3
        return 1.0 - u, 1.0, 0.0, 1.0

    return 0.0, 1.0, 0.0, 1.0


def add_line_box(batch_meta, frame_meta, left, top, width, height, label, color) -> None:
    x1, y1 = round(left), round(top)
    x2, y2 = round(left + width), round(top + height)

    frame_h = int(getattr(frame_meta, "source_frame_height", 0) or 1080)
    font_size = max(1, round(frame_h * 0.005))

    meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    meta.num_lines = 4
    meta.num_labels = 1

    line_width = 3

    lines = (
        (x1, y1, x2, y1),
        (x2, y1, x2, y2),
        (x2, y2, x1, y2),
        (x1, y2, x1, y1),
    )

    for line, coords in zip(meta.line_params, lines):
        line.x1, line.y1, line.x2, line.y2 = coords
        line.line_width = line_width
        line.line_color.set(*color)

    text = meta.text_params[0]
    text.display_text = label
    # Align text flush with the left edge of the box and directly above
    # the top border, accounting for the line width.
    line_width = 3
    text.x_offset = max(0, x1 - line_width)

    # Put the text above the box so the bottom of the label is flush with
    # the top edge of the top line.
    text.y_offset = max(0, y1 - round(3.5 * font_size) - line_width)
    text.font_params.font_name = "Serif"
    text.font_params.font_size = font_size
    text.font_params.font_color.set(*color)
    text.set_bg_clr = 1
    text.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

    pyds.nvds_add_display_meta_to_frame(frame_meta, meta)


def bbox_probe(_pad, info, _data):
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
    frame_list = batch_meta.frame_meta_list

    while frame_list:
        frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
        obj_list = frame_meta.obj_meta_list

        while obj_list:
            obj = pyds.NvDsObjectMeta.cast(obj_list.data)
            rect = obj.rect_params
            rect.border_width = 0

            if obj.class_id == PERSON_CLASS_ID:
                add_line_box(
                    batch_meta,
                    frame_meta,
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                    f"person {obj.confidence:.2f}",
                    conf_color(float(obj.confidence)),
                )

            obj.text_params.display_text = ""

            obj_list = obj_list.next

        frame_list = frame_list.next

    return Gst.PadProbeReturn.OK


class TimeLog:
    def __init__(self, fps_interval: float = 1.0, timing_interval: float = 10.0):
        self.fps_interval = fps_interval
        self.timing_interval = timing_interval
        self.last_fps_time = time.perf_counter()
        self.last_timing_time = self.last_fps_time
        self.frames = 0
        self.last_frames = 0
        self.times = {}

    def fps_probe(self, _pad, info, _data):
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
        max_fps = self.max_fps
        if max_fps <= 0:
            return Gst.PadProbeReturn.OK

        now = time.perf_counter()
        period = 1.0 / max_fps

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
        self.old_term = termios.tcgetattr(sys.stdin)

    def start(self):
        tty.setcbreak(sys.stdin.fileno())
        GLib.io_add_watch(sys.stdin, GLib.IO_IN, self.on_key)
        print(
            "keys: right/up speed up | left/down slow down | r reset 1x | space pause/play | q quit",
            flush=True,
        )

    def stop(self):
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
        elif key == "\x1b[C":      # right
            self.limiter.set_rate(self.limiter.rate + 0.5)
        elif key == "\x1b[D":      # left
            self.limiter.set_rate(self.limiter.rate - 0.5)
        elif key == "\x1b[A":      # up
            self.limiter.set_rate(self.limiter.rate + 0.05)
        elif key == "\x1b[B":      # down
            self.limiter.set_rate(self.limiter.rate - 0.05)

        return True

    def toggle_pause(self):
        self.paused = not self.paused
        self.pipeline.set_state(Gst.State.PAUSED if self.paused else Gst.State.PLAYING)
        print("paused" if self.paused else "playing", flush=True)



def element(factory: str, name: str):
    e = Gst.ElementFactory.make(factory, name)
    if e is None:
        raise RuntimeError(f"Missing GStreamer element: {factory}")
    return e


def on_pad_added(_demux, pad, parser):
    if "video/x-h265" in (pad.get_current_caps() or pad.query_caps(None)).to_string():
        pad.link(parser.get_static_pad("sink"))


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
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--base-fps", type=float, default=30.0)
    args = ap.parse_args()

    Gst.init(None)

    stream = Path(args.stream)
    stream = stream if stream.is_absolute() else PROJECT_DIR / stream

    src_w, src_h = discover_size(stream)
    model_w, model_h, config = ensure_model(args.model, stream, args.long_side, src_w, src_h)

    print(f"video={src_w}x{src_h} model={model_w}x{model_h} config={config}")

    pipeline = Gst.Pipeline.new("yolo-parser")

    source = element("filesrc", "source")
    demux = element("tsdemux", "demux")
    parser = element("h265parse", "parser")
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

    for e in (source, demux, parser, decoder, queue, streammux, pgie, convert, caps, osd, sink):
        pipeline.add(e)

    source.link(demux)
    demux.connect("pad-added", on_pad_added, parser)
    parser.link(decoder)
    decoder.link(queue)
    queue.get_static_pad("src").link(streammux.request_pad_simple("sink_0"))
    streammux.link(pgie)
    pgie.link(convert)
    convert.link(caps)
    caps.link(osd)
    osd.link(sink)

    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, bbox_probe, None)

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
    limiter = RateLimiter(base_fps=args.base_fps)
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, limiter.probe, None)
    controls = KeyboardControls(pipeline, loop, limiter)
    controls.start()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        controls.stop()
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
