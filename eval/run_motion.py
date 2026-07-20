#!/usr/bin/env python3
"""Run one motion approach over a whole video and dump its boxes.

Motion only: no detector, no assessment, no display. Detection boxes are
already on disk from ``dump_detections.py``, so an experiment costs one decode
plus whatever the approach itself does, and different approaches are compared
on identical frames.

The pipeline offers every approach the same two inputs, and an approach takes
what it needs:

  * ``ctx.flow``  -- the ``nvof`` field as (rows, cols, 2) float32, in
                     branch-resolution px/frame, or None on the first frame and
                     when ``needs_flow`` is False.
  * ``ctx.rgba``  -- the frame as an (h, w, 4) uint8 array at branch
                     resolution, or None when ``needs_pixels`` is False.

Queues here are NOT leaky and the sink does not sync, so every frame is seen in
order and frame index i is the same frame as index i in the detection dump.
That is what makes the two comparable; it is an offline measurement, not the
live path.

An approach is a module under ``src/deepstream_yolo/approaches/`` exposing:

    NAME = "short-name"
    class Approach:
        needs_flow = True
        needs_pixels = False
        def __init__(self, cfg: dict): ...
        def process(self, ctx) -> list[dict]:
            '''Return boxes as {"left","top","width","height", ...} in SOURCE
            pixel coordinates. Any extra keys are carried through to the
            output for debugging. Return as many boxes as the frame warrants --
            there is no cap, and imposing one is not the way to score well.'''

Usage:
    python3 eval/run_motion.py --approach baseline [--out eval/runs/baseline.json]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import numpy as np  # noqa: E402
import pyds  # noqa: E402

from deepstream_yolo.media import resolve_stream_source  # noqa: E402
from deepstream_yolo.model_cache import discover_size  # noqa: E402
from deepstream_yolo.paths import DEFAULT_MEDIA  # noqa: E402
from deepstream_yolo.pipeline import element, on_message, set_property_if_present  # noqa: E402

FLOW_FIXED_POINT_SCALE = 32.0


class Context:
    """Everything an approach is given for one frame."""

    __slots__ = (
        "frame_index",
        "pts",
        "flow",
        "rgba",
        "src_w",
        "src_h",
        "width",
        "height",
        "grid_size",
        "scale_x",
        "scale_y",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", required=True, help="module under deepstream_yolo.approaches")
    parser.add_argument("--stream", default=str(DEFAULT_MEDIA))
    parser.add_argument("--width", type=int, default=960, help="branch resolution width")
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--grid-size", type=int, default=4, help="nvof block size")
    parser.add_argument("--cfg", default="{}", help="JSON dict passed to the approach")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def flow_field(frame_meta) -> np.ndarray | None:
    user_list = frame_meta.frame_user_meta_list
    while user_list:
        user_meta = pyds.NvDsUserMeta.cast(user_list.data)
        if user_meta.base_meta.meta_type == pyds.NVDS_OPTICAL_FLOW_META:
            of_meta = pyds.NvDsOpticalFlowMeta.cast(user_meta.user_meta_data)
            raw = pyds.get_optical_flow_vectors(of_meta)
            rows, cols = int(of_meta.rows), int(of_meta.cols)
            if rows <= 0 or cols <= 0 or raw is None:
                return None
            return np.asarray(raw, dtype=np.float32).reshape(rows, cols, 2) / FLOW_FIXED_POINT_SCALE
        user_list = user_list.next
    return None


def build(stream, width, height, needs_flow):
    pipeline = Gst.Pipeline.new("motion-eval")

    source = element("filesrc", "source")
    source.set_property("location", str(stream.path))
    demux = element("qtdemux", "demux")
    h265 = element("h265parse", "h265-parser")
    h264 = element("h264parse", "h264-parser")
    decoder = element("nvv4l2decoder", "decoder")
    queue = element("queue", "queue")
    streammux = element("nvstreammux", "streammux")
    nvof = element("nvof", "nvof") if needs_flow else None
    convert = element("nvvideoconvert", "convert")
    caps = element("capsfilter", "caps")
    sink = element("fakesink", "sink")

    queue.set_property("max-size-buffers", 0)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    queue.set_property("leaky", 0)

    streammux.set_property("batch-size", 1)
    streammux.set_property("width", width)
    streammux.set_property("height", height)
    streammux.set_property("batched-push-timeout", 40000)
    set_property_if_present(streammux, "attach-sys-ts", False)
    set_property_if_present(streammux, "live-source", False)
    set_property_if_present(streammux, "sync-inputs", False)

    if nvof is not None:
        set_property_if_present(nvof, "preset-level", 0)
        set_property_if_present(nvof, "grid-size", 0)

    set_property_if_present(convert, "nvbuf-memory-type", 3)
    caps.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), format=RGBA, width={width}, height={height}"
        ),
    )
    sink.set_property("sync", False)
    set_property_if_present(sink, "async", False)

    elements = [source, demux, h265, h264, decoder, queue, streammux]
    if nvof is not None:
        elements.append(nvof)
    elements.extend([convert, caps, sink])
    for elem in elements:
        pipeline.add(elem)

    source.link(demux)

    def on_pad_added(_demux, pad):
        text = (pad.get_current_caps() or pad.query_caps(None)).to_string()
        parser_elem = h265 if "video/x-h265" in text else h264 if "video/x-h264" in text else None
        if parser_elem is None:
            return
        if not parser_elem.get_static_pad("src").is_linked():
            parser_elem.link(decoder)
        pad.link(parser_elem.get_static_pad("sink"))

    demux.connect("pad-added", on_pad_added)
    decoder.link(queue)
    queue.get_static_pad("src").link(streammux.request_pad_simple("sink_0"))
    if nvof is not None:
        streammux.link(nvof)
        nvof.link(convert)
    else:
        streammux.link(convert)
    convert.link(caps)
    caps.link(sink)
    return pipeline, caps


def main() -> int:
    args = parse_args()
    stream = resolve_stream_source(args.stream)
    if stream.path is None:
        print("run_motion needs a local file", file=sys.stderr)
        return 2

    module = importlib.import_module(f"deepstream_yolo.approaches.{args.approach}")
    approach = module.Approach(json.loads(args.cfg))
    needs_flow = getattr(module.Approach, "needs_flow", True)
    needs_pixels = getattr(module.Approach, "needs_pixels", False)

    Gst.init(None)
    src_w, src_h = discover_size(stream.uri)
    pipeline, tail = build(stream, args.width, args.height, needs_flow)

    scale_x = args.grid_size * (src_w / float(args.width))
    scale_y = args.grid_size * (src_h / float(args.height))
    frames: list[dict] = []
    timings: list[float] = []
    loop = GLib.MainLoop()

    def probe(_pad, info, _data):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            index = len(frames)

            flow = flow_field(frame_meta) if needs_flow else None
            rgba = None
            if needs_pixels:
                try:
                    rgba = np.asarray(
                        pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
                    )
                except Exception:
                    rgba = None

            ctx = Context(
                frame_index=index,
                pts=int(buf.pts),
                flow=flow,
                rgba=rgba,
                src_w=src_w,
                src_h=src_h,
                width=args.width,
                height=args.height,
                grid_size=args.grid_size,
                scale_x=scale_x,
                scale_y=scale_y,
            )

            started = time.perf_counter()
            try:
                boxes = approach.process(ctx) or []
            except Exception as exc:  # a broken frame must not kill the run
                print(f"frame {index}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                boxes = []
            timings.append((time.perf_counter() - started) * 1000.0)

            if rgba is not None:
                try:
                    pyds.unmap_nvds_buf_surface(hash(buf), frame_meta.batch_id)
                except Exception:
                    pass

            frames.append({"index": index, "pts": int(buf.pts), "boxes": boxes})
            if len(frames) % 500 == 0:
                print(f"  {len(frames)} frames", flush=True)
            if args.max_frames and len(frames) >= args.max_frames:
                loop.quit()
                return Gst.PadProbeReturn.OK

            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    tail.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, None)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    out = Path(args.out) if args.out else PROJECT_DIR / "eval" / "runs" / f"{args.approach}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    timings.sort()
    out.write_text(
        json.dumps(
            {
                "approach": getattr(module, "NAME", args.approach),
                "module": args.approach,
                "cfg": json.loads(args.cfg),
                "source": str(stream.path.relative_to(PROJECT_DIR)),
                "width": src_w,
                "height": src_h,
                "branch_width": args.width,
                "branch_height": args.height,
                "ms_per_frame_median": timings[len(timings) // 2] if timings else 0.0,
                "ms_per_frame_p95": timings[int(len(timings) * 0.95)] if timings else 0.0,
                "frames": frames,
            }
        )
    )
    counts = [len(f["boxes"]) for f in frames]
    print(
        f"wrote {out} frames={len(frames)} "
        f"boxes/frame mean={sum(counts)/max(len(counts),1):.2f} max={max(counts, default=0)} "
        f"ms/frame median={timings[len(timings)//2] if timings else 0:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
