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
  * ``ctx.fps``   -- the source frame rate. Approaches quote their temporal
                     parameters in seconds and convert with this; a parameter
                     counted in frames means different things on 30 and 60 fps
                     footage.

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
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyds  # noqa: E402

# Thread policy is set once, here, by the process that owns it. Approaches must
# not call cv2.setNumThreads() themselves: it is global, so one module doing it
# at import silently throttles every other approach sharing the run.
# 8 measured best across the approaches on a 16-core host (klt 9.9 ms, gradient
# 7.3, bgsub 8.3); past that they contend for memory bandwidth more than they
# gain parallelism, and one of them is slightly slower at 16. Override with
# MOTION_CV_THREADS on a machine with a different shape.
cv_threads = int(os.environ.get("MOTION_CV_THREADS", "8"))
cv2.setNumThreads(cv_threads)

from progress import Progress, frame_count  # noqa: E402

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
        "fps",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approach",
        required=True,
        help="module under deepstream_yolo.approaches; comma-separated runs several "
        "in one pass over the video",
    )
    parser.add_argument("--stream", default=str(DEFAULT_MEDIA))
    parser.add_argument("--width", type=int, default=960, help="branch resolution width")
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="branch resolution height; 0 derives it from the source aspect ratio",
    )
    parser.add_argument("--grid-size", type=int, default=4, help="nvof block size")
    parser.add_argument("--cfg", default="{}", help="JSON dict passed to the approach")
    parser.add_argument(
        "--variants",
        default="",
        help='JSON list of {"name","label","cfg"} -- runs one approach under several '
        "configurations in a single pass over the video",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="run batched approaches one after another; use when timing one of them",
    )
    parser.add_argument(
        "--out",
        default="",
        help="output path; with several approaches, a directory to write <name>.json into",
    )
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

    # Several approaches in one pass. Decoding 4K H.265 is the expensive part of
    # a comparison and it does not depend on which approach is asking, so running
    # them together turns N decodes into one. They are independent -- each keeps
    # its own state and never sees another's output -- so this changes cost, not
    # results.
    base_cfg = json.loads(args.cfg)
    if args.variants:
        # One module, many configurations -- a parameter sweep. Every variant
        # sees the same decoded frames, which is the point: differences in the
        # output are differences in the parameters and nothing else.
        spec = json.loads(args.variants)
        module = importlib.import_module(f"deepstream_yolo.approaches.{args.approach}")
        names = [v["name"] for v in spec]
        labels = [v.get("label", v["name"]) for v in spec]
        modules = [module] * len(spec)
        approaches = [module.Approach({**base_cfg, **v.get("cfg", {})}) for v in spec]
        configs = [{**base_cfg, **v.get("cfg", {})} for v in spec]
    else:
        names = [n for n in args.approach.split(",") if n]
        modules = [importlib.import_module(f"deepstream_yolo.approaches.{n}") for n in names]
        labels = [getattr(m, "NAME", n) for m, n in zip(modules, names)]
        approaches = [m.Approach(dict(base_cfg)) for m in modules]
        configs = [base_cfg] * len(names)

    # The pipeline has to satisfy every approach in the batch, so the inputs are
    # the union: one that wants flow means nvof runs for all of them.
    needs_flow = any(getattr(m.Approach, "needs_flow", True) for m in modules)
    needs_pixels = any(getattr(m.Approach, "needs_pixels", False) for m in modules)

    Gst.init(None)
    src_w, src_h = discover_size(stream.uri)

    # Derive the branch height from the source unless one was asked for.
    #
    # nvstreammux scales to exactly what it is told, so a branch pinned to a
    # fixed height stretches any source of a different aspect ratio.
    # streams/lorton-d4-thermal.mp4 is 640x512: forced into 960x540 it comes out
    # 1.5x wider and only 1.055x taller. Everything downstream compares
    # Euclidean distances -- residual magnitudes, cluster radii, match radii --
    # and those assume both axes share a scale, so the stretch silently makes
    # horizontal motion worth 1.42x vertical. Nothing errors; the numbers just
    # quietly mean something different in x than in y.
    #
    # Rounded to an even height because encoders and several DeepStream
    # elements dislike odd dimensions.
    # Frame rate, for the approaches to convert their time-valued parameters
    # with. Falls back to 30 -- the rate everything was tuned at -- when the
    # container will not say, which leaves behaviour exactly as it was.
    fps = 30.0
    try:
        import cv2 as _cv2

        probe = _cv2.VideoCapture(str(stream.path))
        reported = float(probe.get(_cv2.CAP_PROP_FPS) or 0.0)
        probe.release()
        if 1.0 < reported < 1000.0:
            fps = reported
    except Exception:
        pass

    height = args.height or max(2, int(round(args.width * src_h / src_w / 2)) * 2)
    if not args.height:
        print(
            f"  branch {args.width}x{height} from {src_w}x{src_h}"
            f" (aspect preserved), {fps:g} fps",
            file=sys.stderr,
            flush=True,
        )
    args.height = height

    pipeline, tail = build(stream, args.width, args.height, needs_flow)

    scale_x = args.grid_size * (src_w / float(args.width))
    scale_y = args.grid_size * (src_h / float(args.height))
    frames: list[list[dict]] = [[] for _ in approaches]
    timings: list[list[float]] = [[] for _ in approaches]

    def one(slot: int, ctx, index: int):
        started = time.perf_counter()
        try:
            boxes = approaches[slot].process(ctx) or []
        except Exception as exc:  # a broken frame must not kill the run
            print(
                f"{names[slot]} frame {index}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            boxes = []
        timings[slot].append((time.perf_counter() - started) * 1000.0)
        return boxes

    # Approaches are independent, so they run concurrently rather than in
    # sequence -- wall time becomes the slowest of them instead of their sum.
    #
    # The threads are worth having because the work is in OpenCV and numpy,
    # which release the GIL. The thread *budget* is the subtle part: each
    # approach is given its share of the cores rather than all of them, because
    # four approaches each spawning a full-width pool oversubscribes and they
    # spend the time fighting. Measured on 16 cores, sequential at 8 threads
    # each totals 34.7 ms/frame; concurrent at 4 threads each is bounded by the
    # slowest at 13.9.
    #
    # Each approach is touched by exactly one task per frame and all tasks join
    # before the next frame, so the per-approach state needs no locking.
    parallel = len(approaches) > 1 and not args.sequential
    pool = ThreadPoolExecutor(max_workers=len(approaches)) if parallel else None
    if parallel:
        cv2.setNumThreads(max(1, cv_threads // len(approaches)))

    def run_approaches(ctx, index):
        if pool is None:
            return [one(slot, ctx, index) for slot in range(len(approaches))]
        futures = [pool.submit(one, slot, ctx, index) for slot in range(len(approaches))]
        return [f.result() for f in futures]

    loop = GLib.MainLoop()

    total = args.max_frames or frame_count(stream.path)
    label = f"{len(approaches)} variant(s)" if len(approaches) > 1 else names[0]
    bar = Progress(total, label)

    def probe(_pad, info, _data):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            index = len(frames[0])

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
                fps=fps,
            )

            frame_boxes = run_approaches(ctx, index)

            if rgba is not None:
                try:
                    pyds.unmap_nvds_buf_surface(hash(buf), frame_meta.batch_id)
                except Exception:
                    pass

            for slot, boxes in enumerate(frame_boxes):
                frames[slot].append(
                    {"index": index, "pts": int(buf.pts), "boxes": boxes}
                )
            bar.update()
            if args.max_frames and len(frames[0]) >= args.max_frames:
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

    bar.close()
    if pool is not None:
        pool.shutdown()

    default_dir = PROJECT_DIR / "eval" / "runs"
    for slot, name in enumerate(names):
        if not args.out:
            out = default_dir / f"{name}.json"
        elif len(names) > 1 or args.out.endswith("/"):
            out = Path(args.out) / f"{name}.json"
        else:
            out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)

        ms = sorted(timings[slot])
        out.write_text(
            json.dumps(
                {
                    "approach": labels[slot],
                    "module": args.approach if args.variants else name,
                    "cfg": configs[slot],
                    "source": str(stream.path.relative_to(PROJECT_DIR)),
                    "width": src_w,
                    "height": src_h,
                    "branch_width": args.width,
                    "branch_height": args.height,
                    "ms_per_frame_median": ms[len(ms) // 2] if ms else 0.0,
                    "ms_per_frame_p95": ms[int(len(ms) * 0.95)] if ms else 0.0,
                    "frames": frames[slot],
                }
            )
        )
        counts = [len(f["boxes"]) for f in frames[slot]]
        print(
            f"wrote {out} frames={len(frames[slot])} "
            f"boxes/frame mean={sum(counts)/max(len(counts),1):.2f} "
            f"max={max(counts, default=0)} "
            f"ms/frame median={ms[len(ms)//2] if ms else 0:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
