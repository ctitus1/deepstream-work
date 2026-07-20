#!/usr/bin/env python3
"""Dump per-frame YOLO detection boxes for a whole video, once.

This is the ground truth every motion approach is scored against. It runs the
detector alone -- no assessment, no display, no motion branch -- over every
frame of a file, and writes the boxes to JSON keyed by frame index.

Doing it once and reusing the result is what makes the comparison affordable:
an approach under test then only has to run its own motion analysis, instead of
paying for detection and assessment on every experiment.

Frame indices, not timestamps, are the key. The queues here are deliberately
NOT leaky, so a file source yields every frame in order and index i means the
same frame in this dump as it does in a motion run configured the same way.

Usage:
    python3 eval/dump_detections.py [--stream PATH] [--out eval/detections.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

from deepstream_yolo.media import resolve_stream_source  # noqa: E402
from deepstream_yolo.model_cache import discover_size, ensure_model  # noqa: E402
from deepstream_yolo.paths import DEFAULT_MEDIA, DEFAULT_MODEL  # noqa: E402
from deepstream_yolo.pipeline import element, on_message, set_property_if_present  # noqa: E402

PERSON_CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default=str(DEFAULT_MEDIA))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default=str(PROJECT_DIR / "eval" / "detections.json"))
    return parser.parse_args()


def build(stream, src_w, src_h, config):
    """Detector-only graph over a local file, dropping nothing."""
    pipeline = Gst.Pipeline.new("dump-detections")

    source = element("filesrc", "source")
    source.set_property("location", str(stream.path))
    demux = element("qtdemux", "demux")
    h265 = element("h265parse", "h265-parser")
    h264 = element("h264parse", "h264-parser")
    decoder = element("nvv4l2decoder", "decoder")
    queue = element("queue", "queue")
    streammux = element("nvstreammux", "streammux")
    pgie = element("nvinfer", "pgie")
    sink = element("fakesink", "sink")

    # Every frame, in order: this is ground truth, not a live path.
    queue.set_property("max-size-buffers", 0)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    queue.set_property("leaky", 0)

    streammux.set_property("batch-size", 1)
    streammux.set_property("width", src_w)
    streammux.set_property("height", src_h)
    streammux.set_property("batched-push-timeout", 40000)
    set_property_if_present(streammux, "attach-sys-ts", False)
    set_property_if_present(streammux, "live-source", False)
    set_property_if_present(streammux, "sync-inputs", False)
    pgie.set_property("config-file-path", str(config))
    sink.set_property("sync", False)
    set_property_if_present(sink, "async", False)

    for elem in (source, demux, h265, h264, decoder, queue, streammux, pgie, sink):
        pipeline.add(elem)

    source.link(demux)

    def on_pad_added(_demux, pad):
        caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()
        parser_elem = h265 if "video/x-h265" in caps else h264 if "video/x-h264" in caps else None
        if parser_elem is None:
            return
        if not parser_elem.get_static_pad("src").is_linked():
            parser_elem.link(decoder)
        pad.link(parser_elem.get_static_pad("sink"))

    demux.connect("pad-added", on_pad_added)
    decoder.link(queue)
    queue.get_static_pad("src").link(streammux.request_pad_simple("sink_0"))
    streammux.link(pgie)
    pgie.link(sink)
    return pipeline, pgie


def main() -> int:
    args = parse_args()
    stream = resolve_stream_source(args.stream)
    if stream.path is None:
        print("dump_detections needs a local file, not a live stream", file=sys.stderr)
        return 2

    Gst.init(None)
    src_w, src_h = discover_size(stream.uri)
    _, _, config = ensure_model(args.model, stream, args.long_side, src_w, src_h, args.conf)

    pipeline, pgie = build(stream, src_w, src_h, config)
    frames: list[dict] = []

    def probe(_pad, info, _data):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            boxes = []
            obj_list = frame_meta.obj_meta_list
            while obj_list:
                obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                if obj.class_id == PERSON_CLASS_ID:
                    rect = obj.rect_params
                    boxes.append(
                        {
                            "left": float(rect.left),
                            "top": float(rect.top),
                            "width": float(rect.width),
                            "height": float(rect.height),
                            "conf": float(obj.confidence),
                        }
                    )
                obj_list = obj_list.next

            frames.append({"index": len(frames), "pts": int(buf.pts), "boxes": boxes})
            if len(frames) % 500 == 0:
                print(f"  {len(frames)} frames", flush=True)
            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, None)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source": str(stream.path.relative_to(PROJECT_DIR)),
                "width": src_w,
                "height": src_h,
                "model": args.model,
                "conf": args.conf,
                "frames": frames,
            }
        )
    )
    counts = [len(f["boxes"]) for f in frames]
    print(
        f"wrote {out} frames={len(frames)} "
        f"boxes/frame min={min(counts, default=0)} max={max(counts, default=0)} "
        f"mean={sum(counts)/max(len(counts),1):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
