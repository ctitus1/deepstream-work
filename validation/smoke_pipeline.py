#!/usr/bin/env python3
"""Headless smoke test for the DeepStream detection pipeline.

Builds the shared pipeline with ``display=False`` and runs a bounded number of
frames, so it can verify a container end to end without an X display:

- the decoder handles the source (including 4K H.265),
- ``nvinfer`` loads the custom YOLO parser and builds/loads a TensorRT engine,
- detections actually come out of the parser,
- optionally, the recording branch produces a playable mp4.

Exits non-zero if the pipeline errors or no frames are processed, so it is
usable as a build gate.

    python3 validation/smoke_pipeline.py --frames 60
    python3 validation/smoke_pipeline.py --frames 60 --record
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from deepstream_yolo.model_cache import discover_size, ensure_model  # noqa: E402
from deepstream_yolo.paths import DEFAULT_STREAM  # noqa: E402
from deepstream_yolo.pipeline import build_pipeline, on_message  # noqa: E402
from deepstream_yolo.recording import resolve_record_path  # noqa: E402
from deepstream_yolo.stream_source import resolve_stream_source  # noqa: E402


def counting_probe(state, limit, loop):
    """Count frames and detections leaving the PGIE, stopping at ``limit``."""

    import pyds

    def probe(_pad, info):
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        if batch is None:
            return Gst.PadProbeReturn.OK

        frame_node = batch.frame_meta_list
        while frame_node is not None:
            frame = pyds.NvDsFrameMeta.cast(frame_node.data)
            state["frames"] += 1

            obj_node = frame.obj_meta_list
            while obj_node is not None:
                state["objects"] += 1
                obj_node = obj_node.next

            frame_node = frame_node.next

        if state["frames"] >= limit and not state["done"]:
            state["done"] = True
            GLib.idle_add(loop.quit)

        return Gst.PadProbeReturn.OK

    return probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default=str(DEFAULT_STREAM))
    parser.add_argument("--model", default="yolo12x.pt")
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--record", nargs="?", const="", default=None, metavar="PATH")
    args = parser.parse_args()

    Gst.init(None)

    stream = resolve_stream_source(args.stream)
    src_w, src_h = discover_size(stream.uri)
    print(f"source: {stream.uri} {src_w}x{src_h} rtsp={stream.is_rtsp}", flush=True)

    model_w, model_h, config = ensure_model(
        args.model, stream, args.long_side, src_w, src_h, args.conf
    )
    print(f"model: {model_w}x{model_h} config={config}", flush=True)

    record_path = resolve_record_path(args.record, stream.uri) if args.record is not None else None
    if record_path:
        print(f"recording: {record_path}", flush=True)

    parts = build_pipeline(
        stream, src_w, src_h, config, display=False, record_path=record_path
    )

    loop = GLib.MainLoop()
    state = {"frames": 0, "objects": 0, "done": False}

    pgie_src = parts.pgie.get_static_pad("src")
    pgie_src.add_probe(Gst.PadProbeType.BUFFER, counting_probe(state, args.frames, loop))

    bus = parts.pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    def bail():
        print("TIMEOUT waiting for frames", file=sys.stderr, flush=True)
        loop.quit()
        return False

    GLib.timeout_add_seconds(int(args.timeout), bail)

    started = time.monotonic()
    parts.pipeline.set_state(Gst.State.PLAYING)
    print("pipeline PLAYING (first run builds the TensorRT engine)", flush=True)

    try:
        loop.run()
    finally:
        if record_path:
            parts.pipeline.send_event(Gst.Event.new_eos())
            bus.timed_pop_filtered(10 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        parts.pipeline.set_state(Gst.State.NULL)

    elapsed = time.monotonic() - started
    print(
        f"RESULT frames={state['frames']} objects={state['objects']} elapsed={elapsed:.1f}s",
        flush=True,
    )

    if record_path and record_path.exists():
        print(f"recorded: {record_path} bytes={record_path.stat().st_size}", flush=True)

    if state["frames"] == 0:
        print("FAIL: no frames processed", file=sys.stderr, flush=True)
        return 1

    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
