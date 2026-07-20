#!/usr/bin/env python3
"""Bisect where RTSP reference timestamps stop surviving the DeepStream graph.

Probes the same buffer at every stage of ``rtspsrc -> depay -> parse ->
nvv4l2decoder -> nvstreammux -> nvinfer`` and reports, per stage, whether a
wall-clock timestamp is still reachable. Pre-mux stages read
``GstReferenceTimestampMeta`` directly; post-mux stages use the shared
``frame_timestamp()`` resolver, so ``nvstreammux``'s own ``ntp_timestamp`` is
counted too. The closing summary names the first stage that loses the stamp.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gi

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

from deepstream_yolo.assessment_runtime import frame_timestamp  # noqa: E402
from deepstream_yolo.model_cache import discover_size  # noqa: E402
from deepstream_yolo.paths import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_RTSP_URL,
    GENERATED_CONFIG_DIR,
    SETUP_SCRIPT,
)
from deepstream_yolo.pipeline import element, on_message  # noqa: E402
from print_rtsp_timestamps import StageStats, reference_unix_ns, rtsp_protocol_flags  # noqa: E402

STAGE_ORDER = ("depay", "parse", "decode", "mux", "nvinfer")


def find_infer_config() -> Path:
    """Newest generated primary nvinfer config; static configs no longer exist."""
    matches = sorted(
        GENERATED_CONFIG_DIR.glob("config_infer_primary_*.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No generated primary nvinfer config in {GENERATED_CONFIG_DIR}. "
            f"Generate one with: {SETUP_SCRIPT} {DEFAULT_MODEL} 640 "
            "(or run src/parser_app.py once), then retry, or pass --infer-config explicitly."
        )
    return matches[0]


def buffer_probe(stats: StageStats, printer):
    """Pre-mux probe: only GstReferenceTimestampMeta can carry wall-clock time here."""

    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        recv_ns = time.time_ns()
        timestamp = reference_unix_ns(buffer)
        source = "ref" if timestamp is not None else "none"
        stats.record(recv_ns, source, timestamp)
        printer(stats, recv_ns, source, timestamp, getattr(buffer, "pts", None))
        return Gst.PadProbeReturn.OK

    return _probe


def batch_probe(stats: StageStats, printer):
    """Post-mux probe: NvDsFrameMeta.ntp_timestamp, reference meta, then PTS."""

    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        recv_ns = time.time_ns()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            timestamp = reference_unix_ns(buffer)
            source = "ref" if timestamp is not None else "no-batch-meta"
            stats.record(recv_ns, source, timestamp)
            printer(stats, recv_ns, source, timestamp, getattr(buffer, "pts", None))
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            source, timestamp = frame_timestamp(frame_meta, buffer)
            stats.record(recv_ns, source, timestamp)
            printer(stats, recv_ns, source, timestamp, getattr(buffer, "pts", None))
            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe


def make_printer(enabled_stages: set, limit: int, loop: GLib.MainLoop, counter: dict):
    def _print(stats: StageStats, recv_ns: int, source: str, timestamp: int | None, pts: int | None) -> None:
        if stats.stage in enabled_stages:
            print(stats.line(recv_ns, source, timestamp, pts), flush=True)
        if stats.stage == STAGE_ORDER[-1]:
            counter["frames"] += 1
            if limit and counter["frames"] >= limit:
                GLib.idle_add(loop.quit)

    return _print


def parse_stage_selection(text: str) -> set:
    value = text.strip().lower()
    if value in {"all", "*"}:
        return set(STAGE_ORDER)
    if value in {"none", ""}:
        return set()

    stages = set()
    for name in value.replace(" ", "").split(","):
        if name not in STAGE_ORDER:
            raise ValueError(f"Unknown stage {name!r}; expected {','.join(STAGE_ORDER)}, all, or none")
        stages.add(name)
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bisect reference-timestamp survival through nvinfer.")
    parser.add_argument("--uri", default=DEFAULT_RTSP_URL)
    parser.add_argument("--latency", type=int, default=0, help="rtspsrc jitterbuffer latency in ms.")
    parser.add_argument("--protocols", default="tcp")
    parser.add_argument("--codec", choices=("h264", "h265"), default="h264")
    parser.add_argument("--infer-config", default=None, help="Override the generated nvinfer config.")
    parser.add_argument("--width", type=int, default=0, help="nvstreammux width; 0 discovers it from the stream.")
    parser.add_argument("--height", type=int, default=0, help="nvstreammux height; 0 discovers it from the stream.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N inferred frames; 0 runs until EOS.")
    parser.add_argument(
        "--print-stages",
        default="nvinfer",
        help=f"Per-buffer lines to print: comma list of {','.join(STAGE_ORDER)}, all, or none.",
    )
    return parser.parse_args()


def build_pipeline(
    args: argparse.Namespace,
    infer_config: Path,
    mux_size: tuple[int, int],
) -> tuple[Gst.Pipeline, dict]:
    pipeline = Gst.Pipeline.new("rtsp-nvinfer-timestamp-test")

    source = element("rtspsrc", "source")
    depay = element(f"rtp{args.codec}depay", "depay")
    parse = element(f"{args.codec}parse", "parse")
    decoder = element("nvv4l2decoder", "decoder")
    queue = element("queue", "queue")
    mux = element("nvstreammux", "mux")
    infer = element("nvinfer", "infer")
    sink = element("fakesink", "sink")

    source.set_property("location", args.uri)
    source.set_property("latency", max(0, int(args.latency)))
    source.set_property("drop-on-latency", True)
    source.set_property("protocols", rtsp_protocol_flags(args.protocols))
    source.set_property("ntp-sync", True)
    source.set_property("add-reference-timestamp-meta", True)

    mux.set_property("batch-size", 1)
    mux.set_property("width", mux_size[0])
    mux.set_property("height", mux_size[1])
    mux.set_property("live-source", 1)
    mux.set_property("batched-push-timeout", 0)
    # attach-sys-ts=false keeps nvstreammux's ntp_timestamp derived from the stream,
    # which is exactly what this test is checking for.
    if mux.find_property("attach-sys-ts"):
        mux.set_property("attach-sys-ts", False)

    infer.set_property("config-file-path", str(infer_config))

    queue.set_property("leaky", 2)
    queue.set_property("max-size-buffers", 1)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)

    sink.set_property("sync", False)
    sink.set_property("qos", False)

    for elem in (source, depay, parse, decoder, queue, mux, infer, sink):
        pipeline.add(elem)

    def _on_pad_added(_src, pad, target):
        caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()
        if "application/x-rtp" in caps and not target.get_static_pad("sink").is_linked():
            pad.link(target.get_static_pad("sink"))

    source.connect("pad-added", _on_pad_added, depay)

    depay.link(parse)
    parse.link(decoder)
    decoder.link(queue)
    queue.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
    mux.link(infer)
    infer.link(sink)

    pads = {
        "depay": depay.get_static_pad("src"),
        "parse": parse.get_static_pad("src"),
        "decode": decoder.get_static_pad("src"),
        "mux": mux.get_static_pad("src"),
        "nvinfer": infer.get_static_pad("src"),
    }
    return pipeline, pads


def report(stats: dict) -> None:
    for name in STAGE_ORDER:
        print(stats[name].summary(), flush=True)

    survived = [name for name in STAGE_ORDER if stats[name].wall_clock_buffers]
    if not survived:
        print(
            "No stage saw a wall-clock timestamp: the RTSP server is not sending RTCP "
            "sender reports, so nothing downstream can recover stream time.",
            flush=True,
        )
        return

    lost = [
        name
        for index, name in enumerate(STAGE_ORDER)
        if not stats[name].wall_clock_buffers
        and any(stats[earlier].wall_clock_buffers for earlier in STAGE_ORDER[:index])
    ]
    print(f"wall-clock timestamps survive to: {', '.join(survived)}", flush=True)
    if lost:
        print(f"first stage that loses the timestamp: {lost[0]}", flush=True)
    else:
        print("no stage drops the timestamp", flush=True)


def main() -> int:
    args = parse_args()
    Gst.init(None)

    infer_config = Path(args.infer_config) if args.infer_config else find_infer_config()
    if not infer_config.exists():
        raise FileNotFoundError(f"Missing nvinfer config: {infer_config}")

    if args.width > 0 and args.height > 0:
        mux_size = (args.width, args.height)
    else:
        mux_size = discover_size(args.uri)

    print(f"uri={args.uri} mux={mux_size[0]}x{mux_size[1]} infer_config={infer_config}", flush=True)

    loop = GLib.MainLoop()
    pipeline, pads = build_pipeline(args, infer_config, mux_size)

    stats = {name: StageStats(name) for name in STAGE_ORDER}
    printer = make_printer(parse_stage_selection(args.print_stages), args.frames, loop, {"frames": 0})
    for name in STAGE_ORDER:
        probe = batch_probe if name in {"mux", "nvinfer"} else buffer_probe
        pads[name].add_probe(Gst.PadProbeType.BUFFER, probe(stats[name], printer), None)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

    report(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
