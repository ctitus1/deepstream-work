#!/usr/bin/env python3
import argparse
import sys

from deepstream_yolo.gst_warnings import (
    maybe_start_gst_scan_warning_filter,
    stop_gst_scan_warning_filter,
)

GST_SCAN_WARNING_FILTER = maybe_start_gst_scan_warning_filter(sys.argv)

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import GLib, Gst

from deepstream_yolo.assessment_runtime import AssessmentReporter, AssessmentTiming, assessment_probe
from deepstream_yolo.controls import KeyboardControls, RateLimiter
from deepstream_yolo.detection_overlay import bbox_probe
from deepstream_yolo.model_cache import discover_size, ensure_assessment_model, ensure_model
from deepstream_yolo.paths import DEFAULT_STREAM
from deepstream_yolo.pipeline import build_pipeline, on_message
from deepstream_yolo.stream_source import StreamSource, resolve_stream_source
from deepstream_yolo.timing import TimeLog


class RuntimeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        stop_gst_scan_warning_filter(GST_SCAN_WARNING_FILTER)
        super().error(message)


def parse_args() -> argparse.Namespace:
    parser = RuntimeArgumentParser()
    parser.add_argument("--model", default="yolo12x.pt")
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--stream", default=str(DEFAULT_STREAM))
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--base-fps", type=float, default=30.0)
    parser.add_argument(
        "--rtsp-latency-ms",
        type=int,
        default=0,
        help="RTSP jitterbuffer latency before old network packets are dropped; default 0.",
    )
    parser.add_argument("--show-gst-scan-warnings", action="store_true")
    parser.add_argument("--enable-assessment", action="store_true")
    parser.add_argument("--assessment-model", default="models/injury.pt")
    parser.add_argument("--assessment-batch-size", type=int, default=8)
    parser.add_argument(
        "--assessment-log-interval",
        type=float,
        default=0.0,
        help="Seconds between sampled assessment-log frames; 0 logs every assessment, negative disables.",
    )
    parser.add_argument("--show-assessed-only", action="store_true")
    args = parser.parse_args()
    if args.show_assessed_only and not args.enable_assessment:
        parser.error("--show-assessed-only requires --enable-assessment")
    return args


def print_runtime_info(
    stream: StreamSource,
    src_w: int,
    src_h: int,
    model_w: int,
    model_h: int,
    conf: float,
    config,
    assessment_meta: dict | None,
    assessment_config,
    assessment_batch_size: int,
) -> None:
    print(
        f"stream={stream.display} "
        f"video={src_w}x{src_h} "
        f"model={model_w}x{model_h} "
        f"conf={conf} "
        f"config={config}"
    )
    if assessment_config:
        print(
            "assessment="
            f"{assessment_meta.get('architecture', 'injury model')} "
            f"batch={assessment_batch_size} "
            f"config={assessment_config}",
            flush=True,
        )


def attach_runtime_probes(parts, args) -> RateLimiter:
    parts.pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        bbox_probe(args.conf),
        None,
    )
    if parts.sgie:
        assessment_timing = AssessmentTiming()
        parts.streammux.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            assessment_timing.mark_start,
            None,
        )
        reporter = AssessmentReporter(args.assessment_log_interval)
        parts.sgie.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            assessment_probe(
                reporter,
                timing=assessment_timing,
                show_assessed_only=args.show_assessed_only,
            ),
            None,
        )

    limiter = RateLimiter(base_fps=args.base_fps)
    parts.sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, limiter.probe, None)
    return limiter


def attach_debug_probes(parts) -> None:
    timer = TimeLog()
    parts.sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, timer.fps_probe, None)
    timing_pads = [
        (parts.streammux.get_static_pad("src"), "mux"),
        (parts.pgie.get_static_pad("src"), "infer"),
    ]
    if parts.sgie:
        timing_pads.append((parts.sgie.get_static_pad("src"), "assessment"))
    timing_pads.extend(
        [
            (parts.caps.get_static_pad("src"), "convert"),
            (parts.osd.get_static_pad("src"), "osd"),
            (parts.sink.get_static_pad("sink"), "sink"),
        ]
    )
    for pad, stage in timing_pads:
        pad.add_probe(Gst.PadProbeType.BUFFER, timer.mark(stage), None)


def main():
    args = parse_args()
    stream = resolve_stream_source(args.stream)

    try:
        Gst.init(None)
        src_w, src_h = discover_size(stream.uri)
    finally:
        stop_gst_scan_warning_filter(GST_SCAN_WARNING_FILTER)

    model_w, model_h, config = ensure_model(
        args.model,
        stream,
        args.long_side,
        src_w,
        src_h,
        args.conf,
    )

    assessment_config = None
    assessment_meta = None
    if args.enable_assessment:
        assessment_meta, assessment_config = ensure_assessment_model(
            args.assessment_model,
            args.assessment_batch_size,
        )

    print_runtime_info(
        stream,
        src_w,
        src_h,
        model_w,
        model_h,
        args.conf,
        config,
        assessment_meta,
        assessment_config,
        args.assessment_batch_size,
    )

    parts = build_pipeline(
        stream,
        src_w,
        src_h,
        config,
        assessment_config,
        rtsp_latency_ms=args.rtsp_latency_ms,
    )
    limiter = attach_runtime_probes(parts, args)
    if args.debug:
        attach_debug_probes(parts)

    loop = GLib.MainLoop()
    controls = KeyboardControls(parts.pipeline, loop, limiter) if sys.stdin.isatty() else None
    if controls:
        controls.start()

    bus = parts.pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    parts.pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        if controls:
            controls.stop()
        parts.pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
