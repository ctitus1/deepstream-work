#!/usr/bin/env python3
"""Interactive DeepStream parser app.

This entrypoint prepares runtime options and model configs, calls
``deepstream_yolo.pipeline.build_pipeline()`` for the display-oriented
GStreamer graph, attaches parser-specific probes, and runs the GLib/keyboard
loop. The shared pipeline owns decode, inference, OSD, and sink elements; this
file owns app policy such as logging, fresh-assessment display, pacing for local
files, and debug timing probes.
"""

import argparse
import sys
from pathlib import Path

from deepstream_yolo.runtime import (
    maybe_start_gst_scan_warning_filter,
    stop_gst_scan_warning_filter,
)

GST_SCAN_WARNING_FILTER = maybe_start_gst_scan_warning_filter(sys.argv)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from deepstream_yolo.assessment_runtime import AssessmentReporter, AssessmentTiming, assessment_probe
from deepstream_yolo.controls import KeyboardControls, RateLimiter
from deepstream_yolo.detection_overlay import bbox_probe
from deepstream_yolo.model_cache import discover_size, ensure_assessment_model, ensure_model
from deepstream_yolo.motion import (
    MotionConfig,
    MotionStore,
    motion_overlay_probe,
    motion_probe,
)
from deepstream_yolo.paths import DEFAULT_ASSESSMENT_MODEL, DEFAULT_MODEL, DEFAULT_STREAM
from deepstream_yolo.pipeline import build_pipeline, on_message
from deepstream_yolo.media import StreamSource, resolve_record_path, resolve_stream_source
from deepstream_yolo.runtime import install_shutdown_handlers
from deepstream_yolo.timing import TimeLog


class RuntimeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        stop_gst_scan_warning_filter(GST_SCAN_WARNING_FILTER)
        super().error(message)


def parse_args() -> argparse.Namespace:
    parser = RuntimeArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--stream", default=str(DEFAULT_STREAM))
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no-motion",
        dest="enable_motion",
        action="store_false",
        help="Disable the optical-flow bulk-motion branch and its grey overlay boxes.",
    )
    parser.set_defaults(enable_motion=True)
    parser.add_argument(
        "--motion-max-boxes",
        type=int,
        default=MotionConfig.max_boxes,
        help="Cap on grey motion boxes per frame. Default: %(default)s",
    )
    parser.add_argument(
        "--base-fps",
        type=float,
        default=30.0,
        help="Playback-rate baseline for local files; ignored for live RTSP streams.",
    )
    parser.add_argument(
        "--rtsp-latency-ms",
        type=int,
        default=0,
        help="RTSP jitterbuffer latency before old network packets are dropped; default 0.",
    )
    parser.add_argument("--show-gst-scan-warnings", action="store_true")
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Record the annotated video to PATH; omit PATH for an auto-named mp4 under outputs/.",
    )
    assessment_group = parser.add_mutually_exclusive_group()
    assessment_group.add_argument(
        "--enable-assessment",
        dest="enable_assessment",
        action="store_true",
        default=True,
        help="Enable injury assessment; default.",
    )
    assessment_group.add_argument(
        "--no-assessment",
        dest="enable_assessment",
        action="store_false",
        help="Disable injury assessment.",
    )
    parser.add_argument("--assessment-model", default=DEFAULT_ASSESSMENT_MODEL)
    parser.add_argument("--assessment-batch-size", type=int, default=8)
    parser.add_argument(
        "--assessment-log-interval",
        type=float,
        default=0.0,
        help="Seconds between sampled assessment-log frames; 0 logs every assessment, negative disables.",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--show-assessed-only",
        dest="show_assessed_only",
        action="store_true",
        default=False,
        help="Display only frames with fresh assessment output.",
    )
    display_group.add_argument(
        "--show-all-frames",
        dest="show_assessed_only",
        action="store_false",
        help="Display every frame, including frames without fresh assessment output.",
    )
    args = parser.parse_args()
    if args.show_assessed_only and not args.enable_assessment:
        parser.error("--show-assessed-only requires --enable-assessment")
    return args


def print_runtime_info(
    args: argparse.Namespace,
    stream: StreamSource,
    src_size: tuple[int, int],
    model_size: tuple[int, int],
    config,
    assessment_meta: dict | None,
    assessment_config,
) -> None:
    print(
        f"stream={stream.display} "
        f"video={src_size[0]}x{src_size[1]} "
        f"model={model_size[0]}x{model_size[1]} "
        f"conf={args.conf} "
        f"config={config}"
    )
    if assessment_config:
        print(
            "assessment="
            f"{assessment_meta.get('architecture', 'injury model')} "
            f"batch={args.assessment_batch_size} "
            f"config={assessment_config}",
            flush=True,
        )


def attach_runtime_probes(parts, args, stream: StreamSource) -> RateLimiter:
    """Attach parser-app probes to the shared DeepStream pipeline."""
    assessment_timing = None
    if parts.sgie:
        # Measure detect and assessment compute windows from the shared pipeline.
        assessment_timing = AssessmentTiming()
        parts.streammux.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            assessment_timing.mark_start,
            None,
        )
        parts.pgie.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            assessment_timing.mark_detect_done,
            None,
        )

    parts.pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        bbox_probe(args.conf),
        None,
    )
    # Deliberately a second `if parts.sgie` rather than one merged block: probes
    # on the pgie src pad fire in attach order, so mark_detect_done must be
    # attached above, before bbox_probe. Merging pushes it after the box drawing
    # and every detect_ms measurement absorbs that work.
    if parts.sgie:
        # Parse secondary tensor output, render assessment text, and optionally drop stale frames.
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

    limiter = RateLimiter(base_fps=args.base_fps, enabled=not stream.is_rtsp)
    if limiter.enabled:
        # Local files can outrun real time, so pace only non-RTSP playback.
        parts.sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, limiter.probe, None)
    return limiter


def attach_debug_probes(parts) -> None:
    """Attach optional stage timing logs without changing pipeline behavior."""
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


class ShutdownTracker:
    """Remembers how the pipeline terminated, so teardown can flush accordingly."""

    def __init__(self):
        self.saw_eos = False
        self.saw_error = False

    def on_message(self, _bus, msg, _data):
        if msg.type == Gst.MessageType.EOS:
            self.saw_eos = True
        elif msg.type == Gst.MessageType.ERROR:
            self.saw_error = True
        return True


def finish_recording(parts, tracker, timeout_s: float = 10.0) -> None:
    """Flush mp4mux with a clean EOS, otherwise the recording has no moov atom."""
    if parts.record_sink is None:
        return

    location = Path(parts.record_sink.get_property("location"))
    if tracker.saw_error and not tracker.saw_eos:
        # A broken pipeline cannot drain; flushing would just block until timeout.
        print(f"record: pipeline errored, {location} is likely unplayable", file=sys.stderr)
    elif not tracker.saw_eos:
        # A keyboard quit stops the loop without EOS, and a paused pipeline
        # cannot drain, so resume before asking the graph to finish.
        parts.pipeline.set_state(Gst.State.PLAYING)
        parts.pipeline.get_state(Gst.SECOND)
        bus = parts.pipeline.get_bus()
        bus.remove_signal_watch()
        parts.pipeline.send_event(Gst.Event.new_eos())
        msg = bus.timed_pop_filtered(
            int(timeout_s * Gst.SECOND),
            Gst.MessageType.EOS | Gst.MessageType.ERROR,
        )
        if msg is None:
            print(f"record: timed out waiting for EOS, {location} may be unplayable", file=sys.stderr)
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"record ERROR: {err}\nDEBUG: {dbg}", file=sys.stderr)

    parts.pipeline.set_state(Gst.State.NULL)
    if location.exists():
        print(f"recorded={location} bytes={location.stat().st_size}", flush=True)
    else:
        print(f"record: {location} missing", file=sys.stderr)


def main():
    args = parse_args()

    # Resolve inside the filter's scope: the stderr pump swallows anything still
    # in flight at exit, so a missing-stream error raised before the filter is
    # stopped would never reach the terminal.
    try:
        stream = resolve_stream_source(args.stream)
        # Discover source geometry before creating model-specific DeepStream configs.
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
        args, stream, (src_w, src_h), (model_w, model_h), config, assessment_meta, assessment_config
    )

    record_path = resolve_record_path(args.record, stream.uri) if args.record is not None else None
    if record_path:
        print(f"record={record_path}", flush=True)

    motion_cfg = (
        MotionConfig(max_boxes=max(1, int(args.motion_max_boxes)))
        if args.enable_motion
        else None
    )
    motion_store = MotionStore() if motion_cfg else None

    # Build once, then attach app-specific probes around the shared pipeline.
    parts = build_pipeline(
        stream,
        src_w,
        src_h,
        config,
        assessment_config,
        rtsp_latency_ms=args.rtsp_latency_ms,
        record_path=record_path,
        motion=motion_cfg,
    )
    limiter = attach_runtime_probes(parts, args, stream)
    if args.debug:
        attach_debug_probes(parts)

    if motion_store is not None and parts.motion_of is not None:
        # Read side: on the nvof source pad, inside the motion branch's own
        # streaming thread. It never touches the inference path's buffers.
        parts.motion_of.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            motion_probe(motion_cfg, motion_store, src_w, src_h, debug=args.debug),
            None,
        )
        # Draw side: on the OSD sink pad, after every inference stage, so the
        # grey boxes land on top of the detection boxes and no overlay can
        # reach a frame before it has been inferred on.
        parts.osd.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER,
            motion_overlay_probe(motion_store),
            None,
        )

    loop = GLib.MainLoop()
    # Makes SIGTERM/SIGHUP take the same clean exit as Ctrl-C, so `docker stop`
    # and a closed terminal still finalize an in-progress recording.
    install_shutdown_handlers(loop)
    controls = KeyboardControls(parts.pipeline, loop, limiter) if sys.stdin.isatty() else None
    if controls:
        controls.start()

    # Run until EOS, error, or a keyboard/UI stop request.
    tracker = ShutdownTracker()
    bus = parts.pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", tracker.on_message, None)
    bus.connect("message", on_message, loop)

    parts.pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        if controls:
            controls.stop()
        finish_recording(parts, tracker)
        parts.pipeline.set_state(Gst.State.NULL)
        if motion_store is not None:
            # frames_in is what reached the motion branch, frames_processed is
            # what produced a flow field. A gap means the branch could not keep
            # up with the source, which is the one thing its design forbids.
            seen, processed = motion_store.stats()
            print(f"motion frames_in={seen} processed={processed}", flush=True)


if __name__ == "__main__":
    main()
