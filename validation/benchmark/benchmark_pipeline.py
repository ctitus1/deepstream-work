#!/usr/bin/env python3
"""Benchmark end-to-end FPS and per-stage latency for the deployed pipeline.

Builds on the package timing infrastructure rather than re-measuring it:
``deepstream_yolo.timing.TimeLog`` supplies the per-frame stage timestamps
(mux/infer/assessment/convert/osd/sink) and ``AssessmentTiming`` supplies the
same ``detect_ms``/``assess_ms`` values the parser and ROS apps report, so the
numbers here line up with the app's ASSESS log lines. This tool adds
percentiles (p50/p90/p99), warmup trimming, and optional GPU sampling.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import gi

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from deepstream_yolo.assessment_runtime import AssessmentTiming, assessment_probe  # noqa: E402
from deepstream_yolo.detection_overlay import bbox_probe  # noqa: E402
from deepstream_yolo.model_cache import discover_size, ensure_assessment_model, ensure_model  # noqa: E402
from deepstream_yolo.paths import DEFAULT_STREAM, resolve_project_path  # noqa: E402
from deepstream_yolo.pipeline import build_pipeline, on_message  # noqa: E402
from deepstream_yolo.stream_source import resolve_stream_source  # noqa: E402
from deepstream_yolo.timing import TimeLog  # noqa: E402

# Segment name per destination stage, matching TimeLog._print_timing().
SEGMENT_NAMES = {
    "infer": "detect",
    "assessment": "assess",
    "convert": "convert",
    "osd": "osd",
    "sink": "sink",
}
PERCENTILES = (50, 90, 99)


def percentile(values: list, pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def distribution(values: list) -> dict:
    if not values:
        return {"n": 0}
    stats = {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": float(min(values)),
        "max": float(max(values)),
    }
    for pct in PERCENTILES:
        stats[f"p{pct}"] = percentile(values, pct)
    return stats


class GpuSampler(threading.Thread):
    """Optional nvidia-smi sampler; absent or failing nvidia-smi degrades to no data."""

    QUERY = "utilization.gpu,utilization.memory,memory.used"

    def __init__(self, interval: float, gpu_index: int):
        super().__init__(daemon=True)
        self.interval = interval
        self.gpu_index = gpu_index
        self.binary = shutil.which("nvidia-smi")
        self.stop_event = threading.Event()
        self.utilization = []
        self.memory_utilization = []
        self.memory_used = []
        self.error = None if self.binary else "nvidia-smi not found"

    @property
    def available(self) -> bool:
        return self.binary is not None

    def run(self) -> None:
        if not self.binary:
            return
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def sample(self) -> None:
        command = [
            self.binary,
            f"--query-gpu={self.QUERY}",
            "--format=csv,noheader,nounits",
            f"--id={self.gpu_index}",
        ]
        try:
            output = subprocess.check_output(command, text=True, timeout=5, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            self.error = f"nvidia-smi sampling failed: {exc}"
            self.stop_event.set()
            return

        fields = [field.strip() for field in output.strip().splitlines()[0].split(",")]
        try:
            self.utilization.append(float(fields[0]))
            self.memory_utilization.append(float(fields[1]))
            self.memory_used.append(float(fields[2]))
        except (IndexError, ValueError):
            self.error = f"unexpected nvidia-smi output: {output.strip()!r}"
            self.stop_event.set()

    def stop(self) -> None:
        self.stop_event.set()

    def report(self) -> dict:
        if not self.utilization:
            return {"available": False, "error": self.error or "no samples collected"}
        return {
            "available": True,
            "gpu_index": self.gpu_index,
            "samples": len(self.utilization),
            "utilization_pct": distribution(self.utilization),
            "memory_utilization_pct": distribution(self.memory_utilization),
            "memory_used_mib": distribution(self.memory_used),
        }


class FrameCounter:
    """Sink-pad counter that stops the loop once enough frames are measured."""

    def __init__(self, max_frames: int, loop):
        self.max_frames = max_frames
        self.loop = loop
        self.frames = 0
        self.stopping = False

    def probe(self, _pad, _info, _data):
        self.frames += 1
        if self.max_frames and self.frames >= self.max_frames and not self.stopping:
            self.stopping = True
            GLib.idle_add(self.loop.quit)
        return Gst.PadProbeReturn.OK


def compute_time_collector(samples: list):
    def _sink(_buffer, _frame_num, _timestamp_source, _timestamp, _rows, compute_times):
        if compute_times is not None:
            samples.append(compute_times)

    return _sink


def stage_latencies(times: dict, stages: tuple, warmup_frames: int) -> dict:
    """Per-frame stage durations in ms, from the timestamps TimeLog recorded."""
    segments = {SEGMENT_NAMES[dest]: [] for _, dest in zip(stages[:-1], stages[1:])}
    segments["total"] = []

    for frame_num, marks in times.items():
        if frame_num < warmup_frames or not all(stage in marks for stage in stages):
            continue
        for source, dest in zip(stages[:-1], stages[1:]):
            segments[SEGMENT_NAMES[dest]].append((marks[dest] - marks[source]) * 1000.0)
        segments["total"].append((marks[stages[-1]] - marks[stages[0]]) * 1000.0)

    return {name: distribution(values) for name, values in segments.items()}


def throughput(times: dict, warmup_frames: int) -> dict:
    sink_times = sorted(
        marks["sink"]
        for frame_num, marks in times.items()
        if frame_num >= warmup_frames and "sink" in marks
    )
    if len(sink_times) < 2:
        return {"frames": len(sink_times)}

    elapsed = sink_times[-1] - sink_times[0]
    intervals = [(later - earlier) * 1000.0 for earlier, later in zip(sink_times[:-1], sink_times[1:])]
    return {
        "frames": len(sink_times),
        "elapsed_s": elapsed,
        "fps": (len(sink_times) - 1) / elapsed if elapsed > 0 else 0.0,
        "frame_interval_ms": distribution(intervals),
    }


def assessment_report(samples: list) -> dict:
    detect = [sample.detect_ms for sample in samples if sample.detect_ms is not None]
    assess = [sample.assess_ms for sample in samples if sample.assess_ms is not None]
    return {
        "assessed_frames": len(samples),
        "detect_ms": distribution(detect),
        "assess_ms": distribution(assess),
    }


def format_distribution(name: str, stats: dict) -> str:
    if not stats.get("n"):
        return f"  {name:<10} no samples"
    percentile_text = " ".join(f"p{pct}={stats[f'p{pct}']:.2f}" for pct in PERCENTILES)
    return (
        f"  {name:<10} n={stats['n']:<6} mean={stats['mean']:.2f} {percentile_text} "
        f"min={stats['min']:.2f} max={stats['max']:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DeepStream FPS and per-stage latency.")
    parser.add_argument("--stream", default=str(DEFAULT_STREAM))
    parser.add_argument("--model", default="yolo12x-custom.pt")
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to run; 0 runs until EOS.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N sink frames; 0 disables.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames dropped before measuring.")
    parser.add_argument("--assessment-model", default="models/injury.pt")
    parser.add_argument("--assessment-batch-size", type=int, default=8)
    assessment = parser.add_mutually_exclusive_group()
    assessment.add_argument("--enable-assessment", dest="enable_assessment", action="store_true", default=True)
    assessment.add_argument("--no-assessment", dest="enable_assessment", action="store_false")
    parser.add_argument("--overlay", action="store_true", help="Include the OSD bbox probe in the measured path.")
    parser.add_argument("--display", action="store_true", help="Render to a window; sink sync then caps FPS.")
    parser.add_argument("--rtsp-latency-ms", type=int, default=0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5, help="Seconds between nvidia-smi samples.")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--no-gpu-sampling", dest="gpu_sampling", action="store_false", default=True)
    parser.add_argument("--json", default=None, help="Write the full report to this JSON path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress periodic OUTPUT_FPS lines.")
    return parser.parse_args()


def print_report(report: dict) -> None:
    run = report["run"]
    rate = report["throughput"]
    print("")
    print(
        f"run         stream={run['stream']} video={run['source_width']}x{run['source_height']} "
        f"model={run['model_width']}x{run['model_height']} assessment={run['assessment']}"
    )
    print(f"            frames={rate.get('frames', 0)} warmup={run['warmup_frames']} display={run['display']}")
    if rate.get("frames", 0) >= 2:
        print(f"throughput  fps={rate['fps']:.2f} over {rate['elapsed_s']:.1f}s")
        print(format_distribution("interval", rate["frame_interval_ms"]))
    else:
        print("throughput  not enough frames to measure")

    print("latency ms  (per frame, pad-probe timestamps)")
    for name, stats in report["stages"].items():
        print(format_distribution(name, stats))

    assessment = report.get("assessment_runtime")
    if assessment and assessment["assessed_frames"]:
        print("assessment  (as reported by assessment_runtime)")
        print(format_distribution("detect_ms", assessment["detect_ms"]))
        print(format_distribution("assess_ms", assessment["assess_ms"]))

    gpu = report["gpu"]
    if gpu.get("available"):
        util = gpu["utilization_pct"]
        memory = gpu["memory_used_mib"]
        print(
            f"gpu         index={gpu['gpu_index']} samples={gpu['samples']} "
            f"util p50={util['p50']:.0f}% p90={util['p90']:.0f}% max={util['max']:.0f}% "
            f"mem p50={memory['p50']:.0f}MiB max={memory['max']:.0f}MiB"
        )
    else:
        print(f"gpu         unavailable ({gpu.get('error')})")


def main() -> int:
    args = parse_args()
    stream = resolve_stream_source(args.stream)

    Gst.init(None)
    src_w, src_h = discover_size(stream.uri)
    model_w, model_h, config = ensure_model(args.model, stream, args.long_side, src_w, src_h, args.conf)

    assessment_config = None
    if args.enable_assessment:
        _, assessment_config = ensure_assessment_model(args.assessment_model, args.assessment_batch_size)

    print(
        f"stream={stream.display} video={src_w}x{src_h} model={model_w}x{model_h} "
        f"assessment={bool(assessment_config)} duration={args.duration}s",
        flush=True,
    )
    if stream.is_rtsp:
        print(
            "note: RTSP input is paced by the stream clock, so FPS is capped by the sender; "
            "benchmark a local file for maximum throughput.",
            flush=True,
        )

    loop = GLib.MainLoop()
    parts = build_pipeline(
        stream,
        src_w,
        src_h,
        config,
        assessment_config,
        rtsp_latency_ms=args.rtsp_latency_ms,
        display=args.display,
    )

    # An infinite print interval turns TimeLog into a pure per-frame stage recorder.
    timer = TimeLog(timing_interval=float("inf"))
    stages = ["mux", "infer"]
    timing_pads = [(parts.streammux.get_static_pad("src"), "mux"), (parts.pgie.get_static_pad("src"), "infer")]
    if args.overlay:
        parts.pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, bbox_probe(args.conf), None)

    compute_samples = []
    assessment_timing = None
    if parts.sgie:
        assessment_timing = AssessmentTiming()
        parts.streammux.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, assessment_timing.mark_start, None
        )
        parts.pgie.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, assessment_timing.mark_detect_done, None
        )
        parts.sgie.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            assessment_probe(None, timing=assessment_timing, frame_sink=compute_time_collector(compute_samples)),
            None,
        )
        stages.append("assessment")
        timing_pads.append((parts.sgie.get_static_pad("src"), "assessment"))

    stages.extend(("convert", "osd", "sink"))
    timing_pads.extend(
        [
            (parts.caps.get_static_pad("src"), "convert"),
            (parts.osd.get_static_pad("src"), "osd"),
            (parts.sink.get_static_pad("sink"), "sink"),
        ]
    )
    for pad, stage in timing_pads:
        pad.add_probe(Gst.PadProbeType.BUFFER, timer.mark(stage), None)

    counter = FrameCounter(args.max_frames, loop)
    sink_pad = parts.sink.get_static_pad("sink")
    sink_pad.add_probe(Gst.PadProbeType.BUFFER, counter.probe, None)
    if not args.quiet:
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, timer.fps_probe, None)

    sampler = GpuSampler(args.gpu_sample_interval, args.gpu_index)
    if args.gpu_sampling and sampler.available:
        sampler.start()

    bus = parts.pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    if args.duration > 0:

        def stop_after_duration() -> bool:
            loop.quit()
            return False

        GLib.timeout_add(int(args.duration * 1000), stop_after_duration)

    started = time.perf_counter()
    parts.pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        parts.pipeline.set_state(Gst.State.NULL)
        sampler.stop()

    report = {
        "run": {
            "stream": stream.display,
            "model": args.model,
            "infer_config": str(config),
            "source_width": src_w,
            "source_height": src_h,
            "model_width": model_w,
            "model_height": model_h,
            "assessment": bool(assessment_config),
            "overlay": args.overlay,
            "display": args.display,
            "warmup_frames": args.warmup_frames,
            "wall_clock_s": time.perf_counter() - started,
            "sink_frames": counter.frames,
        },
        "throughput": throughput(timer.times, args.warmup_frames),
        "stages": stage_latencies(timer.times, tuple(stages), args.warmup_frames),
        "assessment_runtime": assessment_report(compute_samples) if parts.sgie else None,
        "gpu": sampler.report() if args.gpu_sampling else {"available": False, "error": "sampling disabled"},
    }
    print_report(report)

    if args.json:
        output = resolve_project_path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report      {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
