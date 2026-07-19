#!/usr/bin/env python3
"""Print RTSP reference (NTP) timestamps for each depayloaded buffer.

Plain GStreamer only: no DeepStream elements are involved, so this isolates
whether the RTSP source carries ``GstReferenceTimestampMeta`` at all, before
``nvstreammux``/``nvinfer`` are in the picture. Use
``rtsp_nvinfer_timestamp_test.py`` for the stage-by-stage bisection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
import time
from pathlib import Path

import gi

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from deepstream_yolo.paths import DEFAULT_RTSP_URL  # noqa: E402
from deepstream_yolo.pipeline import element, on_message, on_rtsp_pad_added  # noqa: E402

NTP_TO_UNIX_SECONDS = 2_208_988_800
NS_PER_SEC = 1_000_000_000
WALL_CLOCK_SOURCES = frozenset({"ntp", "ref"})
# GstRTSPLowerTrans flags; set_property() needs the int, not the nick string.
RTSP_PROTOCOLS = {"udp": 0x1, "udp-mcast": 0x2, "tcp": 0x4, "http": 0x10, "tls": 0x20}


def rtsp_protocol_flags(text: str) -> int:
    flags = 0
    for name in text.replace(",", "+").split("+"):
        key = name.strip().lower()
        if not key:
            continue
        if key not in RTSP_PROTOCOLS:
            raise ValueError(f"Unknown RTSP protocol {name!r}; expected {'+'.join(RTSP_PROTOCOLS)}")
        flags |= RTSP_PROTOCOLS[key]
    if not flags:
        raise ValueError("No RTSP protocols selected")
    return flags


def unix_ns_to_utc(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / NS_PER_SEC, tz=dt.timezone.utc).isoformat(timespec="milliseconds")


def ntp_ns_to_unix_ns(ntp_ns: int) -> int:
    return ntp_ns - NTP_TO_UNIX_SECONDS * NS_PER_SEC


def seconds(ns: int | None) -> str:
    if ns is None or ns == Gst.CLOCK_TIME_NONE or ns < 0:
        return "NONE"
    return f"{ns / NS_PER_SEC:.3f}s"


def reference_unix_ns(buffer) -> int | None:
    """Unix ns from GstReferenceTimestampMeta, or None when the meta is absent."""
    getter = getattr(buffer, "get_reference_timestamp_meta", None)
    if getter is None:
        return None

    try:
        ref_meta = getter(None)
    except Exception:
        return None

    if ref_meta is None:
        return None

    timestamp = int(getattr(ref_meta, "timestamp", 0) or 0)
    if timestamp <= 0 or timestamp == Gst.CLOCK_TIME_NONE:
        return None
    return ntp_ns_to_unix_ns(timestamp)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


class StageStats:
    """Per-stage tally of timestamp survival and observed wall-clock latency."""

    def __init__(self, stage: str):
        self.stage = stage
        self.buffers = 0
        self.sources: dict[str, int] = {}
        self.latencies_ms: list[float] = []

    def record(self, recv_ns: int, source: str, timestamp: int | None) -> None:
        self.buffers += 1
        self.sources[source] = self.sources.get(source, 0) + 1
        if timestamp is not None and source in WALL_CLOCK_SOURCES:
            self.latencies_ms.append((recv_ns - timestamp) / 1e6)

    @property
    def wall_clock_buffers(self) -> int:
        return len(self.latencies_ms)

    def line(self, recv_ns: int, source: str, timestamp: int | None, pts: int | None) -> str:
        if timestamp is None or source not in WALL_CLOCK_SOURCES:
            return (
                f"{self.stage}: local time={unix_ns_to_utc(recv_ns)} "
                f"stream time=NONE latency=NONE source={source} "
                f"stream duration={seconds(pts)}"
            )
        return (
            f"{self.stage}: local time={unix_ns_to_utc(recv_ns)} "
            f"stream time={unix_ns_to_utc(timestamp)} "
            f"latency={(recv_ns - timestamp) / 1e6:.1f}ms source={source} "
            f"stream duration={seconds(pts)}"
        )

    def summary(self) -> str:
        sources = ",".join(f"{name}:{count}" for name, count in sorted(self.sources.items())) or "none"
        fields = [
            f"{self.stage}:",
            f"buffers={self.buffers}",
            f"wall_clock={self.wall_clock_buffers}",
            f"sources={sources}",
        ]
        if self.latencies_ms:
            fields.extend(
                [
                    f"latency_p50={statistics.median(self.latencies_ms):.1f}ms",
                    f"latency_p90={percentile(self.latencies_ms, 90):.1f}ms",
                    f"latency_min={min(self.latencies_ms):.1f}ms",
                    f"latency_max={max(self.latencies_ms):.1f}ms",
                ]
            )
        return " ".join(fields)


def make_handoff(stats: StageStats, quiet: bool, limit: int, loop: GLib.MainLoop):
    def _handoff(_sink, buffer, _pad):
        recv_ns = time.time_ns()
        timestamp = reference_unix_ns(buffer)
        source = "ref" if timestamp is not None else "none"
        stats.record(recv_ns, source, timestamp)

        if not quiet:
            print(stats.line(recv_ns, source, timestamp, getattr(buffer, "pts", None)), flush=True)
        if limit and stats.buffers >= limit:
            GLib.idle_add(loop.quit)

    return _handoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print RTSP reference (NTP) timestamps per buffer.")
    parser.add_argument("--uri", default=DEFAULT_RTSP_URL)
    parser.add_argument("--latency", type=int, default=100, help="rtspsrc jitterbuffer latency in ms.")
    parser.add_argument("--protocols", default="tcp", help="RTSP transports, e.g. tcp or udp+tcp.")
    parser.add_argument(
        "--ntp-sync",
        action="store_true",
        help="Ask rtspsrc to translate buffer PTS to the sender clock; reference meta is independent.",
    )
    parser.add_argument("--frames", type=int, default=0, help="Stop after N buffers; 0 runs until EOS.")
    parser.add_argument("--quiet", action="store_true", help="Print only the closing summary.")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace, loop: GLib.MainLoop) -> tuple[Gst.Pipeline, dict[str, StageStats]]:
    pipeline = Gst.Pipeline.new("rtsp-timestamps")
    source = element("rtspsrc", "source")
    source.set_property("location", args.uri)
    source.set_property("latency", max(0, int(args.latency)))
    source.set_property("protocols", rtsp_protocol_flags(args.protocols))
    source.set_property("add-reference-timestamp-meta", True)
    source.set_property("ntp-sync", bool(args.ntp_sync))
    pipeline.add(source)

    # One depay/parse/sink branch per codec: the branch that never links stays idle,
    # so a stream of either codec reports without guessing up front.
    depayloaders = {}
    stats = {}
    for codec in ("h264", "h265"):
        depay = element(f"rtp{codec}depay", f"{codec}-depay")
        parse = element(f"{codec}parse", f"{codec}-parse")
        sink = element("fakesink", f"{codec}-sink")
        sink.set_property("sync", False)
        sink.set_property("signal-handoffs", True)
        for elem in (depay, parse, sink):
            pipeline.add(elem)
        depay.link(parse)
        parse.link(sink)

        stage = StageStats(f"{codec}-parse")
        stats[codec] = stage
        sink.connect("handoff", make_handoff(stage, args.quiet, args.frames, loop))
        depayloaders[codec] = depay

    source.connect("pad-added", on_rtsp_pad_added, depayloaders)
    return pipeline, stats


def main() -> int:
    args = parse_args()
    Gst.init(None)

    loop = GLib.MainLoop()
    pipeline, stats = build_pipeline(args, loop)
    print(f"uri={args.uri} latency={args.latency}ms protocols={args.protocols}", flush=True)

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

    active = [stage for stage in stats.values() if stage.buffers]
    if not active:
        print("No buffers reached the sinks; check the RTSP URI and transport.", file=sys.stderr)
        return 1

    for stage in active:
        print(stage.summary(), flush=True)
    if not any(stage.wall_clock_buffers for stage in active):
        print(
            "No GstReferenceTimestampMeta on any buffer: the server is not sending RTCP "
            "sender reports yet, or add-reference-timestamp-meta had no effect.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
