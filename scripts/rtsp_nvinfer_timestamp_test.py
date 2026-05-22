#!/usr/bin/env python3
import argparse
import datetime as dt
import sys
import time
from collections import deque
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst


PROJECT_DIR = Path("/home/user/deepstream-work")
NTP_TO_UNIX_SECONDS = 2_208_988_800
NS_PER_SEC = 1_000_000_000


def ns_to_utc(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.timezone.utc).isoformat(timespec="milliseconds")


def ntp_ns_to_unix_ns(ntp_ns: int) -> int:
    return ntp_ns - NTP_TO_UNIX_SECONDS * NS_PER_SEC


def seconds(ns: int) -> str:
    if ns == Gst.CLOCK_TIME_NONE or ns < 0:
        return "NONE"
    return f"{ns / 1e9:.3f}s"


def find_default_infer_config() -> Path:
    matches = sorted(
        (PROJECT_DIR / "configs").glob("config_infer_primary_yolo12x_640_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            "No yolo12x 640 nvinfer config found. Run your app once with "
            "--model yolo12x.pt --long-side 640 to generate it."
        )
    return matches[0]


class TimestampBridge:
    def __init__(self):
        self.ntp_queue = deque()
        self.ntp_caps = Gst.Caps.from_string("timestamp/x-ntp")

    def capture_probe(self, _pad, info, _data):
        buffer = info.get_buffer()
        ref_meta = buffer.get_reference_timestamp_meta(None) if buffer else None
        self.ntp_queue.append(ref_meta.timestamp if ref_meta else None)
        return Gst.PadProbeReturn.OK

    def infer_probe(self, _pad, info, _data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        ref_meta = buffer.get_reference_timestamp_meta(None)

        if ref_meta is None and self.ntp_queue:
            ntp_ns = self.ntp_queue.popleft()
            if ntp_ns is not None:
                buffer.add_reference_timestamp_meta(
                    self.ntp_caps,
                    ntp_ns,
                    Gst.CLOCK_TIME_NONE,
                )
                ref_meta = buffer.get_reference_timestamp_meta(None)

        recv_ns = time.time_ns()

        if ref_meta:
            stream_ns = ntp_ns_to_unix_ns(ref_meta.timestamp)
            latency_ms = (recv_ns - stream_ns) / 1e6
            print(
                f"local time={ns_to_utc(recv_ns)} "
                f"stream time={ns_to_utc(stream_ns)} "
                f"latency={latency_ms:.1f}ms "
                f"stream duration={seconds(buffer.pts)}",
                flush=True,
            )
        else:
            print(
                f"local time={ns_to_utc(recv_ns)} "
                f"stream time=NONE "
                f"latency=NONE "
                f"stream duration={seconds(buffer.pts)}",
                flush=True,
            )

        return Gst.PadProbeReturn.OK


def element(factory: str, name: str):
    elem = Gst.ElementFactory.make(factory, name)
    if elem is None:
        raise RuntimeError(f"Missing GStreamer element: {factory}")
    return elem


def on_rtsp_pad_added(_src, pad, depay):
    caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()
    if "application/x-rtp" in caps:
        pad.link(depay.get_static_pad("sink"))


def on_message(_bus, msg, loop):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"ERROR: {err}", file=sys.stderr)
        print(f"DEBUG: {dbg}", file=sys.stderr)
        loop.quit()
    elif msg.type == Gst.MessageType.EOS:
        loop.quit()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="rtsp://127.0.0.1:8554/rgb")
    parser.add_argument("--latency", type=int, default=100)
    parser.add_argument("--infer-config", default=None)
    args = parser.parse_args()

    Gst.init(None)

    infer_config = Path(args.infer_config) if args.infer_config else find_default_infer_config()
    print(f"infer config={infer_config}")

    bridge = TimestampBridge()
    pipeline = Gst.Pipeline.new("rtsp-nvinfer-timestamp-test")

    src = element("rtspsrc", "src")
    depay = element("rtph264depay", "depay")
    parse = element("h264parse", "parse")
    dec = element("nvv4l2decoder", "decoder")
    queue = element("queue", "queue")
    mux = element("nvstreammux", "mux")
    infer = element("nvinfer", "infer")
    sink = element("fakesink", "sink")

    src.set_property("location", args.uri)
    src.set_property("latency", args.latency)
    src.set_property("protocols", "tcp")
    src.set_property("ntp-sync", True)
    src.set_property("add-reference-timestamp-meta", True)

    mux.set_property("batch-size", 1)
    mux.set_property("width", 3840)
    mux.set_property("height", 2160)
    mux.set_property("live-source", 1)
    mux.set_property("batched-push-timeout", 40000)

    infer.set_property("config-file-path", str(infer_config))
    sink.set_property("sync", False)

    for e in (src, depay, parse, dec, queue, mux, infer, sink):
        pipeline.add(e)

    src.connect("pad-added", on_rtsp_pad_added, depay)
    depay.link(parse)
    parse.link(dec)
    dec.link(queue)
    queue.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
    mux.link(infer)
    infer.link(sink)

    parse.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, bridge.capture_probe, None)
    infer.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, bridge.infer_probe, None)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
