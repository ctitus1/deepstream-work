#!/usr/bin/env python3
import argparse
import datetime as dt
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst


NTP_TO_UNIX_SECONDS = 2_208_988_800


def unix_ns_to_utc(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.timezone.utc).isoformat(timespec="milliseconds")


def ntp_ns_to_unix_ns(ntp_ns: int) -> int:
    return ntp_ns - NTP_TO_UNIX_SECONDS * 1_000_000_000


def seconds(ns: int) -> str:
    if ns == Gst.CLOCK_TIME_NONE or ns < 0:
        return "NONE"
    return f"{ns / 1e9:.3f}s"


def on_handoff(_sink, buffer, _pad):
    recv_ns = time.time_ns()

    try:
        ref_meta = buffer.get_reference_timestamp_meta(None)
    except Exception:
        ref_meta = None

    if ref_meta:
        stream_unix_ns = ntp_ns_to_unix_ns(ref_meta.timestamp)
        latency_ms = (recv_ns - stream_unix_ns) / 1e6

        print(
            f"local time={unix_ns_to_utc(recv_ns)} "
            f"stream time={unix_ns_to_utc(stream_unix_ns)} "
            f"latency={latency_ms:.1f}ms "
            f"stream duration={seconds(buffer.pts)}",
            flush=True,
        )
    else:
        print(
            f"recv={unix_ns_to_utc(recv_ns)} "
            f"stream=NONE "
            f"latency=NONE "
            f"pts={seconds(buffer.pts)}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="rtsp://127.0.0.1:8554/rgb")
    ap.add_argument("--latency", type=int, default=100)
    ap.add_argument("--protocols", default="tcp")
    args = ap.parse_args()

    Gst.init(None)

    pipeline = Gst.parse_launch(
        "rtspsrc "
        f"location={args.uri} "
        f"latency={args.latency} "
        f"protocols={args.protocols} "
        # "ntp-sync=true "
        "add-reference-timestamp-meta=true "
        "! rtph264depay "
        "! h264parse "
        "! fakesink name=sink sync=false signal-handoffs=true"
    )

    sink = pipeline.get_by_name("sink")
    sink.connect("handoff", on_handoff)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_msg(_bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"ERROR: {err}")
            print(f"DEBUG: {dbg}")
            loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            loop.quit()

    bus.connect("message", on_msg)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
