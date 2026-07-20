#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import signal
import sys
from pathlib import Path

import gi

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from deepstream_yolo.gst_warnings import (  # noqa: E402
    maybe_start_gst_scan_warning_filter,
    stop_gst_scan_warning_filter,
)
from deepstream_yolo.paths import (  # noqa: E402
    DEFAULT_RTSP_PORT,
    default_media,
    missing_media_message,
)

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstPbutils, GstRtspServer

# The parser/ROS clients reach the server across the container boundary, so it
# binds every interface rather than loopback.
LISTEN_ADDRESS = "0.0.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_RTSP_PORT)
    parser.add_argument(
        "--mount",
        default=None,
        help="Mount name; defaults to the video's basename so clients can derive it.",
    )
    parser.add_argument("--show-gst-scan-warnings", action="store_true")
    args = parser.parse_args()

    # Both defaults come from the media itself, so the server and the client
    # defaults in paths.py always agree on the mount without either being told.
    if args.video is None:
        args.video = str(default_media())
    if args.mount is None:
        args.mount = Path(args.video).stem
    return args


def resolve_video(path: str) -> Path:
    video = Path(path)
    if not video.is_absolute():
        video = PROJECT_DIR / video
    return video.resolve()


def discover_video(path: Path) -> tuple[str, int, int, str]:
    info = GstPbutils.Discoverer.new(5 * Gst.SECOND).discover_uri(path.as_uri())
    streams = info.get_video_streams()
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")

    stream = streams[0]
    caps = stream.get_caps()
    caps_text = caps.to_string() if caps else ""
    codec_text = caps_text.lower()

    if "h.265" in codec_text or "h265" in codec_text or "hevc" in codec_text:
        family = "h265"
    elif "h.264" in codec_text or "h264" in codec_text or "avc" in codec_text:
        family = "h264"
    else:
        family = "encode-h264"

    return family, int(stream.get_width()), int(stream.get_height()), caps_text


def launch_pipeline(path: Path, codec_family: str) -> str:
    location = shlex.quote(str(path))

    if codec_family == "h265":
        return (
            f"( filesrc location={location} ! qtdemux name=demux "
            "demux.video_0 ! queue ! h265parse ! identity sync=true ! "
            "rtph265pay name=pay0 pt=96 config-interval=1 )"
        )

    if codec_family == "h264":
        return (
            f"( filesrc location={location} ! qtdemux name=demux "
            "demux.video_0 ! queue ! h264parse ! identity sync=true ! "
            "rtph264pay name=pay0 pt=96 config-interval=1 )"
        )

    return (
        f"( filesrc location={location} ! decodebin ! queue ! videoconvert ! "
        "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bitrate=8000 ! "
        "h264parse ! identity sync=true ! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )


class LoopingRtspServer:
    def __init__(self, args: argparse.Namespace, launch: str):
        self.args = args
        self.launch = launch
        self.loop = GLib.MainLoop()
        self.bus_refs = []
        self.server = None
        self.factory = None
        self.mounts = None

    def start(self) -> str:
        mount = self.args.mount.strip("/") or "stream"

        self.server = GstRtspServer.RTSPServer()
        self.server.set_address(LISTEN_ADDRESS)
        self.server.set_service(str(self.args.port))

        self.factory = GstRtspServer.RTSPMediaFactory()
        self.factory.set_launch(self.launch)
        self.factory.set_shared(True)
        if hasattr(self.factory, "set_eos_shutdown"):
            self.factory.set_eos_shutdown(False)
        self.factory.connect("media-configure", self.on_media_configure)

        self.mounts = self.server.get_mount_points()
        self.mounts.add_factory(f"/{mount}", self.factory)
        attach_id = self.server.attach(None)
        if attach_id == 0:
            raise RuntimeError(f"Failed to attach RTSP server on {LISTEN_ADDRESS}:{self.args.port}")

        return f"rtsp://127.0.0.1:{self.args.port}/{mount}"

    def on_media_configure(self, _factory, media) -> None:
        element = media.get_element()
        bus = element.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self.on_eos, element)
        # Written and never read: holding the bus keeps its message::eos watch
        # alive past this scope, which is what makes looping work.
        self.bus_refs.append(bus)

    def on_eos(self, _bus, _message, element) -> None:
        element.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
            0,
        )

    def run(self) -> None:
        def stop(_signum, _frame):
            self.loop.quit()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        self.loop.run()


def main() -> int:
    args = parse_args()
    video = resolve_video(args.video)
    if not video.is_file():
        print(missing_media_message(video), file=sys.stderr)
        return 1

    warning_filter = maybe_start_gst_scan_warning_filter(sys.argv)
    try:
        Gst.init(None)
        codec_family, width, height, codec = discover_video(video)
    finally:
        stop_gst_scan_warning_filter(warning_filter)

    launch = launch_pipeline(video, codec_family)

    server = LoopingRtspServer(args, launch)
    url = server.start()
    print(f"Serving {video}", flush=True)
    print(f"Video: {width}x{height} {codec}", flush=True)
    print(f"URL: {url}", flush=True)
    print(
        "Network timestamps are generated from the RTSP server clock via RTCP sender reports.",
        flush=True,
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
