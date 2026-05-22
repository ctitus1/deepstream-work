#!/usr/bin/env python3

import os
from typing import Dict

import gi
import rtsp_config as conf

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer

Gst.init(None)


def make_factory(launch: str) -> GstRtspServer.RTSPMediaFactory:
    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_shared(True)
    factory.set_launch(launch)

    # RTCP sender reports from gst-rtsp-server map RTP time to the server's
    # system clock. Since the host clock is NTP-synced and the container shares
    # that kernel clock, clients can recover network/NTP time from RTCP.
    factory.set_enable_rtcp(True)

    return factory


def cleanup_sockets() -> None:
    for path in getattr(conf, "SOCKETS", {}).values():
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def main() -> None:
    cleanup_sockets()

    producers: Dict[str, Gst.Element] = {}

    for name, pipe in getattr(conf, "PRODUCERS", {}).items():
        print(f"{name} producer starting...")
        producer = Gst.parse_launch(pipe)
        producer.set_state(Gst.State.PLAYING)
        producers[name] = producer
        print(f"{name} producer started!")

    server = GstRtspServer.RTSPServer()
    server.set_service("8554")
    mounts = server.get_mount_points()

    for name, pipe in conf.FACTORIES.items():
        print(f"{name} factory starting...")
        factory = make_factory(pipe)
        mounts.add_factory(f"/{name}", factory)
        print(f"rtsp://127.0.0.1:8554/{name}")
        print(f"{name} factory started!")

    server.attach(None)

    loop = GLib.MainLoop()

    try:
        loop.run()
    finally:
        for producer in producers.values():
            producer.set_state(Gst.State.NULL)
        cleanup_sockets()


if __name__ == "__main__":
    main()
