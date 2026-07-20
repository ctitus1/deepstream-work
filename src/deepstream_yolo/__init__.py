"""Helpers for the DeepStream YOLO parser app."""

import os

# GIO consults a proxy resolver whenever it opens a network connection. The
# libproxy build in the DeepStream base image faults while unwinding when asked,
# killing the process with SIGSEGV before Python can report anything -- so an
# RTSP run exited instantly and silently while local-file input worked fine.
#
# The container sets this too (docker/Dockerfile, docker-compose.yml); repeating
# it here means the crash cannot come back through a path that misses the
# container environment, such as running an app on the host or in an older
# image. setdefault rather than assignment: an explicit setting still wins if a
# deployment genuinely needs proxy resolution.
os.environ.setdefault("GIO_USE_PROXY_RESOLVER", "dummy")
