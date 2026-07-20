"""Process-level plumbing around the GLib main loop.

Both halves exist because a GLib app does not behave like a normal Python
process, and both are about the same thing: making sure the output and the exit
path are trustworthy.

  * **Signals** -- ``GLib.MainLoop.run()`` blocks inside C, so a plain Python
    ``signal`` handler does not run until the loop returns, which for a signal
    meant to *stop* the loop is never. PyGObject special-cases SIGINT, so Ctrl-C
    works, but nothing handles SIGTERM or SIGHUP: ``docker stop``, ``kill``, and
    closing the terminal all kill the process outright. That matters beyond
    tidiness -- the recording branch writes its mp4 moov atom only during normal
    teardown, so a process killed mid-run leaves an unplayable file.
    ``GLib.unix_signal_add`` runs the handler from inside the loop, giving every
    stop signal the same clean exit Ctrl-C gets.
  * **Stderr** -- GStreamer prints known-benign startup noise that looks exactly
    like a real failure, in the same screenful where real failures appear.
    ``StderrLineFilter`` drops only specific known lines and lets everything
    else through.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading

# GLib is imported inside install_shutdown_handlers rather than here on purpose.
# The entrypoints import the stderr filter below *before* `import gi`, so that
# the filter is already in place when GStreamer scans its plugins and prints the
# noise it exists to suppress. Importing GLib at module scope would drag
# PyGObject into that early window for the benefit of the other half of this
# file, which no caller needs until it already has a main loop.


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

# SIGHUP covers the closed-terminal case; the rest are the usual stop signals.
# GLib.unix_signal_add supports only this set plus SIGUSR1/2 and SIGWINCH.
_SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

_SIGNAL_NAMES = {
    signal.SIGINT: "SIGINT",
    signal.SIGTERM: "SIGTERM",
    signal.SIGHUP: "SIGHUP",
}


def install_shutdown_handlers(loop) -> None:
    """Quit ``loop`` on any stop signal; a second signal exits immediately.

    The escalation path matters when teardown itself is what is stuck: the
    first signal starts a clean shutdown, and a user who is not willing to wait
    for it can press Ctrl-C again instead of reaching for ``kill -9``, which
    would skip the flush entirely.
    """
    from gi.repository import GLib

    state = {"stopping": False}

    def handle(signum: int) -> bool:
        name = _SIGNAL_NAMES.get(signum, str(signum))

        if state["stopping"]:
            print(f"\n{name} again: exiting now.", file=sys.stderr, flush=True)
            # os._exit skips the teardown we already know is not finishing.
            os._exit(130)

        state["stopping"] = True
        print(f"\n{name}: shutting down cleanly...", file=sys.stderr, flush=True)
        loop.quit()
        # Stay installed so the second press can escalate.
        return GLib.SOURCE_CONTINUE

    for signum in _SHUTDOWN_SIGNALS:
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signum, handle, signum)


# ---------------------------------------------------------------------------
# Stderr filtering
# ---------------------------------------------------------------------------


class StderrLineFilter:
    def __init__(self, suppress):
        self.suppress = suppress
        self.read_fd = None
        self.saved_stderr_fd = None
        self.thread = None
        self.skip_blank = False

    def start(self) -> None:
        read_fd, write_fd = os.pipe()
        self.read_fd = read_fd
        self.saved_stderr_fd = os.dup(2)
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        # While the filter is running, fd 2 is a pipe drained by a daemon
        # thread. If the process exits without stop() -- an uncaught exception
        # before the usual teardown, for instance -- fd 2 is never restored and
        # the daemon thread dies at interpreter shutdown with the traceback
        # still sitting in the pipe, so the error vanishes. atexit runs while
        # daemon threads can still be joined, so it guarantees a drain.
        atexit.register(self.stop)

    def stop(self) -> None:
        if self.saved_stderr_fd is None:
            return

        sys.stderr.flush()
        # Restoring fd 2 drops the last reference to the pipe's write end, so
        # the pump sees EOF, flushes what is buffered and exits.
        os.dup2(self.saved_stderr_fd, 2)
        if self.thread:
            self.thread.join(timeout=1.0)
        os.close(self.saved_stderr_fd)
        self.saved_stderr_fd = None
        atexit.unregister(self.stop)

    def _emit(self, line: bytes) -> None:
        if self.suppress(line):
            self.skip_blank = True
            return

        if self.skip_blank and not line.strip():
            return

        self.skip_blank = False
        os.write(self.saved_stderr_fd, line)

    def _pump(self) -> None:
        pending = b""
        try:
            while True:
                chunk = os.read(self.read_fd, 4096)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._emit(line + b"\n")
            if pending:
                self._emit(pending)
        finally:
            os.close(self.read_fd)


def is_gst_plugin_scan_warning(line: bytes) -> bool:
    """Startup noise that is expected in this image and means nothing is wrong.

    Both of these are printed before a single frame moves and look like errors
    to anyone reading the first screenful of output, which is exactly where
    real failures also appear. Suppressing them by default keeps that screenful
    meaningful; --show-gst-scan-warnings brings them back.

    Deliberately narrow: each pattern is a specific known-benign message, not a
    category. Anything unrecognized still reaches the terminal.
    """
    # One per plugin whose optional codec libraries this image does not ship
    # (libFLAC, libmpg123, libtritonserver, ...). None are used by this pipeline.
    if b"gst-plugin-scanner" in line and b"Failed to load plugin" in line:
        return True

    # nvv4l2decoder asking the NVIDIA decoder device for *capture* capabilities
    # it does not advertise. Emitted once per decoder created, including the one
    # gst-discoverer builds, which is why it usually appears twice. Decoding is
    # unaffected -- the 4K H.265 stream decodes normally right after it.
    if b"Failed to query video capabilities" in line:
        return True

    return False


def maybe_start_gst_scan_warning_filter(argv: list[str]):
    if "--show-gst-scan-warnings" in argv:
        return None

    stderr_filter = StderrLineFilter(is_gst_plugin_scan_warning)
    stderr_filter.start()
    return stderr_filter


def stop_gst_scan_warning_filter(stderr_filter) -> None:
    if stderr_filter:
        stderr_filter.stop()
