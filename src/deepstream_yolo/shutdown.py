"""Graceful shutdown for the GLib-driven apps.

``GLib.MainLoop.run()`` blocks inside C, so a plain Python ``signal`` handler
does not run until the loop returns -- which, for a signal meant to *stop* the
loop, is never. PyGObject special-cases SIGINT, so Ctrl-C works, but nothing
handles SIGTERM or SIGHUP: ``docker stop``, ``kill``, and closing the terminal
all kill the process outright.

That matters beyond tidiness. The recording branch only writes its mp4 moov
atom during the normal teardown path, so a process killed mid-run leaves an
unplayable file. ``GLib.unix_signal_add`` runs the handler from inside the loop
itself, which lets every stop signal take the same clean exit Ctrl-C does.
"""

from __future__ import annotations

import os
import signal
import sys

from gi.repository import GLib

# SIGHUP covers the closed-terminal case; the rest are the usual stop signals.
# GLib.unix_signal_add supports only this set plus SIGUSR1/2 and SIGWINCH.
_SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

_SIGNAL_NAMES = {
    signal.SIGINT: "SIGINT",
    signal.SIGTERM: "SIGTERM",
    signal.SIGHUP: "SIGHUP",
}


def install_shutdown_handlers(loop: GLib.MainLoop) -> None:
    """Quit ``loop`` on any stop signal; a second signal exits immediately.

    The escalation path matters when teardown itself is what is stuck: the
    first signal starts a clean shutdown, and a user who is not willing to wait
    for it can press Ctrl-C again instead of reaching for ``kill -9``, which
    would skip the flush entirely.
    """
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
