import os
import sys
import threading


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

    def stop(self) -> None:
        if self.saved_stderr_fd is None:
            return

        sys.stderr.flush()
        os.dup2(self.saved_stderr_fd, 2)
        if self.thread:
            self.thread.join(timeout=1.0)
        os.close(self.saved_stderr_fd)
        self.saved_stderr_fd = None

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
    return b"gst-plugin-scanner" in line and b"Failed to load plugin" in line


def maybe_start_gst_scan_warning_filter(argv: list[str]):
    if "--show-gst-scan-warnings" in argv:
        return None

    stderr_filter = StderrLineFilter(is_gst_plugin_scan_warning)
    stderr_filter.start()
    return stderr_filter


def stop_gst_scan_warning_filter(stderr_filter) -> None:
    if stderr_filter:
        stderr_filter.stop()
