#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR=".venv-yolo"

if [ ! -x "$VENV_DIR/bin/python3" ] || ! "$VENV_DIR/bin/python3" -m pip --version >/dev/null 2>&1; then
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python3"

PIP_ATTEMPTS="${PIP_ATTEMPTS:-4}"

# This requirement set pulls torch, which drags in several nvidia-*-cu12 wheels
# in the 150-500 MB range. Those large downloads are intermittently corrupted on
# some networks, and pip reports it as:
#
#   ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE
#      unknown package: Expected sha256 <a> / Got <b>
#
# despite this repo pinning no hashes at all - the hash comes from the package
# index, and pip is telling you the bytes it received do not match what the
# index advertised. It is a truncated or mangled transfer, not a tampered
# package: the "Got" value differs on every attempt, whereas a substituted
# artifact would hash the same way each time.
#
# pip's own --retries only covers connection errors, not a post-download hash
# mismatch, so retry the whole install. Small wheels are unaffected, so a retry
# only re-fetches what is missing.
pip_install_retry() {
    local attempt=1
    while true; do
        if "$PYTHON_BIN" -m pip install --timeout 60 --retries 5 "$@"; then
            return 0
        fi
        if [ "$attempt" -ge "$PIP_ATTEMPTS" ]; then
            echo "pip install failed after ${attempt} attempts: $*" >&2
            echo "If it always fails on the same large wheel, suspect a proxy or MITM" >&2
            echo "rewriting the download rather than a bad package." >&2
            return 1
        fi
        echo "pip install attempt ${attempt} failed; retrying..." >&2
        attempt=$((attempt + 1))
    done
}

pip_install_retry --upgrade 'pip' 'setuptools<82' wheel

pip_install_retry -r requirements/yolo-export.txt
