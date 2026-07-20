#!/usr/bin/env bash
# Create (or reuse) the YOLO export virtualenv.
#
# This is on the hot path: model_cache.ensure_model() shells out to
# yolo_export.sh on every cache miss, which lands here. It used to
# re-resolve and re-install the whole requirements file every time, so exporting
# a second .pt paid the full torch download again. The install is now guarded by
# a stamp over the requirements file, making a warm environment a no-op.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/common.sh

VENV_DIR="${VENV_DIR:-$(yolo_venv_dir)}"
REQUIREMENTS="requirements/yolo-export.txt"
PYTHON_BIN="$VENV_DIR/bin/python3"
FORCE="${FORCE:-0}"

[ "${1:-}" = "--force" ] && FORCE=1

# The venv is tied to the interpreter that built it, so a Python upgrade has to
# invalidate it too: the stamp covers both inputs.
SIGNATURE="$(hash_files "$REQUIREMENTS")-$(python_tag)"

if [ "$FORCE" -eq 0 ] && stamp_valid "yolo-export-env-$(python_tag)" "$SIGNATURE" "$VENV_DIR/bin/yolo"; then
    skip "YOLO export environment"
    exit 0
fi

if [ ! -x "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    step "Creating YOLO export environment ($VENV_DIR)"
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    step "Updating YOLO export environment ($VENV_DIR)"
fi

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
        if [ "$attempt" -ge "${PIP_ATTEMPTS:-4}" ]; then
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
pip_install_retry -r "$REQUIREMENTS"

[ -x "$VENV_DIR/bin/yolo" ] \
    || die "the YOLO CLI is missing after install: $VENV_DIR/bin/yolo"

# Stamped only after a verified-good install, so an interrupted run re-installs
# rather than being treated as complete.
stamp_write "yolo-export-env-$(python_tag)" "$SIGNATURE"
log "YOLO export environment ready."
