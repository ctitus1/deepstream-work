#!/usr/bin/env python3
"""Export the detector and assessment models for this checkout.

Setup-time front end to ``deepstream_yolo.model_cache``, which is the same code
path the apps use at startup. Running it here means a first run of the parser or
the ROS stack finds everything already exported instead of stalling on it.

Both exports are content-addressed by the cache, so re-running is a no-op and
pointing ``--model`` at a new checkpoint only rebuilds that one model:

    python3 scripts/prepare_models.py
    python3 scripts/prepare_models.py --model runs/detect/train/weights/best.pt

Unlike the apps, this defaults to the local media file rather than the RTSP URL:
setup runs before any RTSP server is up, and only the source geometry is needed
to pick the inference resolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from deepstream_yolo.gst_warnings import (  # noqa: E402
    maybe_start_gst_scan_warning_filter,
    stop_gst_scan_warning_filter,
)

# Started before gi is imported, which is when the plugin scanner runs and emits
# a screenful of "Failed to load plugin" lines for codecs this image does not
# ship. Setup output is worth keeping readable; --show-gst-scan-warnings opts in.
GST_SCAN_WARNING_FILTER = maybe_start_gst_scan_warning_filter(sys.argv)

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from deepstream_yolo.model_cache import (  # noqa: E402
    discover_size,
    ensure_assessment_model,
    ensure_model,
)
from deepstream_yolo.paths import (  # noqa: E402
    DEFAULT_ASSESSMENT_MODEL,
    DEFAULT_MEDIA,
    DEFAULT_MODEL,
)
from deepstream_yolo.stream_source import resolve_stream_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--stream", default=str(DEFAULT_MEDIA))
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--assessment-model", default=DEFAULT_ASSESSMENT_MODEL)
    parser.add_argument("--assessment-batch-size", type=int, default=8)
    parser.add_argument(
        "--no-assessment",
        dest="assessment",
        action="store_false",
        help="Export only the detector.",
    )
    parser.add_argument("--show-gst-scan-warnings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve inside the filter's scope, then restore stderr: a missing stream
    # or an undiscoverable source must reach the terminal, and the pump drops
    # whatever is still in flight when the process exits.
    try:
        Gst.init(None)
        stream = resolve_stream_source(args.stream)
        src_w, src_h = discover_size(stream.uri)
    finally:
        stop_gst_scan_warning_filter(GST_SCAN_WARNING_FILTER)

    print(f"source: {stream.display} {src_w}x{src_h}", flush=True)

    model_w, model_h, config = ensure_model(
        args.model, stream, args.long_side, src_w, src_h, args.conf
    )
    print(f"detector: {args.model} {model_w}x{model_h}", flush=True)
    print(f"  config: {config}", flush=True)

    if args.assessment:
        meta, assess_config = ensure_assessment_model(
            args.assessment_model, args.assessment_batch_size
        )
        print(
            f"assessment: {meta.get('architecture', args.assessment_model)} "
            f"batch={args.assessment_batch_size}",
            flush=True,
        )
        print(f"  config: {assess_config}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
