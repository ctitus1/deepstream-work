#!/usr/bin/env python3
"""Hand a fine-tuned .pt to scripts/setup_and_export_yolo.sh, after checking the
consistency traps that silently break custom-class deployments.

Checks performed:
  1. The exporter writes correct per-class labels.txt, then install_labels()
     overwrites models/coco_labels.txt with the 80-class COCO list and the next
     line runs "rm -f labels.txt". A custom N-class model therefore deploys with
     COCO names unless labels/coco_labels.txt already holds the trained names.
  2. num-detected-classes is hardcoded to 80 in both the generated nvinfer
     config and src/deepstream_yolo/configs.py.
  3. Training imgsz is square; the deployed ONNX shape is static and non-square.
  4. The model stem must match a family the setup script dispatches on.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from deepstream_yolo.paths import (  # noqa: E402
    DEFAULT_MEDIA,
    GENERATED_CONFIG_DIR,
    LABELS_PATH,
    LABELS_SOURCE_PATH,
    MODELS_DIR,
    SETUP_SCRIPT,
    YOLO_PYTHON,
    resolve_project_path,
)

# Families dispatched by the case statement in scripts/setup_and_export_yolo.sh.
SUPPORTED_PREFIXES = ("yolo11", "yolo12", "yolov12", "yolo8", "yolov8")

PROBE = """
import json, sys
from ultralytics import YOLO

model = YOLO(sys.argv[1])
names = model.names
payload = {"names": [str(names[key]) for key in sorted(names)]}

try:
    train_args = (model.ckpt or {}).get("train_args") or {}
    payload["train_imgsz"] = train_args.get("imgsz")
    payload["base_model"] = train_args.get("model")
except Exception:
    payload["train_imgsz"] = None
    payload["base_model"] = None

print(json.dumps(payload))
"""


def probe_weights(weights: Path) -> dict:
    """Read class names and train args from a checkpoint using the export venv."""
    if not YOLO_PYTHON.exists():
        raise SystemExit(
            f"Missing YOLO export environment: {YOLO_PYTHON}\nCreate it with: scripts/setup_yolo_export_env.sh"
        )

    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / "outputs" / "ultralytics-config"))
    Path(env["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    try:
        out = subprocess.check_output(
            [str(YOLO_PYTHON), "-c", PROBE, str(weights)], cwd=PROJECT_DIR, env=env, text=True
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Could not read {weights} with {YOLO_PYTHON} (exit {exc.returncode}).\n"
            "If ultralytics is missing there, run: scripts/setup_yolo_export_env.sh"
        ) from exc

    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"Empty probe output for {weights}")

    return json.loads(lines[-1])


def read_labels(path: Path) -> list[str]:
    """Read a DeepStream label file as a list of names."""
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def deployed_size(long_side: int, src_w: int, src_h: int, stride: int = 32) -> tuple[int, int]:
    """Mirror the stride-safe sizing done by scripts/setup_and_export_yolo.sh."""

    def round_stride(value: float) -> int:
        return max(stride, ((int(value) + stride - 1) // stride) * stride)

    if src_w >= src_h:
        return round_stride(long_side), round_stride(long_side * src_h / src_w)
    return round_stride(long_side * src_w / src_h), round_stride(long_side)


def check_family(stem: str, base_model: str | None) -> list[tuple[str, str]]:
    """The setup script exits 1 on stems it cannot dispatch."""
    if stem.startswith(SUPPORTED_PREFIXES):
        return [("ok", f"Model stem '{stem}' dispatches to a DeepStream-Yolo exporter.")]

    suggested = f"{Path(base_model).stem}-custom.pt" if base_model else "yolo11s-custom.pt"
    return [
        (
            "error",
            f"Model stem '{stem}' matches no exporter in {SETUP_SCRIPT.name}; it accepts "
            f"{', '.join(SUPPORTED_PREFIXES)}.\n"
            f"    Stage it under a dispatchable name: --stage-as {suggested}",
        )
    ]


def check_labels(names: list[str], labels: list[str]) -> list[tuple[str, str]]:
    """Class-count and name agreement against the labels the export will install."""
    if not labels:
        return [("error", f"Missing {LABELS_SOURCE_PATH}; the setup script exits 1 without it.")]

    if len(labels) != len(names):
        return [
            (
                "error",
                f"Class-count mismatch: model has {len(names)}, {LABELS_SOURCE_PATH} has {len(labels)}.\n"
                f"    install_labels() copies that file over models/coco_labels.txt and the next line runs\n"
                "    'rm -f labels.txt', deleting the correct labels the exporter just generated, so this\n"
                f"    model would deploy with {len(labels)} wrong names.\n"
                "    Fix with --install-labels (writes the trained names to labels/coco_labels.txt).",
            )
        ]

    if labels != names:
        mismatch = next(f"{a} != {b}" for a, b in zip(labels, names) if a != b)
        return [
            (
                "warn",
                f"Counts agree but names differ (first mismatch: {mismatch}).\n"
                "    Overlay text will be wrong. Fix with --install-labels.",
            )
        ]

    return [("ok", f"labels/coco_labels.txt matches the model's {len(names)} classes.")]


def check_imgsz(train_imgsz, long_side: int, src_w: int, src_h: int) -> list[tuple[str, str]]:
    """Training is square; deployment bakes a static non-square shape."""
    width, height = deployed_size(long_side, src_w, src_h)
    trained = f"{int(train_imgsz)}x{int(train_imgsz)}" if train_imgsz else "unknown"
    if width == height:
        return [("ok", f"Deployed input {width}x{height} is square; trained at {trained}.")]

    return [
        (
            "warn",
            f"Trained at {trained}, deploying static {width}x{height}. ultralytics train/val take a "
            "single square int,\n    so offline mAP is not measured at the deployed resolution. "
            "Validate on real frames after export.",
        )
    ]


def check_generated_config(stem: str, class_count: int) -> list[tuple[str, str]]:
    """num-detected-classes must equal the trained class count."""
    config = GENERATED_CONFIG_DIR / f"config_infer_primary_{stem}.txt"
    if not config.is_file():
        return [("warn", f"No generated config at {config}.")]

    for line in config.read_text().splitlines():
        if line.startswith("num-detected-classes="):
            found = int(line.split("=", 1)[1])
            if found == class_count:
                return [("ok", f"num-detected-classes={found} in {config.name}.")]
            return [
                (
                    "error",
                    f"num-detected-classes={found} in {config} but the model has {class_count}.\n"
                    "    It is hardcoded to 80 in scripts/setup_and_export_yolo.sh AND in\n"
                    "    src/deepstream_yolo/configs.py:write_infer_config, which rewrites this file on\n"
                    "    every parser_app run. Rerun with --fix-generated-config for deepstream-app use,\n"
                    "    but the parser_app path needs configs.py changed by its owner.",
                )
            ]

    return [("warn", f"No num-detected-classes entry in {config}.")]


def emit(results: list[tuple[str, str]]) -> int:
    """Print check results and return the error count."""
    marks = {"ok": "OK  ", "warn": "WARN", "error": "FAIL"}
    for level, message in results:
        print(f"[{marks[level]}] {message}")
    return sum(1 for level, _ in results if level == "error")


def install_labels(names: list[str]) -> None:
    """Write the trained names where install_labels() will pick them up."""
    if LABELS_SOURCE_PATH.is_file():
        backup = LABELS_SOURCE_PATH.with_suffix(".txt.bak")
        shutil.copy2(LABELS_SOURCE_PATH, backup)
        print(f"Backed up {LABELS_SOURCE_PATH} to {backup}")

    LABELS_SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_SOURCE_PATH.write_text("\n".join(names) + "\n")
    print(f"Wrote {len(names)} classes to {LABELS_SOURCE_PATH}")


def fix_generated_config(stem: str, class_count: int) -> None:
    """Patch num-detected-classes in the generated nvinfer config."""
    config = GENERATED_CONFIG_DIR / f"config_infer_primary_{stem}.txt"
    if not config.is_file():
        return

    lines = [
        f"num-detected-classes={class_count}" if line.startswith("num-detected-classes=") else line
        for line in config.read_text().splitlines()
    ]
    config.write_text("\n".join(lines) + "\n")
    print(f"Patched num-detected-classes={class_count} in {config}")
    print("Note: src/deepstream_yolo/configs.py rewrites this file with 80 on the next parser_app run.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", help="trained .pt, e.g. outputs/training/finetune/weights/best.pt")
    parser.add_argument("--long-side", type=int, default=640, help="passed to the setup script")
    parser.add_argument("--stream", default=None, help="stream used to derive the deployed aspect ratio")
    parser.add_argument("--source-width", type=int, default=1920, help="assumed source width for checks")
    parser.add_argument("--source-height", type=int, default=1080, help="assumed source height for checks")
    parser.add_argument("--stage-as", default=None, help="copy weights into models/ under this filename")
    parser.add_argument("--labels-out", default=None, help="where to write the trained label list")
    parser.add_argument("--install-labels", action="store_true", help="also write labels/coco_labels.txt")
    parser.add_argument("--fix-generated-config", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="run checks without exporting")
    parser.add_argument("--force", action="store_true", help="export even if checks fail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    weights = resolve_project_path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")
    if weights.suffix != ".pt":
        raise SystemExit(f"Expected a .pt checkpoint, got {weights.suffix or 'no suffix'}: {weights}")

    stream = args.stream or str(DEFAULT_MEDIA.relative_to(PROJECT_DIR))

    info = probe_weights(weights)
    names = info["names"]
    print(f"Model:   {weights}")
    print(f"Classes: {len(names)} -> {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}\n")

    labels_out = resolve_project_path(args.labels_out or f"outputs/training/{weights.stem}.labels.txt")
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.write_text("\n".join(names) + "\n")
    print(f"Wrote trained label list: {labels_out}\n")

    if args.install_labels:
        install_labels(names)
        print()

    staged = weights
    if args.stage_as:
        staged = MODELS_DIR / Path(args.stage_as).name
        staged.parent.mkdir(parents=True, exist_ok=True)
        if staged.resolve() != weights.resolve():
            shutil.copy2(weights, staged)
        print(f"Staged weights as {staged}\n")

    results: list[tuple[str, str]] = []
    results += check_family(staged.stem, info.get("base_model"))
    results += check_labels(names, read_labels(LABELS_SOURCE_PATH))
    results += check_imgsz(info.get("train_imgsz"), args.long_side, args.source_width, args.source_height)

    failures = emit(results)

    if args.check_only:
        return 1 if failures else 0

    if failures and not args.force:
        print(f"\n{failures} blocking issue(s); not exporting. Use --force to override.")
        return 1

    env = os.environ.copy()
    env["SOURCE_WIDTH"] = str(args.source_width)
    env["SOURCE_HEIGHT"] = str(args.source_height)

    print(f"\nRunning {SETUP_SCRIPT.name} {staged.name} {args.long_side} {stream}\n")
    subprocess.run(
        [str(SETUP_SCRIPT), str(staged), str(args.long_side), stream],
        cwd=PROJECT_DIR,
        env=env,
        check=True,
    )

    if args.fix_generated_config:
        fix_generated_config(staged.stem, len(names))

    print("\nPost-export checks:")
    post = check_generated_config(staged.stem, len(names))
    deployed = read_labels(LABELS_PATH)
    if len(deployed) != len(names):
        post.append(
            (
                "error",
                f"{LABELS_PATH} has {len(deployed)} names, model has {len(names)}: "
                "install_labels() overwrote them. Rerun with --install-labels.",
            )
        )
    else:
        post.append(("ok", f"{LABELS_PATH} has {len(deployed)} names."))

    return 1 if emit(post) else 0


if __name__ == "__main__":
    raise SystemExit(main())
