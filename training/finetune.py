#!/usr/bin/env python3
"""Fine-tune an ultralytics detection model against a prepared dataset YAML."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from deepstream_yolo.paths import resolve_project_path  # noqa: E402

REQUIREMENTS = PROJECT_DIR / "requirements" / "training.txt"
DEFAULT_PROJECT = PROJECT_DIR / "outputs" / "training"
STRIDE = 32


def load_yolo():
    """Import ultralytics lazily so this module stays importable without it."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            f"ultralytics is not available in {sys.executable}.\n"
            "Training deps live in their own venv, not the DeepStream interpreter:\n"
            "  python3 -m venv .venv-train\n"
            f"  .venv-train/bin/python3 -m pip install -r {REQUIREMENTS}\n"
            f"  .venv-train/bin/python3 {Path(__file__).resolve()} --help"
        ) from exc

    return YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo11s.pt", help="starting checkpoint, or last.pt with --resume")
    parser.add_argument("--data", required=True, help="dataset YAML from prepare_dataset.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="square training size; see training/README.md")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu; default auto")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="finetune")
    parser.add_argument("--resume", action="store_true", help="resume --weights; other args come from the run")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data = resolve_project_path(args.data)
    if not data.is_file():
        raise SystemExit(f"Missing dataset YAML: {data}\nGenerate one with training/prepare_dataset.py")

    if args.imgsz % STRIDE:
        raise SystemExit(f"--imgsz must be a multiple of {STRIDE}, got {args.imgsz}")

    config_dir = PROJECT_DIR / "outputs" / "ultralytics-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    YOLO = load_yolo()

    project = resolve_project_path(args.project)
    project.mkdir(parents=True, exist_ok=True)


    model = YOLO(args.weights)

    if args.resume:
        print(f"Resuming {args.weights}; epochs/imgsz/batch come from the interrupted run.")
        model.train(resume=True)
    else:
        model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(project),
            name=args.name,
            resume=False,
            pretrained=args.pretrained,
        )

    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", project / args.name))
    best = save_dir / "weights" / "best.pt"

    print(f"\nRun:  {save_dir}")
    print(f"Best: {best}")
    print(
        f"\nmAP above was measured at {args.imgsz}x{args.imgsz}. DeepStream deploys a "
        "non-square static shape (e.g. 640x384), so these are not deployed-resolution "
        "numbers. See training/README.md."
    )
    print(f"\nExport:\n  python3 training/export_to_deepstream.py {best} --long-side {args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
