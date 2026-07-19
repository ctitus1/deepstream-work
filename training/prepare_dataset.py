#!/usr/bin/env python3
"""Convert a detection dataset into the ultralytics layout and emit its YAML.

Output layout:
    <out>/images/{train,val}/...
    <out>/labels/{train,val}/...   YOLO-normalized "cls cx cy w h"
    <out>/<out-name>.yaml
    <out>/<out-name>.labels.txt    class names, one per line

To add a format, write a loader with the signature
``load_x(annotations: Path, images_dir: Path) -> Dataset`` and register it in
LOADERS.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from deepstream_yolo.paths import resolve_project_path  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MISSING_REPORT_LIMIT = 10


@dataclass
class Box:
    """One normalized detection box."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Sample:
    """One image plus its boxes."""

    image: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)


@dataclass
class Dataset:
    """A loaded dataset in a format-independent form."""

    names: list[str]
    samples: list[Sample]


def image_size(path: Path) -> tuple[int, int]:
    """Read image dimensions, used only when annotations omit them."""
    from PIL import Image

    with Image.open(path) as img:
        return int(img.width), int(img.height)


def normalize_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Clamp an absolute xywh box and convert it to normalized cxcywh."""
    x, y, w, h = (float(v) for v in bbox[:4])
    x0 = min(max(x, 0.0), float(width))
    y0 = min(max(y, 0.0), float(height))
    x1 = min(max(x + w, 0.0), float(width))
    y1 = min(max(y + h, 0.0), float(height))

    bw = x1 - x0
    bh = y1 - y0
    if bw <= 1e-6 or bh <= 1e-6:
        return None

    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, bw / width, bh / height)


def load_coco(annotations: Path, images_dir: Path) -> Dataset:
    """Load a COCO-JSON detection dataset."""
    payload = json.loads(annotations.read_text())

    categories = sorted(payload.get("categories", []), key=lambda c: int(c["id"]))
    if not categories:
        raise ValueError(f"No categories in {annotations}")

    class_index = {int(c["id"]): i for i, c in enumerate(categories)}
    names = [str(c["name"]) for c in categories]

    samples: dict[int, Sample] = {}
    missing: list[Path] = []
    for image in payload.get("images", []):
        path = images_dir / str(image["file_name"])
        if not path.is_file():
            missing.append(path)
            continue

        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = image_size(path)

        samples[int(image["id"])] = Sample(image=path, width=width, height=height)

    if missing:
        head = "\n".join(f"  {p}" for p in missing[:MISSING_REPORT_LIMIT])
        extra = "" if len(missing) <= MISSING_REPORT_LIMIT else f"\n  ... and {len(missing) - MISSING_REPORT_LIMIT} more"
        raise FileNotFoundError(
            f"{len(missing)} image(s) referenced by {annotations} are missing under {images_dir}:\n{head}{extra}"
        )

    if not samples:
        raise ValueError(f"No usable images in {annotations}")

    skipped_crowd = 0
    skipped_degenerate = 0
    for ann in payload.get("annotations", []):
        if int(ann.get("iscrowd", 0)):
            skipped_crowd += 1
            continue

        sample = samples.get(int(ann["image_id"]))
        if sample is None:
            continue

        category_id = int(ann["category_id"])
        if category_id not in class_index:
            raise ValueError(f"Annotation references unknown category_id={category_id} in {annotations}")

        box = normalize_bbox(list(ann["bbox"]), sample.width, sample.height)
        if box is None:
            skipped_degenerate += 1
            continue

        sample.boxes.append(Box(class_index[category_id], *box))

    if skipped_crowd or skipped_degenerate:
        print(f"Skipped annotations: iscrowd={skipped_crowd} degenerate={skipped_degenerate}")

    return Dataset(names=names, samples=list(samples.values()))


LOADERS = {"coco": load_coco}


def split_samples(samples: list[Sample], val_split: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Deterministically split samples into train/val."""
    ordered = sorted(samples, key=lambda s: str(s.image))
    if val_split <= 0.0:
        return ordered, []

    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_split)))
    val_count = min(val_count, len(shuffled) - 1)
    return sorted(shuffled[val_count:], key=lambda s: str(s.image)), sorted(
        shuffled[:val_count], key=lambda s: str(s.image)
    )


def destination_names(samples: list[Sample], rename_collisions: bool) -> dict[Path, str]:
    """Map each source image to a unique basename inside its split."""
    used: dict[str, Path] = {}
    mapping: dict[Path, str] = {}
    collisions: list[tuple[Path, Path]] = []

    for sample in samples:
        stem = sample.image.stem
        suffix = sample.image.suffix
        name = f"{stem}{suffix}"
        if name in used:
            if not rename_collisions:
                collisions.append((used[name], sample.image))
                continue
            counter = 1
            while f"{stem}_{counter}{suffix}" in used:
                counter += 1
            name = f"{stem}_{counter}{suffix}"

        used[name] = sample.image
        mapping[sample.image] = name

    if collisions:
        head = "\n".join(f"  {a}\n  {b}" for a, b in collisions[:MISSING_REPORT_LIMIT])
        raise ValueError(
            f"{len(collisions)} image basename collision(s) within one split:\n{head}\n"
            "Rerun with --rename-collisions to disambiguate."
        )

    return mapping


def link_image(src: Path, dst: Path, mode: str) -> None:
    """Place a source image into the dataset tree."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        dst.symlink_to(src.resolve())


def write_split(out: Path, split: str, samples: list[Sample], mode: str, rename_collisions: bool) -> None:
    """Write images and label files for one split."""
    images_dir = out / "images" / split
    labels_dir = out / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    mapping = destination_names(samples, rename_collisions)
    for sample in samples:
        name = mapping[sample.image]
        link_image(sample.image, images_dir / name, mode)
        lines = [f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in sample.boxes]
        (labels_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def write_yaml(path: Path, out: Path, names: list[str], has_val: bool) -> None:
    """Emit the ultralytics dataset YAML."""
    entries = "\n".join(f"  {i}: {json.dumps(name)}" for i, name in enumerate(names))
    val = "images/val" if has_val else "images/train"
    path.write_text(
        f"""# Generated by training/prepare_dataset.py
# nc={len(names)} must match num-detected-classes in the generated nvinfer
# config and the line count of labels/coco_labels.txt at deploy time.
path: {out}
train: images/train
val: {val}

nc: {len(names)}
names:
{entries}
"""
    )


def class_histogram(samples: list[Sample], names: list[str]) -> list[tuple[str, int]]:
    """Count boxes per class."""
    counts = [0] * len(names)
    for sample in samples:
        for box in sample.boxes:
            counts[box.cls] += 1
    return list(zip(names, counts))


def report(dataset: Dataset, train: list[Sample], val: list[Sample]) -> None:
    """Print dataset composition."""
    print(f"Classes: {len(dataset.names)}")
    for name, count in class_histogram(dataset.samples, dataset.names):
        print(f"  {name}: {count} boxes")

    empty = sum(1 for s in dataset.samples if not s.boxes)
    print(f"Images: {len(dataset.samples)} (train={len(train)} val={len(val)} background={empty})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=sorted(LOADERS), default="coco")
    parser.add_argument("--images", required=True, help="directory holding the train images")
    parser.add_argument("--annotations", required=True, help="train annotation file")
    parser.add_argument("--val-images", help="directory holding the val images")
    parser.add_argument("--val-annotations", help="val annotation file; omit to split from train")
    parser.add_argument("--val-split", type=float, default=0.1, help="fraction split off train when no val set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="dataset root to create, e.g. outputs/datasets/casualty")
    parser.add_argument("--link", choices=("symlink", "copy", "hardlink"), default="symlink")
    parser.add_argument("--rename-collisions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader = LOADERS[args.format]

    images = resolve_project_path(args.images)
    annotations = resolve_project_path(args.annotations)
    out = resolve_project_path(args.out)

    if not images.is_dir():
        raise SystemExit(f"Missing images directory: {images}")
    if not annotations.is_file():
        raise SystemExit(f"Missing annotations file: {annotations}")

    try:
        dataset = loader(annotations, images)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.val_annotations:
        val_images = resolve_project_path(args.val_images or args.images)
        val_annotations = resolve_project_path(args.val_annotations)
        if not val_annotations.is_file():
            raise SystemExit(f"Missing val annotations file: {val_annotations}")

        try:
            val_dataset = loader(val_annotations, val_images)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

        if val_dataset.names != dataset.names:
            raise SystemExit(
                "Train and val class lists differ:\n"
                f"  train: {dataset.names}\n"
                f"  val:   {val_dataset.names}"
            )
        train_samples = sorted(dataset.samples, key=lambda s: str(s.image))
        val_samples = sorted(val_dataset.samples, key=lambda s: str(s.image))
        dataset = Dataset(names=dataset.names, samples=train_samples + val_samples)
    else:
        train_samples, val_samples = split_samples(dataset.samples, args.val_split, args.seed)

    report(dataset, train_samples, val_samples)

    try:
        destination_names(train_samples, args.rename_collisions)
        destination_names(val_samples, args.rename_collisions)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    yaml_path = out / f"{out.name}.yaml"
    labels_path = out / f"{out.name}.labels.txt"

    if args.dry_run:
        print("\nDry run, nothing written. Would create:")
        print(f"  {out / 'images' / 'train'} ({len(train_samples)} images, {args.link})")
        print(f"  {out / 'images' / 'val'} ({len(val_samples)} images, {args.link})")
        print(f"  {out / 'labels' / 'train'}, {out / 'labels' / 'val'}")
        print(f"  {yaml_path}")
        print(f"  {labels_path}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    write_split(out, "train", train_samples, args.link, args.rename_collisions)
    write_split(out, "val", val_samples, args.link, args.rename_collisions)
    write_yaml(yaml_path, out, dataset.names, bool(val_samples))
    labels_path.write_text("\n".join(dataset.names) + "\n")

    print(f"\nDataset: {out}")
    print(f"YAML:    {yaml_path}")
    print(f"Labels:  {labels_path}")
    print(f"\nTrain:\n  python3 training/finetune.py --weights yolo11s.pt --data {yaml_path}")
    print(
        f"\nDeploy labels: this {len(dataset.names)}-class list must replace the 80-class\n"
        "labels/coco_labels.txt before export, or DeepStream renders COCO names.\n"
        "training/export_to_deepstream.py checks that and can install it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
