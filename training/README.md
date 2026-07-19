# Detection Training

Fine-tune a detection model and deploy it through the repo's existing export
path (`scripts/setup_and_export_yolo.sh`). Nothing here replaces that script; it
only prepares data, trains, and validates the handoff.

```text
raw dataset -> prepare_dataset.py -> finetune.py -> export_to_deepstream.py -> setup_and_export_yolo.sh
```

## Environment

Training deps are separate from both the DeepStream system interpreter and
`.venv-yolo` (the export env):

```bash
python3 -m venv .venv-train
.venv-train/bin/python3 -m pip install --upgrade pip 'setuptools<82' wheel
.venv-train/bin/python3 -m pip install -r requirements/training.txt
```

`prepare_dataset.py` and `export_to_deepstream.py` run on the system
interpreter. `finetune.py` must run on `.venv-train/bin/python3`.
`export_to_deepstream.py` reads checkpoints through `.venv-yolo` (create it with
`scripts/setup_yolo_export_env.sh`).

## 1. Prepare the dataset

COCO-JSON in, ultralytics layout out:

```bash
python3 training/prepare_dataset.py \
  --format coco \
  --images /data/casualty/images \
  --annotations /data/casualty/instances_train.json \
  --out outputs/datasets/casualty \
  --dry-run
```

Drop `--dry-run` to write. Result:

```text
outputs/datasets/casualty/
  images/{train,val}/          symlinks by default (--link copy|hardlink)
  labels/{train,val}/          "cls cx cy w h", normalized
  casualty.yaml                dataset YAML for --data
  casualty.labels.txt          class names, one per line
```

Notes:

- Every image referenced by the annotations must exist; the script fails with
  the missing paths rather than training on a partial set.
- With no `--val-annotations`, a deterministic `--val-split` (default 0.1) is
  taken with `--seed`. Pass `--val-images/--val-annotations` for a real split;
  the class lists must match exactly.
- Class indices come from COCO `category_id` sorted ascending, remapped to
  contiguous `0..N-1`. `iscrowd` and zero-area boxes are dropped.
- Basename collisions across subdirectories are fatal unless
  `--rename-collisions` is passed.
- `outputs/` is gitignored, so datasets built there stay out of Git.

Other formats: add `load_x(annotations, images_dir) -> Dataset` and register it
in `LOADERS`.

## 2. Fine-tune

```bash
.venv-train/bin/python3 training/finetune.py \
  --weights yolo11s.pt \
  --data outputs/datasets/casualty/casualty.yaml \
  --epochs 100 --imgsz 640 --batch 16 --device 0
```

Runs land in `outputs/training/<name>/`, best weights at
`weights/best.pt`. `--resume` takes that run's `last.pt` and restores epochs,
imgsz, and batch from the checkpoint — the other flags are ignored.

## 3. Export to DeepStream

```bash
python3 training/export_to_deepstream.py \
  outputs/training/finetune/weights/best.pt \
  --stage-as yolo11s-custom.pt \
  --long-side 640 \
  --install-labels
```

It reads the checkpoint's class names, writes them to
`outputs/training/<stem>.labels.txt`, runs the checks below, then invokes
`scripts/setup_and_export_yolo.sh`. `--check-only` stops before exporting;
`--force` exports anyway. Verify first, then run the pipeline:

```bash
python3 src/parser_app.py --model yolo11s-custom.pt --long-side 640
```

---

## Sharp edges

These are real defects in the current repo, not hypotheticals. The export script
is owned by another work area, so `export_to_deepstream.py` detects them rather
than patching it.

### 1. The 80-class labels overwrite (silent, breaks every custom model)

In `scripts/setup_and_export_yolo.sh`:

- The DeepStream-Yolo exporter writes a correct per-class `labels.txt` at the
  repo root, one line per trained class.
- `install_labels()` then unconditionally copies the 80-class
  `labels/coco_labels.txt` over `models/coco_labels.txt` — which is what
  `labelfile-path` points at.
- The very next line, `rm -f labels.txt`, deletes the correct file.

A 3-class model therefore deploys with 80 COCO names and the OSD renders
garbage. Nothing errors.

Workaround: put the trained names in `labels/coco_labels.txt` **before**
exporting. `export_to_deepstream.py --install-labels` does that (backing up the
existing file to `coco_labels.txt.bak`), and refuses to export on a class-count
mismatch otherwise. It re-checks `models/coco_labels.txt` after the export.

### 2. `num-detected-classes` is hardcoded to 80

Two places, both must equal the trained class count:

- `scripts/setup_and_export_yolo.sh` writes `num-detected-classes=80` into
  `configs/generated/config_infer_primary_<stem>.txt`.
- `src/deepstream_yolo/configs.py:write_infer_config` also emits
  `num-detected-classes=80`, and `model_cache.ensure_model` calls it on **every**
  parser-app run — so it overwrites the generated file each time.

`--fix-generated-config` patches the generated file for direct
`deepstream-app -c` use, but the parser-app path needs `configs.py` changed by
its owner. Patching only the generated file is not a fix for `src/parser_app.py`.

### 3. Offline metrics are not measured at the deployed resolution

The exporter bakes a **static** input shape. `setup_and_export_yolo.sh` derives
it from the source aspect ratio and rounds to stride 32, so a 1920x1080 stream
at `--long-side 640` deploys **640x384**, and the generated config sets
`maintain-aspect-ratio=0` (streammux matches the model shape, no letterbox).

ultralytics `train`/`val` accept only a single square int for `imgsz`, so
training and validation happen at 640x640. Reported mAP is therefore **not** the
deployed-resolution number: vertical resolution is ~40% lower in production and
the aspect handling differs. Treat offline mAP as a relative signal for
comparing runs, and confirm accuracy on real frames after export.

### 4. YOLO11 vs YOLO12 export paths

- **YOLO11** fine-tunes and exports normally: `pip install ultralytics`, train
  with `finetune.py`, export via `utils/export_yolo11.py` (what the setup script
  calls for `yolo11*` stems).
- **YOLO12 does not come from ultralytics.** `pip install ultralytics` gives you
  no YOLO12; it lives in the `sunsmarterjie/yolov12` research repo, which ships
  its own patched ultralytics fork. Fine-tuning YOLO12 means cloning that repo
  into its own venv — installing it alongside `requirements/training.txt`
  produces two conflicting `ultralytics` packages.
- **Casing bug in the vendored docs:** `DeepStream-Yolo/docs/YOLOv12.md` tells
  you to copy and run `export_yoloV12.py` (capital V). The real file is
  `utils/export_yolov12.py` (lowercase v). Copy-pasting from the doc fails with
  "No such file or directory". `setup_and_export_yolo.sh` already uses the
  correct lowercase name.

### 5. The setup script only dispatches on known stems

Its `case` matches `yolo11*`, `yolo12*`/`yolov12*`, `yolov8*`/`yolo8*` and exits
1 on anything else. A run's `best.pt` matches none of them, so stage it under a
dispatchable name:

```bash
python3 training/export_to_deepstream.py <run>/weights/best.pt --stage-as yolo11s-custom.pt
```

`export_to_deepstream.py` checks this before invoking the script and suggests a
name derived from the checkpoint's base model.
