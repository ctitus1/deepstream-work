# Detection accuracy validation

Measure what the *deployed* DeepStream pipeline detects, and compare it to the
model as trained. `dump_detections.py` runs the shared pipeline over a video or
RTSP source and writes COCO-format JSON; `score_detections.py` scores that JSON
against COCO ground truth with pycocotools.

```bash
# rung 3: what the deployed pipeline actually detects
python3 validation/accuracy/dump_detections.py \
  --stream streams/dtc-d4-trimmed.mp4 \
  --output outputs/validation/detections.json

# score it (pycocotools lives in .venv-yolo, not the system interpreter)
.venv-yolo/bin/python3 validation/accuracy/score_detections.py \
  --detections outputs/validation/detections.json \
  --ground-truth /path/to/instances.json
```

Install the scorer dependency once: `.venv-yolo/bin/python3 -m pip install pycocotools`.

## The three-rung ladder

A single accuracy number cannot tell you *where* a regression came from. Run
all three rungs; each one holds the previous constant, so the rung where the
number drops is the layer that broke.

| Rung | What runs | What it isolates |
| --- | --- | --- |
| 1. PyTorch baseline | `ultralytics` `val` on the `.pt` in `.venv-yolo` | Model quality itself. The ceiling every other rung is measured against. |
| 2. Offline ONNX / TRT | The exported `models/*.onnx` and its `*_b1_gpu0_fp16.engine` over the same images, outside GStreamer | Export and precision. Opset/graph-surgery damage, FP16 range loss, letterbox and normalization drift. |
| 3. Deployed DeepStream | `dump_detections.py` + `score_detections.py` | Everything the deployment adds: `nvinfer` preprocessing, threshold config, non-square input geometry, decoder color conversion, dropped frames on live RTSP. |

Read the drops this way:

- Rung 1 low: the model is the problem, not the deployment.
- Rung 1 fine, rung 2 low: the export is lossy. Suspect FP16 (rebuild the
  engine at `network-mode=0`/FP32 to confirm) or the ONNX export path.
- Rungs 1-2 fine, rung 3 low: the pipeline is the problem. Check the four
  sharp edges below before touching the model.

Rung 3 is the only rung that can drop frames. On a live RTSP source
`build_pipeline()` sets `drop-on-latency`, leaky queues, and
`batched-push-timeout=0`, so a slow GPU shows up as *missing frames*, not as
lower per-frame accuracy. Dump from a local file when you want every frame
scored, and from RTSP when you want to know what the live system sees.

## Sharp edges

These four make a healthy model look broken. The tools surface each one in the
dump metadata and the score report.

### 1. Threshold mismatch (the big one)

`configs.write_infer_config()` writes `pre-cluster-threshold` from `--conf`
(0.25 is the deployment value used here; `parser_app.py` runs 0.2) and hardcodes
`nms-iou-threshold=0.45`. Ultralytics `val` defaults to `conf=0.001` and
`iou=0.7`.

A 0.25 confidence floor **truncates the precision-recall curve**: every
detection below 0.25 is discarded before scoring, so the low-precision /
high-recall tail that mAP integrates over simply does not exist. The deployed
model can score dramatically worse than the baseline while being byte-identical.

- For a *deployment* number, dump at the deployed threshold. That is what the
  system really emits.
- For a *comparison* against a PyTorch baseline, dump with
  `--conf 0.001` so the curve is complete.
- Never compare a 0.25-floored mAP to an ultralytics number and call it a
  regression.

`score_detections.py` reads the thresholds back out of the generated nvinfer
config and prints them on every report.

### 2. `category_id` off-by-one

Ultralytics maps predictions through `coco80_to_coco91_class()` when it decides
the dataset is COCO. Contiguous model class ids (`0..N-1`) are **not** COCO
category ids: contiguous 0 is `person`, COCO's `person` is category 1, and the
91-class ids skip numbers (11 -> 13, 25 -> 27, ...).

The mapping here is explicit and configurable, never implicit:

```bash
--category-map coco91     # contiguous 0..79 -> 91-class COCO ids (default)
--category-map identity   # model class id unchanged (custom datasets)
--category-map offset1    # class id + 1
```

Get it wrong and mAP is exactly 0.0 with no error, because no predicted
category matches any ground-truth category. `score_detections.py` checks the
dumped category ids against the ground-truth categories and says so, and
`--category-map` on the scorer re-maps an existing dump without re-running the
pipeline.

### 3. `image_id` typing

Ultralytics uses `int(stem)` when the filename stem is numeric and the string
otherwise. pycocotools matches by value **and** type, so a dump with `"42"` and
ground truth with `42` silently scores nothing.

```bash
--image-id-mode frame --image-id-offset 0   # image_id = frame number (default)
--image-id-mode map --image-id-map ids.json # explicit frame -> ground-truth id
--image-id-type int|str|auto                # auto reproduces the ultralytics rule
```

`auto` is provided for parity with ultralytics, but it is the mode that
produces mixed types; the dumper refuses to write a file whose ids are not all
one type, and the scorer reports a type mismatch against ground truth instead
of returning 0.0.

### 4. Non-square deploy resolution

The deployed model is exported to fit the source aspect ratio (e.g. 640x384 for
16:9), while offline eval letterboxes into a square (640x640) with
`maintain-aspect-ratio=1` and `symmetric-padding=1`. Different effective pixel
scale means different small-object recall and slightly different box
regression. Expect a small systematic gap between rung 2 and rung 3, and do not
attribute it to the pipeline until it exceeds a point or two of mAP. The dump
records both geometries; the score report flags a non-square deployment.

## Other things worth checking before blaming the model

- `--images overlap` (default) scores only ground-truth images the dump
  actually covered. `--images all` counts uncovered images as complete misses,
  which is the honest number when the dump was supposed to cover the whole set.
- `--class-ids 0` dumps person only; the scorer then restricts categories to
  match, otherwise every other class scores 0 and drags mAP down.
- Boxes come out of `nvinfer` in source-video coordinates. If the ground truth
  is in a different resolution, pass `--target-size WxH`.
