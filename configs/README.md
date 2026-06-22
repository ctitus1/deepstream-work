# DeepStream configs

Static config variants are intentionally not tracked here. Use
`scripts/setup_and_export_yolo.sh`, `scripts/setup_injury_model.sh`, or
`src/deepstream_yolo_parser_app.py` to generate model-specific primary and
secondary configs under `configs/generated/`.

Generated configs contain machine-specific absolute paths, TensorRT engine
names, stream URIs, and confidence thresholds, so they are treated as local
runtime artifacts.
