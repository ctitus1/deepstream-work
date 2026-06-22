# DeepStream configs

Static config variants are intentionally not tracked here. Use
`scripts/setup_and_export_yolo.sh` or `src/deepstream_yolo_parser_app.py` to
generate model-specific configs under `configs/generated/`.

Generated configs contain machine-specific absolute paths, TensorRT engine
names, stream URIs, and confidence thresholds, so they are treated as local
runtime artifacts.
