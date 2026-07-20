import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import Gst, GstPbutils

from .configs import generated_config_path, write_assessment_config, write_infer_config
from .injury import assessment_stem, default_engine_path, default_meta_path, default_onnx_path
from .paths import (
    INJURY_SETUP_SCRIPT,
    LABELS_PATH,
    MODELS_DIR,
    PROJECT_DIR,
    SETUP_SCRIPT,
    YOLO_PYTHON,
)
from .stream_source import StreamSource


CACHE_POLICY = "parser_line_osd_conf_v2"


@dataclass(frozen=True)
class ModelArtifacts:
    onnx: Path
    meta: Path
    engine: Path
    config: Path
    # Per-model snapshot of the class names, taken at export time. Configs point
    # here rather than at the shared models/coco_labels.txt, which only ever
    # reflects the most recent export.
    labels: Path


@dataclass(frozen=True)
class AssessmentArtifacts:
    onnx: Path
    meta: Path
    engine: Path
    config: Path


def discover_size(stream_uri: str) -> tuple[int, int]:
    info = GstPbutils.Discoverer.new(10 * Gst.SECOND).discover_uri(stream_uri)
    stream = info.get_video_streams()[0]
    return int(stream.get_width()), int(stream.get_height())


def onnx_size(path: Path) -> tuple[int, int]:
    if not YOLO_PYTHON.exists():
        raise FileNotFoundError(f"Missing YOLO export Python environment: {YOLO_PYTHON}")

    code = (
        "import onnx;"
        f"m=onnx.load({str(path)!r});"
        "d=[x.dim_value or x.dim_param for x in m.graph.input[0].type.tensor_type.shape.dim];"
        "print(int(d[3]), int(d[2]))"
    )
    out = subprocess.check_output([str(YOLO_PYTHON), "-c", code], text=True)
    return tuple(map(int, out.split()))


def artifacts_for_onnx(onnx: Path) -> ModelArtifacts:
    return ModelArtifacts(
        onnx=onnx,
        meta=onnx.with_suffix(".meta.json"),
        engine=Path(f"{onnx}_b1_gpu0_fp16.engine"),
        config=generated_config_path(f"config_infer_primary_{onnx.stem}.txt"),
        labels=onnx.with_suffix(".labels.txt"),
    )


def snapshot_model_labels(artifacts: ModelArtifacts) -> Path:
    """Copy the labels the export just installed into this model's own file.

    Always overwrites: re-exporting an existing tag with a different class set
    must replace the snapshot, not keep the previous one.
    """
    if not LABELS_PATH.is_file():
        return LABELS_PATH
    artifacts.labels.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LABELS_PATH, artifacts.labels)
    return artifacts.labels


def ensure_model_labels(artifacts: ModelArtifacts) -> Path:
    """Return this model's own labels without disturbing an existing snapshot.

    LABELS_PATH is shared and holds whatever the last export installed, so a
    config pointing at it can end up describing a different model. Entries
    cached before per-model labels existed are migrated from it once.
    """
    if artifacts.labels.is_file():
        return artifacts.labels
    return snapshot_model_labels(artifacts)


def tagged_artifacts(stem: str, long_side: int, width: int, height: int) -> ModelArtifacts:
    tag = f"{stem}_{long_side}_{width}x{height}"
    return artifacts_for_onnx(MODELS_DIR / f"{tag}.onnx")


def read_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def meta_matches(meta: dict, long_side: int, src_w: int, src_h: int) -> bool:
    # read_meta() reports a missing or unparseable file as {}. Every check here
    # used to be guarded on "present and equal", so an empty meta matched every
    # resolution. Ctrl-C between copying the ONNX and writing its meta was
    # enough to leave that state behind, after which the model was a cache hit
    # for any source and silently reused at the wrong aspect ratio.
    if not meta:
        return False

    # The policy tag exists so a change to config generation can invalidate
    # previously cached artifacts. It was written but never compared, so
    # bumping it invalidated nothing.
    if meta.get("cache_policy") != CACHE_POLICY:
        return False

    for key, expected in (
        ("requested_long_side", long_side),
        ("source_width", src_w),
        ("source_height", src_h),
    ):
        value = meta.get(key)
        if value is None or int(value) != expected:
            return False

    return True


def size_from_meta(meta: dict) -> tuple[int, int] | None:
    width = meta.get("model_width")
    height = meta.get("model_height")
    if width is None or height is None:
        return None
    return int(width), int(height)


def stream_for_setup(stream: StreamSource) -> str:
    if stream.path is None:
        return stream.raw

    try:
        return str(stream.path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(stream.path)


def ensure_model(
    model: str,
    stream: StreamSource,
    long_side: int,
    src_w: int,
    src_h: int,
    conf: float,
) -> tuple[int, int, Path]:
    stem = Path(model).stem

    for candidate in sorted(MODELS_DIR.glob(f"{stem}_{long_side}_*.onnx")):
        artifacts = artifacts_for_onnx(candidate)
        meta = read_meta(artifacts.meta)
        if not meta_matches(meta, long_side, src_w, src_h):
            continue
        width, height = size_from_meta(meta) or onnx_size(candidate)
        write_infer_config(
            artifacts.config,
            artifacts.onnx,
            artifacts.engine,
            conf,
            labels_path=ensure_model_labels(artifacts),
        )
        return width, height, artifacts.config

    env = os.environ.copy()
    env["SOURCE_WIDTH"] = str(src_w)
    env["SOURCE_HEIGHT"] = str(src_h)

    subprocess.run(
        [str(SETUP_SCRIPT), model, str(long_side), stream_for_setup(stream)],
        cwd=PROJECT_DIR,
        env=env,
        check=True,
    )

    base = MODELS_DIR / f"{stem}.onnx"
    width, height = onnx_size(base)
    artifacts = tagged_artifacts(stem, long_side, width, height)

    shutil.copy2(base, artifacts.onnx)
    # The engine is derived from the ONNX but named independently of its
    # contents, and nvinfer deserializes whatever engine it finds without
    # checking it against the ONNX beside it. Re-exporting the same tag -- a new
    # checkpoint at the same resolution, say -- would otherwise keep running the
    # previous model's weights, silently and indefinitely.
    artifacts.engine.unlink(missing_ok=True)
    artifacts.meta.write_text(
        json.dumps(
            {
                "model": stem,
                "model_arg": model,
                "requested_long_side": long_side,
                "source_stream": stream_for_setup(stream),
                "source_width": src_w,
                "source_height": src_h,
                "model_width": width,
                "model_height": height,
                "onnx": str(artifacts.onnx.relative_to(PROJECT_DIR)),
                "engine": str(artifacts.engine.relative_to(PROJECT_DIR)),
                "labels": str(artifacts.labels.relative_to(PROJECT_DIR)),
                "cache_policy": CACHE_POLICY,
            },
            indent=2,
        )
        + "\n"
    )
    write_infer_config(
        artifacts.config,
        artifacts.onnx,
        artifacts.engine,
        conf,
        labels_path=snapshot_model_labels(artifacts),
    )
    return width, height, artifacts.config


def ensure_assessment_model(model: str, batch_size: int) -> tuple[dict, Path]:
    model_path = PROJECT_DIR / model if not Path(model).is_absolute() else Path(model)
    artifacts = AssessmentArtifacts(
        onnx=default_onnx_path(model_path),
        meta=default_meta_path(model_path),
        engine=default_engine_path(model_path, batch_size),
        config=generated_config_path(
            f"config_infer_secondary_{assessment_stem(model_path)}_b{batch_size}.txt"
        ),
    )

    if not artifacts.onnx.exists() or not artifacts.meta.exists():
        subprocess.run(
            [str(INJURY_SETUP_SCRIPT), str(model_path), str(batch_size)],
            cwd=PROJECT_DIR,
            check=True,
        )
        # Same stale-engine trap as the detector above: a re-export must not
        # leave the previous checkpoint's engine sitting at the derived path.
        artifacts.engine.unlink(missing_ok=True)

    meta = read_meta(artifacts.meta)
    if not artifacts.onnx.exists():
        raise FileNotFoundError(f"Missing injury ONNX export: {artifacts.onnx}")
    if not meta:
        raise FileNotFoundError(f"Missing or invalid injury model metadata: {artifacts.meta}")

    write_assessment_config(artifacts.config, artifacts.onnx, artifacts.engine, batch_size)
    return meta, artifacts.config
