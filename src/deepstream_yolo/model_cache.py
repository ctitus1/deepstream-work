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
from .paths import INJURY_SETUP_SCRIPT, MODELS_DIR, PROJECT_DIR, SETUP_SCRIPT, YOLO_PYTHON
from .stream_source import StreamSource


CACHE_POLICY = "parser_line_osd_conf_v2"


@dataclass(frozen=True)
class ModelArtifacts:
    onnx: Path
    meta: Path
    engine: Path
    config: Path


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


def model_stem(model: str) -> str:
    return Path(model).stem


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
    )


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
    requested = meta.get("requested_long_side")
    if requested is not None and int(requested) != long_side:
        return False

    source_width = meta.get("source_width")
    source_height = meta.get("source_height")
    if source_width is not None and int(source_width) != src_w:
        return False
    if source_height is not None and int(source_height) != src_h:
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
    stem = model_stem(model)

    for candidate in sorted(MODELS_DIR.glob(f"{stem}_{long_side}_*.onnx")):
        artifacts = artifacts_for_onnx(candidate)
        meta = read_meta(artifacts.meta)
        if not meta_matches(meta, long_side, src_w, src_h):
            continue
        width, height = size_from_meta(meta) or onnx_size(candidate)
        write_infer_config(artifacts.config, artifacts.onnx, artifacts.engine, conf)
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
                "labels": "models/coco_labels.txt",
                "cache_policy": CACHE_POLICY,
            },
            indent=2,
        )
        + "\n"
    )
    write_infer_config(artifacts.config, artifacts.onnx, artifacts.engine, conf)
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

    meta = read_meta(artifacts.meta)
    if not artifacts.onnx.exists():
        raise FileNotFoundError(f"Missing injury ONNX export: {artifacts.onnx}")
    if not meta:
        raise FileNotFoundError(f"Missing or invalid injury model metadata: {artifacts.meta}")

    write_assessment_config(artifacts.config, artifacts.onnx, artifacts.engine, batch_size)
    return meta, artifacts.config
