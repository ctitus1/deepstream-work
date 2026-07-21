"""Batch nvinfer config writer (DESIGN.md Sec 6 "Engines", Sec 9).

Writes ``ds_ros_pipeline/generated/ds_ros_infer_batch_yolo12x_640_640x384_b{N}
.txt`` — identical to the existing b1 primary yolo12x config (same ONNX
``models/yolo12x_640_640x384.onnx``, labels snapshot, custom parser
``lib/libnvdsinfer_custom_impl_Yolo.so``, net-scale-factor, letterboxing and
cluster settings) except ``batch-size=N`` and ``model-engine-file=
models/yolo12x_640_640x384.onnx_b{N}_gpu0_fp16.engine``. nvinfer builds and
caches that engine next to the ONNX on first start (the repo's engine
convention — the single one-folder-rule exception). The injury-CLIP SGIE
reuses ``configs/generated/config_infer_secondary_injury_clip_vit_l14_336_b8
.txt`` verbatim.

This module deliberately duplicates one small piece of config logic from
``deepstream_yolo.model_cache`` (whose writer hardcodes b1) rather than
modifying it (Sec 6). Pure file I/O — no Gst/pyds/rclpy; unit-tested against
a golden config (Sec 10).
"""

from __future__ import annotations

from pathlib import Path

GENERATED_DIR = Path(__file__).resolve().parent / "generated"

# SGIE config reused as-is (repo-root-relative), Sec 6.
SGIE_CONFIG_RELPATH = "configs/generated/config_infer_secondary_injury_clip_vit_l14_336_b8.txt"

# The existing b1 primary config the b8 one is derived from (Sec 6).
TEMPLATE_RELPATH = "configs/generated/config_infer_primary_yolo12x_640_640x384.txt"

GITIGNORE_TEXT = "*\n!.gitignore\n"

# Top-level repo dirs a template path may point into. The template embeds the
# container's absolute repo mount (/workspace/deepstream-work/...); existence
# checks re-anchor at whatever repo_root this process actually sees.
_REPO_DIRS = ("models", "lib", "configs")

_ARTIFACT_HINTS = {
    "onnx-file": "export it: scripts/setup/yolo_export.sh yolo12x.pt 640",
    "labelfile-path": "the export writes it next to the ONNX: scripts/setup/yolo_export.sh yolo12x.pt 640",
    "custom-lib-path": "build it: scripts/setup/yolo_parser.sh",
}

_TEMPLATE_HINT = (
    "it is written by deepstream_yolo.model_cache when the existing yolo12x "
    "pipeline runs (after scripts/setup/yolo_export.sh yolo12x.pt 640)"
)


def _default_repo_root() -> Path:
    try:
        from deepstream_yolo.paths import PROJECT_DIR
        return Path(PROJECT_DIR)
    except ImportError:
        return Path(__file__).resolve().parents[1]


def _parse_sections(text: str, origin: object) -> list[tuple[str, list[tuple[str, str]]]]:
    """key=value lines grouped by [section], order preserved, comments dropped."""
    sections: list[tuple[str, list[tuple[str, str]]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            sections.append((line[1:-1], []))
            continue
        if "=" not in line or not sections:
            raise ValueError(f"Unparseable nvinfer config line in {origin}: {raw!r}")
        key, _, value = line.partition("=")
        sections[-1][1].append((key.strip(), value.strip()))
    return sections


def _render_sections(sections: list[tuple[str, list[tuple[str, str]]]]) -> str:
    blocks = [
        "\n".join([f"[{name}]"] + [f"{key}={value}" for key, value in entries])
        for name, entries in sections
    ]
    return "\n\n".join(blocks) + "\n"


def _property_entries(
    sections: list[tuple[str, list[tuple[str, str]]]], origin: object,
) -> list[tuple[str, str]]:
    for name, entries in sections:
        if name == "property":
            return entries
    raise ValueError(f"No [property] section in {origin}; {_TEMPLATE_HINT}")


def _require(entries: list[tuple[str, str]], key: str, origin: object) -> str:
    for entry_key, value in entries:
        if entry_key == key:
            return value
    raise ValueError(f"No {key} in [property] of {origin}; {_TEMPLATE_HINT}")


def _reanchor(value: str, repo_root: Path) -> Path:
    """The path a template value denotes under this process's repo_root."""
    path = Path(value)
    if not path.is_absolute():
        return repo_root / path
    if path.is_file():
        return path
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in _REPO_DIRS:
            return repo_root.joinpath(*parts[index:])
    return path


def batch_yolo_config_path(engine_batch: int, generated_dir: Path = GENERATED_DIR) -> Path:
    """Pure: the output path for the b{engine_batch} yolo config."""
    return generated_dir / f"ds_ros_infer_batch_yolo12x_640_640x384_b{engine_batch}.txt"


def render_batch_yolo_config(engine_batch: int, repo_root: Path) -> str:
    """Pure: return the full config text for batch-size ``engine_batch``.

    Derived from the existing b1 primary config's settings with only
    batch-size and model-engine-file changed (Sec 6). Separated from the
    writer so the golden test compares strings.
    """
    if engine_batch < 1:
        raise ValueError(f"engine_batch must be >= 1, got {engine_batch}")
    template = Path(repo_root) / TEMPLATE_RELPATH
    if not template.is_file():
        raise FileNotFoundError(
            f"Template b1 config missing: {template}; {_TEMPLATE_HINT}"
        )
    sections = _parse_sections(template.read_text(), template)
    entries = _property_entries(sections, template)
    onnx = _require(entries, "onnx-file", template)
    for key in ("batch-size", "model-engine-file"):
        _require(entries, key, template)
    # Engine naming duplicates model_cache.artifacts_for_onnx, generalized to N.
    overrides = {
        "batch-size": str(engine_batch),
        "model-engine-file": f"{onnx}_b{engine_batch}_gpu0_fp16.engine",
    }
    rewritten = [(key, overrides.get(key, value)) for key, value in entries]
    return _render_sections(
        [(name, rewritten if name == "property" else section_entries)
         for name, section_entries in sections]
    )


def write_batch_yolo_config(engine_batch: int = 8,
                            repo_root: Path | None = None,
                            generated_dir: Path = GENERATED_DIR) -> Path:
    """Write the batch yolo config and return its path (Sec 9 wiring).

    Asserts the ONNX, labels file, and custom parser .so exist under
    ``repo_root`` (default: the repo root containing this package, resolved
    via deepstream_yolo.paths) and raises FileNotFoundError otherwise.
    Creates ``generated_dir``. Called from the main thread at startup before
    the batch pipeline is built; also importable standalone for a run.sh
    --prebuild. Blocking file I/O only.
    """
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    text = render_batch_yolo_config(engine_batch, root)
    entries = _property_entries(_parse_sections(text, "rendered config"), "rendered config")
    missing = []
    for key, hint in _ARTIFACT_HINTS.items():
        value = _require(entries, key, "rendered config")
        resolved = _reanchor(value, root)
        if not resolved.is_file():
            missing.append(f"  {key}={value} (checked {resolved}) — {hint}")
    if missing:
        raise FileNotFoundError(
            "Batch nvinfer config references files that do not exist under "
            f"repo root {root}:\n" + "\n".join(missing)
        )
    generated_dir.mkdir(parents=True, exist_ok=True)
    gitignore = generated_dir / ".gitignore"
    if not gitignore.is_file():
        gitignore.write_text(GITIGNORE_TEXT)
    path = batch_yolo_config_path(engine_batch, generated_dir)
    path.write_text(text)
    return path


def sgie_config_path(repo_root: Path | None = None) -> Path:
    """Absolute path of the reused injury-CLIP b8 config; asserts it exists."""
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    path = root / SGIE_CONFIG_RELPATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Injury-CLIP SGIE b8 config missing: {path} — generate it (and "
            "its engine): scripts/setup/injury_model.sh models/injury.pt 8"
        )
    return path
