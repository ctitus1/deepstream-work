from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

from .configs import CLIP_INPUT_SIZE, INJURY_CLASS_COUNTS, INJURY_HEADS, write_assessment_config
from .paths import MODELS_DIR, PROJECT_DIR

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
ONNX_SUFFIX = "clip_vit_l14_336"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing torch. Run this exporter with a Python environment that has "
            "torch, clip, and onnx installed."
        ) from exc
    return torch


def _import_clip_model():
    try:
        from clip.model import CLIP
    except ImportError as exc:
        raise RuntimeError(
            "Missing the OpenAI CLIP Python package. Install the export "
            "dependencies before exporting injury.pt."
        ) from exc
    return CLIP


def resolve_model_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def assessment_stem(model_path: Path) -> str:
    return f"{model_path.stem}_{ONNX_SUFFIX}"


def default_onnx_path(model_path: Path) -> Path:
    return MODELS_DIR / f"{assessment_stem(model_path)}.onnx"


def default_meta_path(model_path: Path) -> Path:
    return MODELS_DIR / f"{assessment_stem(model_path)}.meta.json"


def default_engine_path(model_path: Path, batch_size: int) -> Path:
    return Path(f"{default_onnx_path(model_path)}_b{batch_size}_gpu0_fp16.engine")


def load_state_dict(path: Path):
    torch = _import_torch()
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        kwargs["mmap"] = True
        state = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        state = torch.load(path, **kwargs)

    if not isinstance(state, (dict, OrderedDict)):
        raise TypeError(f"Expected a state dict in {path}, got {type(state).__name__}")
    return state


def split_state_dict(state: dict):
    clip_state = OrderedDict()
    head_state = OrderedDict()

    for key, value in state.items():
        if key.startswith("clip_model."):
            clip_state[key.removeprefix("clip_model.")] = value
        elif key.startswith("heads."):
            head_state[key.removeprefix("heads.")] = value

    missing = []
    for head_name in INJURY_HEADS:
        for suffix in ("weight", "bias"):
            key = f"{head_name}.{suffix}"
            if key not in head_state:
                missing.append(f"heads.{key}")

    if missing:
        raise KeyError(f"Missing injury head weights: {', '.join(missing)}")

    return clip_state, head_state


def infer_clip_spec(state: dict) -> dict:
    visual_blocks = sorted(
        {
            int(match.group(1))
            for key in state
            if (match := re.match(r"clip_model\.visual\.transformer\.resblocks\.(\d+)\.", key))
        }
    )
    text_blocks = sorted(
        {
            int(match.group(1))
            for key in state
            if (match := re.match(r"clip_model\.transformer\.resblocks\.(\d+)\.", key))
        }
    )

    visual_pos = state["clip_model.visual.positional_embedding"].shape[0]
    patch_size = state["clip_model.visual.conv1.weight"].shape[-1]
    visual_grid = round((visual_pos - 1) ** 0.5)

    heads = OrderedDict()
    for head_name in INJURY_HEADS:
        weight = state[f"heads.{head_name}.weight"]
        heads[head_name] = {
            "classes": int(weight.shape[0]),
            "in_features": int(weight.shape[1]),
        }

    return {
        "checkpoint_type": type(state).__name__,
        "state_dict_keys": len(state),
        "architecture": "CLIP ViT-L/14@336 image encoder with custom injury heads",
        "image_resolution": int(visual_grid * patch_size),
        "input_channels": 3,
        "input_format": "RGB float image scaled to [0, 1], normalized inside ONNX",
        "clip_mean": CLIP_MEAN,
        "clip_std": CLIP_STD,
        "patch_size": int(patch_size),
        "visual_grid": int(visual_grid),
        "visual_width": int(state["clip_model.visual.conv1.weight"].shape[0]),
        "visual_layers": len(visual_blocks),
        "visual_heads": int(state["clip_model.visual.conv1.weight"].shape[0] // 64),
        "embedding_dim": int(state["clip_model.visual.proj"].shape[1]),
        "context_length": int(state["clip_model.positional_embedding"].shape[0]),
        "vocab_size": int(state["clip_model.token_embedding.weight"].shape[0]),
        "transformer_width": int(state["clip_model.token_embedding.weight"].shape[1]),
        "transformer_layers": len(text_blocks),
        "transformer_heads": int(state["clip_model.token_embedding.weight"].shape[1] // 64),
        "heads": heads,
    }


def build_clip_model(clip_state: dict):
    CLIP = _import_clip_model()

    vision_width = clip_state["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [
            key
            for key in clip_state
            if key.startswith("visual.") and key.endswith(".attn.in_proj_weight")
        ]
    )
    vision_patch_size = clip_state["visual.conv1.weight"].shape[-1]
    grid_size = round((clip_state["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = clip_state["text_projection"].shape[1]
    context_length = clip_state["positional_embedding"].shape[0]
    vocab_size = clip_state["token_embedding.weight"].shape[0]
    transformer_width = clip_state["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(key.split(".")[2] for key in clip_state if key.startswith("transformer.resblocks"))
    )

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )
    model.load_state_dict(clip_state, strict=True)
    return model.float().eval()


def build_assessment_model(state: dict, *, normalize_features: bool = False):
    torch = _import_torch()
    clip_state, head_state = split_state_dict(state)

    class InjuryAssessmentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip_model = build_clip_model(clip_state)
            self.normalize_features = normalize_features
            self.head_names = tuple(INJURY_HEADS)
            self.heads = torch.nn.ModuleDict()
            self.register_buffer(
                "clip_mean",
                torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "clip_std",
                torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )

            for head_name in self.head_names:
                weight = head_state[f"{head_name}.weight"]
                bias = head_state[f"{head_name}.bias"]
                expected = INJURY_CLASS_COUNTS[head_name]
                if int(weight.shape[0]) != expected:
                    raise ValueError(
                        f"Unexpected class count for {head_name}: "
                        f"{int(weight.shape[0])} != {expected}"
                    )

                layer = torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
                layer.load_state_dict({"weight": weight.float(), "bias": bias.float()})
                self.heads[head_name] = layer

        def forward(self, images):
            images = (images - self.clip_mean) / self.clip_std
            features = self.clip_model.encode_image(images).float()
            if self.normalize_features:
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return tuple(self.heads[name](features) for name in self.head_names)

    return InjuryAssessmentModel().eval()


def export_onnx(
    model_path: Path,
    onnx_path: Path,
    meta_path: Path,
    *,
    batch_size: int,
    opset: int = 18,
    normalize_features: bool = False,
) -> dict:
    torch = _import_torch()
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Missing onnx. Install export dependencies before exporting.") from exc

    model_path = resolve_model_path(model_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    stale_external_data = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if stale_external_data.exists():
        stale_external_data.unlink()

    state = load_state_dict(model_path)
    spec = infer_clip_spec(state)
    if spec["image_resolution"] != CLIP_INPUT_SIZE:
        raise ValueError(
            f"Expected injury model input {CLIP_INPUT_SIZE}, got {spec['image_resolution']}"
        )

    model = build_assessment_model(state, normalize_features=normalize_features)
    dummy = torch.zeros((1, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE), dtype=torch.float32)
    dynamic_axes = {"images": {0: "batch"}}
    for output_name in INJURY_HEADS:
        dynamic_axes[output_name] = {0: "batch"}

    export_kwargs = {
        "input_names": ["images"],
        "output_names": list(INJURY_HEADS),
        "dynamic_axes": dynamic_axes,
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamo": False,
    }
    try:
        export_kwargs["external_data"] = True
        torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
    except TypeError:
        export_kwargs.pop("external_data", None)
        torch.onnx.export(model, dummy, onnx_path, **export_kwargs)

    loaded = onnx.load(str(onnx_path))
    onnx.checker.check_model(loaded)

    meta = {
        **spec,
        "source_checkpoint": str(model_path.relative_to(PROJECT_DIR)),
        "onnx": str(onnx_path.relative_to(PROJECT_DIR)),
        "default_batch_size": batch_size,
        "normalize_features": normalize_features,
        "opset": opset,
        "outputs": list(INJURY_HEADS),
        "runtime": "DeepStream secondary nvinfer, TensorRT fp16",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def write_config_for_model(model_path: Path, batch_size: int) -> Path:
    model_path = resolve_model_path(model_path)
    onnx_path = default_onnx_path(model_path)
    engine_path = default_engine_path(model_path, batch_size)
    config_path = PROJECT_DIR / "configs" / "generated" / (
        f"config_infer_secondary_{assessment_stem(model_path)}_b{batch_size}.txt"
    )
    write_assessment_config(config_path, onnx_path, engine_path, batch_size)
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--model", default="models/injury.pt")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--model", default="models/injury.pt")
    export_parser.add_argument("--batch-size", type=int, default=8)
    export_parser.add_argument("--opset", type=int, default=18)
    export_parser.add_argument("--normalize-features", action="store_true")

    args = parser.parse_args()
    model_path = resolve_model_path(args.model)

    if args.command == "inspect":
        print(json.dumps(infer_clip_spec(load_state_dict(model_path)), indent=2))
        return 0

    onnx_path = default_onnx_path(model_path)
    meta_path = default_meta_path(model_path)
    meta = export_onnx(
        model_path,
        onnx_path,
        meta_path,
        batch_size=args.batch_size,
        opset=args.opset,
        normalize_features=args.normalize_features,
    )
    config_path = write_config_for_model(model_path, args.batch_size)
    print("Injury model export complete.")
    print(f"PT:           {model_path.relative_to(PROJECT_DIR)}")
    print(f"ONNX:         {onnx_path.relative_to(PROJECT_DIR)}")
    print(f"Engine:       {default_engine_path(model_path, args.batch_size).relative_to(PROJECT_DIR)}")
    print(f"Config:       {config_path.relative_to(PROJECT_DIR)}")
    print(f"Input:        3x{CLIP_INPUT_SIZE}x{CLIP_INPUT_SIZE}")
    print(f"Outputs:      {', '.join(meta['outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
