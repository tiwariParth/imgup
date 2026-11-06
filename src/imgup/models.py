"""Model resolution and loading via spandrel."""

from pathlib import Path

import torch
from spandrel import ImageModelDescriptor, ModelLoader, UnsupportedModelError

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Short names -> weight filenames in models/ (see scripts/download_models.py).
ALIASES = {
    "x4plus": "RealESRGAN_x4plus.pth",
    "anime": "RealESRGAN_x4plus_anime_6B.pth",
}


class ModelError(Exception):
    pass


def resolve_model_path(name_or_path: str) -> Path:
    """Accept an alias, a filename/stem in models/, or a path to a weights file."""
    candidates = [Path(name_or_path).expanduser()]
    if name_or_path in ALIASES:
        candidates.insert(0, MODELS_DIR / ALIASES[name_or_path])
    candidates += [MODELS_DIR / name_or_path] + [MODELS_DIR / f"{name_or_path}{ext}" for ext in (".pth", ".safetensors")]
    for c in candidates:
        if c.is_file():
            return c
    available = sorted(p.name for p in MODELS_DIR.glob("*") if p.suffix in (".pth", ".safetensors"))
    raise ModelError(
        f"Model '{name_or_path}' not found. Aliases: {', '.join(ALIASES)}. "
        f"In {MODELS_DIR}: {', '.join(available) or 'nothing (run scripts/download_models.py)'}."
    )


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path: Path, device: torch.device, half: bool = True) -> ImageModelDescriptor:
    try:
        desc = ModelLoader(device="cpu").load_from_file(path)
    except UnsupportedModelError:
        raise ModelError(f"{path.name}: architecture not supported by spandrel.") from None
    except Exception as e:
        raise ModelError(f"{path.name}: could not load weights ({e}).") from None
    if not isinstance(desc, ImageModelDescriptor):
        raise ModelError(f"{path.name}: not a single-image model (e.g. it's an inpainting model).")
    desc.eval().to(device)
    # fp16 on CPU is slow or unsupported for many ops; keep CPU in fp32.
    if half and desc.supports_half and device.type == "cuda":
        desc.half()
    return desc
