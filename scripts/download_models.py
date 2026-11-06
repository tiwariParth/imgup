"""Download the default Real-ESRGAN weights into models/ and check they load with spandrel.

URLs are the official release assets from https://github.com/xinntao/Real-ESRGAN/releases
Run: uv run python scripts/download_models.py
"""

import sys
import urllib.request
from pathlib import Path

from spandrel import ModelLoader

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# filename -> (url, size in bytes as listed on the GitHub release)
WEIGHTS = {
    "RealESRGAN_x4plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        67040989,
    ),
    "RealESRGAN_x4plus_anime_6B.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        17938799,
    ),
}


def fetch(name: str, url: str, size: int) -> None:
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size == size:
        print(f"{name}: already present")
    else:
        print(f"{name}: downloading {url}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if (got := tmp.stat().st_size) != size:
            tmp.unlink()
            raise RuntimeError(f"size mismatch: expected {size} bytes, got {got}")
        tmp.replace(dest)
    desc = ModelLoader().load_from_file(dest)
    print(f"{name}: OK ({desc.architecture.name}, x{desc.scale})")


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    failed = []
    for name, (url, size) in WEIGHTS.items():
        try:
            fetch(name, url, size)
        except Exception as e:
            print(f"{name}: FAILED: {e}", file=sys.stderr)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
