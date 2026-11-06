"""Download the default Real-ESRGAN weights into models/ and check they load with spandrel.

URLs are the official release assets from https://github.com/xinntao/Real-ESRGAN/releases
Run: uv run python scripts/download_models.py
"""

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from spandrel import ModelLoader

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# filename -> (url, SHA-256 of the official release file)
WEIGHTS = {
    "RealESRGAN_x4plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    ),
    "RealESRGAN_x4plus_anime_6B.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    ),
}


def sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(name: str, url: str, digest: str) -> None:
    dest = MODELS_DIR / name
    if dest.exists() and sha256(dest) == digest:
        print(f"{name}: already present")
    else:
        print(f"{name}: downloading {url}")
        tmp = dest.with_suffix(".part")
        # timeout applies per network read, so a stalled connection fails instead of hanging
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if (got := sha256(tmp)) != digest:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch: expected {digest}, got {got}")
        tmp.replace(dest)
    desc = ModelLoader().load_from_file(dest)
    print(f"{name}: OK ({desc.architecture.name}, x{desc.scale})")


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    failed = []
    for name, (url, digest) in WEIGHTS.items():
        try:
            fetch(name, url, digest)
        except Exception as e:
            print(f"{name}: FAILED: {e}", file=sys.stderr)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
