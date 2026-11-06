"""Image loading/saving: EXIF orientation, mode conversion, alpha, ICC/EXIF metadata."""

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from tqdm import tqdm

FORMATS = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP"}
EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


class ImageError(Exception):
    pass


@dataclass
class LoadedImage:
    rgb: np.ndarray  # H x W x 3 uint8
    alpha: Image.Image | None  # mode "L"
    format: str  # Pillow format of the source, e.g. "JPEG"
    icc: bytes | None
    exif: bytes | None


def _icc_space(icc: bytes | None) -> bytes | None:
    return icc[16:20] if icc and len(icc) >= 20 else None


def _normalize(im: Image.Image) -> tuple[Image.Image, bytes | None]:
    """Convert any mode to RGB or RGBA; returns the image and an ICC profile valid for RGB."""
    icc = im.info.get("icc_profile")
    if im.mode.startswith("I"):  # 16/32-bit grayscale: scale down, Pillow's convert would clip at 255
        im = Image.fromarray((np.asarray(im, np.float32) / 257).round().clip(0, 255).astype(np.uint8))
    if im.mode == "CMYK":
        if _icc_space(icc) == b"CMYK":
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            im = ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
            icc = SRGB_ICC
        else:
            im = im.convert("RGB")
    has_alpha = im.mode in ("RGBA", "RGBa", "LA", "La", "PA") or "transparency" in im.info
    im = im.convert("RGBA" if has_alpha else "RGB")
    # shortcut: non-RGB profiles (e.g. gray) are dropped, pixels treated as sRGB; convert via ImageCms if it matters.
    return im, icc if _icc_space(icc) == b"RGB " else None


def load(path: Path) -> LoadedImage:
    try:
        with Image.open(path) as im:
            fmt = "JPEG" if im.format == "MPO" else im.format  # phone JPEGs often open as MPO
            im = ImageOps.exif_transpose(im)  # also strips the orientation tag from info["exif"]
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageError(f"cannot read image ({e})") from None
    exif = im.info.get("exif")
    im, icc = _normalize(im)
    alpha = im.getchannel("A") if im.mode == "RGBA" else None
    return LoadedImage(np.array(im.convert("RGB")), alpha, fmt, icc, exif)


def target_size(w: int, h: int, scale: float) -> tuple[int, int]:
    return max(1, round(w * scale)), max(1, round(h * scale))


def finalize(native_rgb: np.ndarray, alpha: Image.Image | None, size: tuple[int, int]) -> tuple[Image.Image, Image.Image | None]:
    """Lanczos-resize the model output and the original alpha to the target size."""
    rgb = Image.fromarray(native_rgb)
    if rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    return rgb, alpha.resize(size, Image.Resampling.LANCZOS) if alpha else None


def save(rgb: Image.Image, alpha: Image.Image | None, path: Path, fmt: str, quality: int, icc: bytes | None, exif: bytes | None) -> None:
    if alpha is not None:
        if fmt == "JPEG":
            tqdm.write(f"{path.name}: JPEG has no alpha, flattening onto white")
            bg = Image.new("RGB", rgb.size, "white")
            bg.paste(rgb, mask=alpha)
            rgb = bg
        else:
            rgb.putalpha(alpha)
    # Pillow ignores options a format doesn't support (e.g. exif for BMP).
    opts = {"quality": quality, "icc_profile": icc, "exif": exif}
    tmp = path.with_name(path.name + ".part")  # never leave a half-written file that a later run would skip
    try:
        rgb.save(tmp, format=fmt, **{k: v for k, v in opts.items() if v is not None})
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
