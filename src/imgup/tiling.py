"""Tiled inference with overlap padding, written tile by tile into a uint8 buffer."""

import numpy as np
import torch
from spandrel import ImageModelDescriptor
from tqdm import tqdm

MIN_TILE = 32


class UpscaleError(Exception):
    pass


class _NonFinite(Exception):
    pass


def tile_boxes(h: int, w: int, tile: int) -> list[tuple[int, int, int, int]]:
    """(y0, y1, x0, x1) boxes covering an h x w image."""
    return [(y, min(y + tile, h), x, min(x + tile, w)) for y in range(0, h, tile) for x in range(0, w, tile)]


@torch.inference_mode()
def run_tiles(img: np.ndarray, model: ImageModelDescriptor, tile: int, pad: int, progress: bool = False) -> np.ndarray:
    """Upscale an HxWx3 uint8 array at the model's native scale. Each tile is run with `pad`
    pixels of real context on every side, and only its centre is kept, so tiles don't show seams."""
    h, w, _ = img.shape
    s = model.scale
    out = np.empty((h * s, w * s, 3), np.uint8)
    src = torch.from_numpy(img).permute(2, 0, 1)
    for y0, y1, x0, x1 in tqdm(tile_boxes(h, w, tile), disable=not progress, unit="tile", leave=False):
        py0, py1, px0, px1 = max(y0 - pad, 0), min(y1 + pad, h), max(x0 - pad, 0), min(x1 + pad, w)
        t = src[None, :, py0:py1, px0:px1].to(model.device, model.dtype) / 255
        res = model(t)[0, :, (y0 - py0) * s : (y1 - py0) * s, (x0 - px0) * s : (x1 - px0) * s]
        if not torch.isfinite(res).all():
            raise _NonFinite
        out[y0 * s : y1 * s, x0 * s : x1 * s] = res.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return out


def upscale(
    img: np.ndarray,
    model: ImageModelDescriptor,
    tile: int = 256,
    pad: int = 16,
    progress: bool = False,
    min_tile: int = MIN_TILE,
) -> np.ndarray:
    """run_tiles with recovery: on CUDA OOM halve the tile size down to `min_tile`; on NaN/Inf in fp16, switch the model to fp32."""
    while True:
        try:
            return run_tiles(img, model, tile, pad, progress)
        except torch.OutOfMemoryError:
            if tile // 2 < min_tile:
                raise UpscaleError(
                    f"Out of GPU memory even at tile size {tile}. Close other GPU apps, or use a smaller model."
                ) from None
            tile //= 2
            tqdm.write(f"CUDA out of memory, retrying with tile {tile}")
        except _NonFinite:
            if model.dtype == torch.float32:
                raise UpscaleError("Model produced NaN/Inf output even in fp32.") from None
            model.float()
            tqdm.write("fp16 produced NaN/Inf, falling back to fp32")
        # Exception frames are released here, so their tensors can actually be freed.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
