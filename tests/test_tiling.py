import numpy as np
import pytest
import torch

from imgup import models, tiling


def rand_img(h: int, w: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(int) - b.astype(int))


def test_tiled_matches_whole_image(model):
    img = rand_img(53, 71)  # odd sizes -> ragged edge tiles
    whole = tiling.run_tiles(img, model, tile=1000, pad=0)
    tiled = tiling.run_tiles(img, model, tile=16, pad=8)  # pad >= receptive-field radius
    assert tiled.shape == (106, 142, 3)
    assert diff(tiled, whole).max() <= 1  # fp rounding only
    # Sanity: without padding the seams are real, so the test above is meaningful.
    assert diff(tiling.run_tiles(img, model, tile=16, pad=0), whole).max() > 1


@pytest.mark.skipif(not (models.MODELS_DIR / "RealESRGAN_x4plus.pth").exists(), reason="weights not downloaded")
def test_real_model_default_pad_has_no_visible_seams():
    desc = models.load_model(models.MODELS_DIR / "RealESRGAN_x4plus.pth", models.pick_device(), half=False)
    yy, xx = np.mgrid[0:96, 0:96]
    img = np.stack([xx * 2, yy * 2, (xx + yy)], -1).astype(np.uint8)  # smooth gradient: seams would stand out
    whole = tiling.run_tiles(img, desc, tile=1000, pad=0)
    tiled = tiling.run_tiles(img, desc, tile=32, pad=16)
    d = diff(tiled, whole)
    # RRDB's receptive field exceeds 16 px, so tiles are close but not exact (measured: mean 0.72, max 7 of 255).
    assert d.mean() < 1 and d.max() <= 8


def test_oom_halves_tile_then_succeeds(monkeypatch, model):
    tiles = []

    def fake(img, m, tile, pad, progress):
        tiles.append(tile)
        if tile > 64:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return "ok"

    monkeypatch.setattr(tiling, "run_tiles", fake)
    assert tiling.upscale(rand_img(8, 8), model, tile=256) == "ok"
    assert tiles == [256, 128, 64]


def test_oom_gives_up_at_floor(monkeypatch, model):
    def fake(img, m, tile, pad, progress):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tiling, "run_tiles", fake)
    with pytest.raises(tiling.UpscaleError, match="Out of GPU memory even at tile size 32"):
        tiling.upscale(rand_img(8, 8), model, tile=256, min_tile=32)


def test_nan_falls_back_to_fp32(monkeypatch, model):
    model.half()
    real = tiling.run_tiles

    def fake(img, m, tile, pad, progress):
        if m.dtype == torch.float16:
            raise tiling._NonFinite
        return real(img, m, tile, pad, progress)

    monkeypatch.setattr(tiling, "run_tiles", fake)
    out = tiling.upscale(rand_img(8, 8), model)
    assert model.dtype == torch.float32 and out.shape == (16, 16, 3)
