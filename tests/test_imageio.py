import numpy as np
import pytest
from PIL import Image

from imgup import imageio


def test_target_size_math():
    assert imageio.target_size(100, 50, 4, 4) == (400, 200)
    assert imageio.target_size(100, 50, 2, 4) == (200, 100)
    assert imageio.target_size(33, 10, 1.5, 4) == (50, 15)
    assert imageio.target_size(1, 1, 0.1, 4) == (1, 1)
    for bad in (0, -1, 4.01):
        with pytest.raises(ValueError):
            imageio.target_size(10, 10, bad, 4)


def test_alpha_roundtrip(tmp_path):
    rgba = np.zeros((10, 20, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = np.linspace(0, 255, 20, dtype=np.uint8)  # alpha ramp left -> right
    src = tmp_path / "a.png"
    Image.fromarray(rgba).save(src, icc_profile=imageio.SRGB_ICC)

    img = imageio.load(src)
    assert img.rgb.shape == (10, 20, 3) and img.alpha is not None and img.icc == imageio.SRGB_ICC

    native = np.repeat(np.repeat(img.rgb, 4, 0), 4, 1)  # stand-in for a 4x model
    rgb, alpha = imageio.finalize(native, img.alpha, (40, 20))
    imageio.save(rgb, alpha, tmp_path / "o.png", "PNG", 95, img.icc, img.exif)

    with Image.open(tmp_path / "o.png") as out:
        assert out.mode == "RGBA" and out.size == (40, 20)
        assert out.info.get("icc_profile") == imageio.SRGB_ICC
        a = np.asarray(out.getchannel("A")).astype(int)
    expected = np.asarray(img.alpha.resize((40, 20), Image.Resampling.LANCZOS)).astype(int)
    assert np.abs(a - expected).max() == 0
    assert a[:, 0].max() < 20 and a[:, -1].min() > 235


def test_alpha_to_jpeg_flattens_onto_white(tmp_path):
    rgba = np.zeros((8, 8, 4), np.uint8)  # fully transparent black
    Image.fromarray(rgba).save(tmp_path / "t.png")
    img = imageio.load(tmp_path / "t.png")
    rgb, alpha = imageio.finalize(img.rgb, img.alpha, (8, 8))
    imageio.save(rgb, alpha, tmp_path / "t.jpg", "JPEG", 95, None, None)
    with Image.open(tmp_path / "t.jpg") as out:
        assert out.mode == "RGB" and np.asarray(out).min() > 250


def test_exif_orientation_applied_and_cleared(tmp_path):
    arr = np.zeros((20, 30, 3), np.uint8)
    arr[:, :15] = (255, 0, 0)  # left half red, right half blue
    arr[:, 15:] = (0, 0, 255)
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: display = rotate 90 degrees clockwise
    exif[0x010F] = "TestCam"  # Make, should survive
    Image.fromarray(arr).save(tmp_path / "r.jpg", exif=exif.tobytes(), quality=100)

    img = imageio.load(tmp_path / "r.jpg")
    assert img.rgb.shape == (30, 20, 3)  # width/height swapped
    assert img.rgb[2, 10, 0] > 200 and img.rgb[-3, 10, 2] > 200  # red on top, blue at bottom after CW rotation

    rgb, _ = imageio.finalize(img.rgb, None, (20, 30))
    imageio.save(rgb, None, tmp_path / "o.jpg", "JPEG", 95, img.icc, img.exif)
    with Image.open(tmp_path / "o.jpg") as out:
        e = out.getexif()
        assert out.size == (20, 30) and e.get(0x0112, 1) == 1 and e.get(0x010F) == "TestCam"


@pytest.mark.parametrize(
    "make, expect",
    [
        (lambda: Image.new("CMYK", (4, 4), (0, 255, 255, 0)), (255, 0, 0)),  # CMYK red
        (lambda: Image.new("P", (4, 4), 0), (0, 0, 0)),
        (lambda: Image.new("L", (4, 4), 128), (128, 128, 128)),
        (lambda: Image.new("1", (4, 4), 1), (255, 255, 255)),
        (lambda: Image.fromarray(np.full((4, 4), 65535, np.uint16)), (255, 255, 255)),  # 16-bit gray
        (lambda: Image.fromarray(np.full((4, 4), 32896, np.uint16)), (128, 128, 128)),
    ],
)
def test_mode_conversion(tmp_path, make, expect):
    im = make()
    path = tmp_path / ("x.tif" if im.mode == "CMYK" else "x.png")
    im.save(path)
    img = imageio.load(path)
    assert img.rgb.dtype == np.uint8 and img.rgb.shape == (4, 4, 3) and img.alpha is None
    assert tuple(img.rgb[0, 0]) == expect


def test_palette_transparency_becomes_alpha(tmp_path):
    im = Image.new("P", (4, 4), 0)
    im.putpalette([0, 0, 0, 255, 255, 255])
    im.save(tmp_path / "p.png", transparency=0)
    assert imageio.load(tmp_path / "p.png").alpha is not None


def test_unreadable_image(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(imageio.ImageError):
        imageio.load(tmp_path / "bad.png")
