import numpy as np
import torch
from PIL import Image
from typer.testing import CliRunner

from imgup.cli import app
from spandrel.architectures.Compact import Compact as SRVGGNetCompact


def tiny_weights(tmp_path):
    torch.manual_seed(0)
    weights = tmp_path / "tiny.pth"
    torch.save(SRVGGNetCompact(num_feat=8, num_conv=4, upscale=2).state_dict(), weights)
    return weights


def test_cli_batch_skip_and_scale(tmp_path):
    weights = tiny_weights(tmp_path)
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(src / "a.jpg")
    Image.fromarray(np.zeros((6, 6, 4), np.uint8)).save(src / "sub" / "b.png")
    out = tmp_path / "out"
    args = [str(src), "-o", str(out), "--model", str(weights), "-r"]

    r = CliRunner().invoke(app, [*args, "--scale", "1.5"])
    assert r.exit_code == 0, r.output
    assert Image.open(out / "a.jpg").size == (18, 15)
    assert Image.open(out / "sub" / "b.png").mode == "RGBA"

    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0 and r.output.count("exists") == 2

    r = CliRunner().invoke(app, [*args, "--scale", "3"])
    assert r.exit_code == 1 and "--scale must be in (0, 2]" in r.output

    r = CliRunner().invoke(app, [*args, "--overwrite", "--max-pixels", "100"])
    assert r.exit_code == 1 and "--max-pixels" in r.output


def test_cli_default_writes_next_to_input(tmp_path):
    weights = tiny_weights(tmp_path)
    src = tmp_path / "in"
    src.mkdir()
    Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(src / "a.jpg")

    r = CliRunner().invoke(app, [str(src / "a.jpg"), "--model", str(weights), "--scale", "1.5"])
    assert r.exit_code == 0, r.output
    assert Image.open(src / "a_x1.5.jpg").size == (18, 15)

    # Folder run at 2x: makes a_x2.jpg; a second run must not upscale a_x2.jpg into a_x2_x2.jpg.
    for _ in range(2):
        assert CliRunner().invoke(app, [str(src), "--model", str(weights)]).exit_code == 0
    assert sorted(p.name for p in src.iterdir()) == ["a.jpg", "a_x1.5.jpg", "a_x2.jpg"]


def test_cli_output_folder_above_input(tmp_path):
    trip = tmp_path / "Pictures" / "trip"
    trip.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(trip / "a.png")
    r = CliRunner().invoke(app, [str(trip), "-o", str(tmp_path / "Pictures"), "--model", str(tiny_weights(tmp_path))])
    assert r.exit_code == 0, r.output
    assert Image.open(tmp_path / "Pictures" / "a.png").size == (8, 8)


def test_cli_refuses_two_inputs_with_one_output(tmp_path):
    Image.new("RGB", (4, 4), "red").save(tmp_path / "pic.png")
    Image.new("RGB", (4, 4), "blue").save(tmp_path / "pic.jpg")
    r = CliRunner().invoke(app, [str(tmp_path), "-f", "webp", "--overwrite", "--model", str(tiny_weights(tmp_path))])
    assert r.exit_code == 1 and "would both be saved as" in r.output
    assert sorted(p.name for p in tmp_path.glob("pic*")) == ["pic.jpg", "pic.png", "pic_x2.webp"]


def test_cli_never_overwrites_the_input(tmp_path):
    Image.new("RGB", (4, 4), "red").save(tmp_path / "a.png")
    before = (tmp_path / "a.png").read_bytes()
    r = CliRunner().invoke(app, [str(tmp_path / "a.png"), "-o", str(tmp_path), "--overwrite", "--model", str(tiny_weights(tmp_path))])
    assert "would overwrite the input" in r.output
    assert (tmp_path / "a.png").read_bytes() == before


def test_cli_missing_model(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    r = CliRunner().invoke(app, [str(tmp_path / "a.png"), "--model", "nope"])
    assert r.exit_code == 1 and "Model 'nope' not found" in r.output and "Traceback" not in r.output


def test_dry_run_lists_recursive_actions_without_loading_or_writing(tmp_path, monkeypatch):
    from imgup import models

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not resolve model weights or select a device")

    monkeypatch.setattr(models, "resolve_model_path", forbidden)
    monkeypatch.setattr(models, "pick_device", forbidden)
    monkeypatch.setattr(models, "load_model", forbidden)
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    Image.new("RGB", (6, 4)).save(src / "a.jpg")
    Image.new("RGB", (3, 2)).save(src / "sub" / "b.png")
    out = tmp_path / "out"
    before = {path: path.read_bytes() for path in src.rglob("*") if path.is_file()}

    result = CliRunner().invoke(app, [str(src), "-r", "-o", str(out), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert f"{src / 'a.jpg'} -> {out / 'a.jpg'} (24x16): would write" in result.output
    assert f"{src / 'sub/b.png'} -> {out / 'sub/b.png'} (12x8): would write" in result.output
    assert not out.exists()
    assert {path: path.read_bytes() for path in src.rglob("*") if path.is_file()} == before


def test_dry_run_reports_existing_output_collision_and_same_input(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    for name in ("a.jpg", "a.png"):
        Image.new("RGB", (6, 4)).save(src / name)
    Image.new("RGB", (1, 1)).save(out / "a.webp")
    before = (out / "a.webp").read_bytes()
    runner = CliRunner()
    result = runner.invoke(app, [str(src), "-o", str(out), "-f", "webp", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "skip: output exists" in result.output
    assert "skip: same output as" in result.output
    assert (out / "a.webp").read_bytes() == before

    image = src / "a.png"
    result = runner.invoke(app, [str(image), "-o", str(src), "--overwrite", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "skip: output would overwrite the input" in result.output
    assert Image.open(image).size == (6, 4)


def test_dry_run_custom_weights_need_explicit_scale_and_honor_exif(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (6, 4)).save(path, exif=exif)
    runner = CliRunner()
    args = [str(path), "--model", "uninstalled-custom.pth", "--dry-run"]
    result = runner.invoke(app, args)
    assert result.exit_code != 0 and "--scale" in result.output
    result = runner.invoke(app, [*args, "--scale", "2"])
    assert result.exit_code == 0, result.output
    assert "(8x12): would write" in result.output
    assert list(tmp_path.iterdir()) == [path]


def test_dry_run_reads_png_header_without_decoding(tmp_path, monkeypatch):
    from PIL import PngImagePlugin

    path = tmp_path / "frame.png"
    Image.new("RGB", (6, 4)).save(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not decode pixels")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", forbidden)
    result = CliRunner().invoke(app, [str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "(24x16): would write" in result.output
