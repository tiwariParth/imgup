import numpy as np
import torch
from PIL import Image
from typer.testing import CliRunner

from imgup.cli import app
from spandrel.architectures.Compact import Compact as SRVGGNetCompact


def test_cli_batch_skip_and_scale(tmp_path):
    torch.manual_seed(0)
    weights = tmp_path / "tiny.pth"
    torch.save(SRVGGNetCompact(num_feat=8, num_conv=4, upscale=2).state_dict(), weights)
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
    torch.manual_seed(0)
    weights = tmp_path / "tiny.pth"
    torch.save(SRVGGNetCompact(num_feat=8, num_conv=4, upscale=2).state_dict(), weights)
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


def test_cli_missing_model(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    r = CliRunner().invoke(app, [str(tmp_path / "a.png"), "--model", "nope"])
    assert r.exit_code == 1 and "Model 'nope' not found" in r.output and "Traceback" not in r.output
