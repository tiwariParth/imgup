"""imgup command-line interface."""

import re
import sys
from enum import Enum
from pathlib import Path

import typer
from spandrel import ImageModelDescriptor
from tqdm import tqdm

from imgup import imageio, models, tiling

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


class Fmt(str, Enum):
    png = "png"
    jpg = "jpg"
    webp = "webp"


def collect(src: Path, out_dir: Path | None, recursive: bool) -> list[Path]:
    if src.is_file():
        return [src]
    files = [p for p in (src.rglob("*") if recursive else src.glob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    # Skip previous results so re-runs don't upscale them again: either inside the output
    # folder, or name_x<scale>.* sitting next to its original name.*.
    if out_dir:
        out = out_dir.resolve()
        if src.resolve() in out.parents:  # only an output folder inside the input folder can hold old results
            files = [p for p in files if out not in p.resolve().parents]
        return sorted(files)
    stems = {(p.parent, p.stem) for p in files}
    return sorted(p for p in files if not ((m := re.fullmatch(r"(.+)_x[\d.]+", p.stem)) and (p.parent, m[1]) in stems))


def dest_for(path: Path, src: Path, out_dir: Path | None, fmt: Fmt | None, scale: float) -> Path:
    suffix = imageio.EXTENSIONS[imageio.FORMATS[fmt.value]] if fmt else path.suffix
    if out_dir is None:
        return path.with_name(f"{path.stem}_x{scale:g}{suffix}")
    rel = Path(path.name) if src.is_file() else path.relative_to(src)
    return out_dir / rel.with_suffix(suffix)


def process(path: Path, dest: Path, model: ImageModelDescriptor, scale: float | None, tile: int, pad: int,
            fmt: Fmt | None, quality: int, max_pixels: int, progress: bool) -> tuple[int, int]:
    img = imageio.load(path)
    h, w = img.rgb.shape[:2]
    s = model.scale
    size = imageio.target_size(w, h, scale or s)
    if w * h * s * s > max_pixels:
        raise tiling.UpscaleError(
            f"{w}x{h} at native x{s} is {w * s}x{h * s} ({w * h * s * s / 1e6:.0f} MP), "
            f"over --max-pixels {max_pixels / 1e6:.0f} MP. Raise --max-pixels if you have the RAM."
        )
    native = tiling.upscale(img.rgb, model, tile, pad, progress)
    rgb, alpha = imageio.finalize(native, img.alpha, size)
    del native
    dest.parent.mkdir(parents=True, exist_ok=True)
    imageio.save(rgb, alpha, dest, imageio.FORMATS[fmt.value] if fmt else img.format, quality, img.icc, img.exif)
    return size


def report(msg: str) -> None:
    """Print a per-file error without breaking the batch progress bar."""
    with tqdm.external_write_mode(file=sys.stderr):
        typer.secho(f"error: {msg}", fg="red", err=True)


def fail(msg: str) -> None:
    report(msg)
    raise typer.Exit(1)


@app.command()
def main(
    input: Path = typer.Argument(..., help="Image file or folder of images."),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output folder. Default: next to each input as name_x<scale>.ext."),
    model: str = typer.Option("x4plus", "-m", "--model", help="Alias (x4plus, anime), name in models/, or path to .pth/.safetensors."),
    scale: float | None = typer.Option(None, "-s", "--scale", help="Output scale, up to the model's native scale. Default: native."),
    tile: int = typer.Option(256, "-t", "--tile", min=tiling.MIN_TILE, help="Tile size in input pixels. Lower it if you run out of VRAM."),
    pad: int = typer.Option(16, "--pad", min=0, help="Overlap context around each tile, in input pixels."),
    fmt: Fmt | None = typer.Option(None, "-f", "--format", help="Output format. Default: same as input."),
    quality: int = typer.Option(95, "-q", "--quality", min=1, max=100, help="JPEG/WebP quality."),
    recursive: bool = typer.Option(False, "-r", "--recursive", help="Recurse into subfolders."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-process files whose output already exists."),
    max_pixels: int = typer.Option(256_000_000, "--max-pixels", help="Refuse images whose native-scale output exceeds this many pixels."),
    fp32: bool = typer.Option(False, "--fp32", help="Disable fp16 inference."),
) -> None:
    """Upscale images locally with a pretrained super-resolution model."""
    if not input.exists():
        fail(f"{input} does not exist")
    files = collect(input, output, recursive)
    if not files:
        fail(f"no images found in {input}" + ("" if recursive else " (use -r for subfolders)"))

    try:
        weights = models.resolve_model_path(model)
        device = models.pick_device()
        if device.type == "cpu":
            typer.secho("warning: no CUDA GPU available, running on CPU (slow)", fg="yellow", err=True)
        desc = models.load_model(weights, device, half=not fp32)
    except models.ModelError as e:
        fail(str(e))
    if scale is not None and not 0 < scale <= desc.scale:
        fail(f"--scale must be in (0, {desc.scale}] for {weights.name}, got {scale}")

    jobs = [(p, dest_for(p, input, output, fmt, scale or desc.scale)) for p in files]
    single = len(jobs) == 1
    failures = 0
    claimed: dict[Path, Path] = {}
    for path, dest in tqdm(jobs, disable=single, unit="img"):
        if dest.resolve() == path.resolve():
            tqdm.write(f"skip {path}: output would overwrite the input")
            continue
        if (other := claimed.setdefault(dest.resolve(), path)) != path:
            failures += 1
            report(f"{path} and {other} would both be saved as {dest}; rename one of them")
            continue
        if dest.exists() and not overwrite:
            tqdm.write(f"skip {path}: {dest} exists (use --overwrite)")
            continue
        try:
            w, h = process(path, dest, desc, scale, tile, pad, fmt, quality, max_pixels, progress=single)
            tqdm.write(f"{path} -> {dest} ({w}x{h})")
        except (imageio.ImageError, tiling.UpscaleError, OSError) as e:
            failures += 1
            report(f"{path}: {e}")
    if failures:
        raise typer.Exit(1)
