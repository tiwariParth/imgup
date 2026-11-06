# imgup: a local image upscaler

> **Note: this is a vibe-coded project.** It was built by describing what I wanted to an AI
> coding assistant (Claude Code) and testing the result, not by writing every line by hand.
> It works on my machine and has tests, but read the code before trusting it with anything
> important. Fixes and improvements are very welcome; see [Working on the code](#working-on-the-code).

`imgup` makes small or blurry images bigger and sharper using AI models that run **on your
own computer**. Nothing is uploaded anywhere, there's no account, and it costs nothing.

You give it a picture and it makes a 2x, 3x or 4x larger version next to the original:

```
holiday.jpg  (1000 x 750)   ->   holiday_x4.jpg  (4000 x 3000)
```

It's a command-line tool: you run it by typing a command in a terminal. If you've never
done that before, the steps below walk you through it.

---

## What you need

| | |
|---|---|
| **Computer** | Linux or Windows. Tested only on Fedora Linux. macOS is **not** supported as-is (see [FAQ](#faq)). |
| **Graphics card (recommended)** | An NVIDIA GPU. 4 GB of video memory is enough. Without one it still works, just much slower. |
| **NVIDIA driver** | Version 580 or newer. Check by running `nvidia-smi` in a terminal. |
| **Disk space** | About 6 GB (mostly the PyTorch AI library). |
| **Internet** | Only for the one-time setup. After that it works offline. |

---

## Setup (one time)

**1. Install `uv`**, a tool that installs the right Python version and all dependencies for you.

Linux:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Then close and reopen your terminal.

**2. Download this project:**
```sh
git clone https://github.com/tiwariParth/imgup.git
cd imgup
```
(No git? Click the green **Code** button on GitHub → **Download ZIP**, unzip it, and open a
terminal in that folder.)

**3. Install everything.** The first run downloads about 4 GB, so give it a while:
```sh
uv sync
```

**4. Download the AI models** (about 85 MB, from the official Real-ESRGAN releases):
```sh
uv run python scripts/download_models.py
```
You should see `OK` for both models.

You're done with setup.

---

## How to use it

Run these from inside the `imgup` folder.

**Upscale one image (4x bigger):**
```sh
uv run imgup path/to/photo.jpg
```
This creates `photo_x4.jpg` in the same folder as the original. **Your original is never changed.**

**Make it only 2x bigger:**
```sh
uv run imgup path/to/photo.jpg --scale 2
```

**For drawings, anime or cartoons**, use the anime model, which keeps lines cleaner:
```sh
uv run imgup path/to/drawing.png --model anime
```

**Upscale a whole folder** (add `-r` to include subfolders):
```sh
uv run imgup path/to/folder -r
```

**Put the results in a separate folder** instead of next to the originals:
```sh
uv run imgup path/to/folder -o upscaled/
```

**Save in a different format:**
```sh
uv run imgup photo.png --format jpg --quality 90
```

If you run the same command again, images that already have a result are skipped. Add
`--overwrite` to redo them.

### Which model should I use?

| Model | Use for | Flag |
|---|---|---|
| Real-ESRGAN x4plus (default) | Photos and real-world images | `--model x4plus` |
| Real-ESRGAN x4plus anime 6B | Drawings, anime, cartoons, illustrations | `--model anime` |
| Any other model | Downloaded `.pth` or `.safetensors` file | `--model path/to/file.pth` |

Many architectures work (ESRGAN, SwinIR, HAT, DAT and more) thanks to the
[spandrel](https://github.com/chaiNNer-org/spandrel) library. You can find community models
at [OpenModelDB](https://openmodeldb.info/).

### All options

Run `uv run imgup --help` to see these in the terminal.

| Option | Default | What it does |
|---|---|---|
| `-o, --output` | next to the input | Folder to save results in. If not given, saves `name_x4.ext` beside each original. |
| `-m, --model` | `x4plus` | Which model: `x4plus`, `anime`, or a path to a model file. |
| `-s, --scale` | model's maximum (4) | How much bigger: any number up to the model's maximum, e.g. `2`, `2.5`, `3`. |
| `-f, --format` | same as input | Output format: `png`, `jpg` or `webp`. |
| `-q, --quality` | `95` | JPEG/WebP quality, 1-100. Higher means better quality and larger files. |
| `-r, --recursive` | off | Also process images in subfolders. |
| `--overwrite` | off | Redo images that already have a result. |
| `-t, --tile` | `256` | Size of the pieces the image is processed in. Lower it if you run out of GPU memory. |
| `--pad` | `16` | Overlap between pieces, so no lines show where they join. |
| `--max-pixels` | `256000000` | Safety limit: refuses results bigger than this many pixels (to protect your RAM). |
| `--fp32` | off | Use full precision. Slower; only needed if results look broken. |

---

## Good to know

- **Transparency is kept.** PNG and WebP images with transparent backgrounds stay transparent.
  (JPEG can't store transparency, so converting to JPEG fills it with white.)
- **Photo info is kept.** Colour profiles and camera info (EXIF) are copied over. Photos taken
  sideways on a phone are rotated correctly.
- **Big images take time and memory.** On a laptop RTX 3050, a 1000 x 1000 image takes about
  15 seconds at 4x. A 4000 x 4000 image takes about 5 minutes and needs about 4 GB of RAM,
  because the result is 16000 x 16000.
- **Upscaling makes up detail.** The AI guesses what finer detail probably looked like. Results
  usually look great, but they aren't a true recovery of the original, so don't use them as evidence.

### If you run out of GPU memory

The tool already handles this: it automatically retries with smaller pieces. If it still
fails, close other programs using the GPU (games, browsers with many tabs), or start smaller:
```sh
uv run imgup photo.jpg --tile 128
```

Measured on an RTX 3050 6 GB laptop GPU (1000 x 1000 image → 4000 x 4000):

| `--tile` | GPU memory used | Time (photo model) | Time (anime model) |
|---|---|---|---|
| 128 | 0.7 GB | 17 s | 9 s |
| 256 (default) | 1.8 GB | 14 s | 7 s |
| 384 | 3.6 GB | 14 s | 7 s |
| 512 | 3.3 GB | 13 s | 7 s |
| 768 | 4.9 GB | 13 s | 7 s |

Bigger tiles barely make it faster, so the default is a good choice for most cards.

---

## FAQ

**It says "no CUDA GPU available, running on CPU".**
It will still work, just slowly. If you do have an NVIDIA card, check that `nvidia-smi` works and
that your driver is version 580 or newer.

**My NVIDIA driver is older than 580.**
Open `pyproject.toml`, change `cu130` to `cu126` in both places, then run `uv sync` again.

**Does it work on a Mac?**
Not as-is. `pyproject.toml` pulls NVIDIA-specific PyTorch builds, which don't exist for macOS.
Removing the `[tool.uv.sources]` and `[[tool.uv.index]]` sections should make it install
normal PyTorch, which would then run on the CPU. This is untested; a pull request adding
Apple GPU (MPS) support would be welcome.

**Does it work with AMD or Intel GPUs?**
Not tested. It will fall back to CPU.

**Can I install it with `pip install` or `uv tool install` instead of cloning?**
Not yet. The tool looks for models in the `models/` folder of the cloned project, so after a
package install `--model x4plus` says "not found", and the download script isn't included.
Use the clone + `uv run` steps above. If you do install it as a package, pass the model file
directly: `imgup photo.jpg --model /path/to/RealESRGAN_x4plus.pth`.

**Is anything sent to the internet?**
No. Only the setup steps download things. Upscaling happens entirely on your computer.

---

## Working on the code

You're welcome to fix bugs, add features or adapt it for yourself. Here's everything you need.

### Get a development copy

```sh
git clone https://github.com/tiwariParth/imgup.git
cd imgup
uv sync                                    # installs the app plus pytest
uv run python scripts/download_models.py   # needed for one of the tests
uv run pytest                              # all tests should pass
```

`uv sync` uses `uv.lock`, so you get the exact same library versions I tested with.

### How it works

1. **Load the image** ([src/imgup/imageio.py](src/imgup/imageio.py)): rotate it per its EXIF
   tag, convert unusual formats (CMYK, 16-bit, palette, grayscale) to plain RGB, and set aside
   the transparency (alpha) channel and the colour profile.
2. **Load the model** ([src/imgup/models.py](src/imgup/models.py)): spandrel's `ModelLoader`
   reads the weights file and works out the architecture by itself. It uses half precision
   (fp16) on the GPU to save memory.
3. **Upscale in tiles** ([src/imgup/tiling.py](src/imgup/tiling.py)): a whole photo won't fit
   in GPU memory, so the image is cut into 256 px tiles. Each tile is processed with 16 px of
   extra surrounding context, and only its centre is kept, which hides the joins. Results go
   straight into an 8-bit buffer to keep RAM use low. If the GPU runs out of memory, the tile
   size is halved and the image retried. If fp16 produces broken values (NaN), it switches to fp32.
4. **Resize and save** ([src/imgup/imageio.py](src/imgup/imageio.py)): the model always
   upscales at its native size (4x). For `--scale 2`, the result is then shrunk with Lanczos
   filtering. The transparency channel is resized the same way and put back. Metadata is
   reattached, and the file is written to a temporary name first, so a crash never leaves a
   half-written image.
5. **Command line** ([src/imgup/cli.py](src/imgup/cli.py)): options, finding files, skipping
   finished ones, and readable error messages (built with [typer](https://typer.tiangolo.com/)).

### Project layout

```
src/imgup/
  cli.py        command-line options and the main loop
  models.py     finding and loading model files
  tiling.py     tiled GPU inference, out-of-memory and NaN recovery
  imageio.py    reading/writing images, alpha, EXIF, colour profiles, resizing
scripts/
  download_models.py   fetches the default models and checks they load
tests/                 pytest tests; they generate small images on the fly
models/                downloaded model files (not committed to git)
```

### Tests

```sh
uv run pytest
```

The tests check that:
- tiled output matches processing the whole image at once (no seams)
- transparency survives and is resized correctly
- EXIF rotation is applied and camera info is kept
- `--scale` sizes are calculated correctly
- the out-of-memory retry and NaN fallback work (simulated)
- CMYK, palette, grayscale and 16-bit images convert correctly
- the CLI skips finished files and gives clean errors

The tests make their own tiny images, so no large files are needed. One test uses the real
x4plus model and is skipped if you haven't downloaded it.

### Known limitations (good first contributions)

- Animated GIF/WebP: only the first frame is upscaled.
- Output is always 8-bit, even for 16-bit inputs.
- Grayscale colour profiles are dropped rather than converted.
- After one image falls back to fp32, the rest of that batch stays in fp32.
- Tiles aren't pixel-identical to whole-image processing: they differ by about 0.7 out of 255
  on average, which is invisible. `--pad 32` reduces it further.
- Inputs over about 179 megapixels are refused by Pillow's safety check.
- Only tested on Fedora Linux with an RTX 3050.

### How this project was made

It was built with Claude Code from a written brief that asked for:
- a Python 3.12 CLI managed with `uv`, using `torch`, `spandrel`, `pillow`, `typer` and `tqdm`
- any spandrel-supported model, chosen by name or file path
- tiled inference with overlap (defaults: tile 256, pad 16) for low-VRAM GPUs
- fp16 with fp32 fallback, and automatic tile halving on out-of-memory
- `--scale` up to the model's native scale, with Lanczos downsampling
- correct handling of alpha, EXIF orientation, ICC profiles, CMYK/16-bit/palette images
- a memory guard (`--max-pixels`), clean error messages, and CPU fallback
- pytest tests, plus real runs measuring output size and peak VRAM

The full original brief, and the decisions made along the way, are in [PROMPT.md](PROMPT.md).
To rebuild or extend it the same way, give that file and the code to an AI coding assistant
and ask it to keep the tests passing. Either way, run `uv run pytest`
and try a real image before opening a pull request.

---

## Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al. for the default
  models. The model files are downloaded from their official releases and are not part of this repo.
- [spandrel](https://github.com/chaiNNer-org/spandrel) for loading many model architectures.
- [PyTorch](https://pytorch.org/), [Pillow](https://python-pillow.org/),
  [Typer](https://typer.tiangolo.com/), [tqdm](https://github.com/tqdm/tqdm).

## License

The code is under the [MIT License](LICENSE): use it, change it, share it, including
commercially. Just keep the license notice. The Real-ESRGAN model files have their own
[BSD-3-Clause license](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
