# The prompt this project was built from

This project was vibe-coded with Claude Code. Below is the exact brief it was given, followed
by the decisions made during the build. To rebuild or extend it the same way, give this file
(and the existing code) to an AI coding assistant.

## Original brief

Build a personal-use image upscaler CLI in this directory. Read the whole brief first, then make a short plan, then implement.

### Context
- Machine: Fedora 44 (GNOME), Acer Nitro V laptop with an RTX 3050 (assume 4 to 6 GB VRAM).
- Goal: a scriptable CLI to upscale my own images locally. No training, only pretrained models. No cloud APIs.
- Check `nvidia-smi` to confirm the driver works. If it doesn't, stop and tell me what's missing. Do NOT install or change system drivers or run sudo commands without asking me first.

### Stack
- Python 3.12 managed with `uv` (pyproject.toml, no requirements.txt).
- `torch` + `torchvision` (CUDA wheel), `spandrel` (model loading), `pillow`, `typer`, `tqdm`. Add `opencv-python-headless` only if you actually use it.
- Use spandrel's `ModelLoader` so any supported architecture (ESRGAN, SwinIR, HAT, etc.) loads from a weights file. Do not hardcode a single architecture.

### Models
- Put weights in `models/` (gitignored).
- Write a `scripts/download_models.py` that fetches `RealESRGAN_x4plus.pth` (photos) and `RealESRGAN_x4plus_anime_6B.pth` (illustrations) from their official release pages. Do not guess URLs: look them up, and tell me if a download fails. Verify the file loads with spandrel after download.
- The model is selected with a `--model` flag that accepts a name or a path to any .pth/.safetensors file.

### Features
1. Tiled inference with overlap padding so it works on low VRAM. Defaults: tile 256, pad 16, both configurable. No visible seams.
2. fp16 by default with automatic fallback to fp32 if output contains NaNs. On CUDA OOM, halve the tile size and retry (down to a floor), then fail with a clear message.
3. CLI via typer, e.g.:
   `imgup photo.jpg -o out/ --model x4plus --scale 2 --tile 256`
   - Accepts a single file or a folder (batch mode, recursive optional).
   - Skips files whose output already exists unless `--overwrite`.
   - tqdm progress bar (per tile for single files, per image for batches).
4. `--scale` can be any value up to the model's native scale: run the model at native scale, then downsample with Lanczos to the target. Default output scale equals the model's native scale.
5. Image handling edge cases:
   - PNG/WebP with alpha: upscale RGB with the model, resize alpha separately with Lanczos, recombine.
   - Preserve ICC profile and EXIF where the output format supports it.
   - Apply EXIF orientation before processing.
   - Convert CMYK, palette, grayscale, and 16-bit images to RGB safely.
   - Output format defaults to the input format. Add `--format png|jpg|webp` and `--quality`.
6. Memory guards: write results into a uint8 buffer tile by tile rather than a full fp32 output tensor. Add a `--max-pixels` safety check that refuses absurd outputs with a clear message.
7. Clean error messages (no raw tracebacks for expected failures such as missing model, unreadable image, no GPU). If no GPU is available, fall back to CPU with a warning.

### Structure
- `src/imgup/` package with separate modules: `models.py` (loading), `tiling.py` (core inference), `imageio.py` (loading/saving/alpha/ICC), `cli.py`.
- Keep functions small and typed. No unnecessary abstractions.
- `README.md` with setup, usage examples, and the VRAM/tile tuning table.

### Testing and verification
- Add pytest tests for: tiling seam correctness (tiled output vs. whole-image output on a small image should match within a small tolerance), alpha handling, EXIF orientation, scale flag math, and the OOM retry logic (mock the OOM).
- Generate small synthetic test images in the tests, don't commit large binaries.
- Then actually run the CLI on at least one real image and one PNG with transparency, report the output sizes and peak VRAM, and confirm the output opens correctly.
- Do not claim it works until you have run it. If something fails, show me the error and fix it.

### Constraints
- Do not add a web UI, Docker, or extra features beyond this brief. I may add Gradio later.
- Do not commit model weights or test outputs.
- Ask me before making any decision that isn't covered here and that changes behavior (not for naming or style).
- When done, give me a short summary: what was built, how to run it, what you verified, and any known limitations.

## Decisions made during the build

- **Transparent image saved as JPEG:** flattened onto a white background, with a warning.
- **Default output location:** changed after the first version. Without `-o`, results are saved
  next to the input as `name_x<scale>.ext` (e.g. `photo_x4.jpg`); the original is never
  overwritten. With `-o folder/`, results go into that folder under their original filenames.
- **Re-running on a folder:** files named `name_x<scale>.*` that sit next to their original
  `name.*` are treated as previous results and skipped.
- **PyTorch build:** CUDA 13.0 wheels (`cu130`), which need NVIDIA driver 580 or newer.
- **Real-model seam test tolerance:** Real-ESRGAN's receptive field is larger than the 16 px
  pad, so tiled output differs from whole-image output by about 0.7/255 on average (max 7).
  That isn't visible; the test allows mean < 1 and max <= 8.
- **License:** MIT.
