#!/usr/bin/env python
"""Compare MAT, OmniEraser, and land-cover fill on the Fort Riley test tile.

Writes named GeoTIFFs + JPEG previews under outputs/ground_fill/Fort_Riley/testing.
Does not overwrite objects/bare or tree/orchard/roof checkpoints.
"""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "ground_fill_compare.log"
OUT = ROOT / "outputs" / "ground_fill" / "Fort_Riley" / "testing"
ORIG = ROOT / "data" / "Fort_Riley" / "Imagery" / "FortRiley_r81920_c61440.tif"
MASK = ROOT / "outputs" / "ground_fill" / "Fort_Riley" / "objects" / "FortRiley_r81920_c61440_objects.tif"
PM = ROOT / "outputs" / "ground_fill" / "Fort_Riley" / "bare" / "FortRiley_r81920_c61440_bare.tif"
OMNI_SRC = ROOT / "outputs" / "checkpoints" / "ground_fill" / "omnieraser_src"
HF_CACHE = ROOT / "outputs" / "checkpoints" / "ground_fill" / "hf"

CROPS = [
    ("dense", 4096, 4096, 1280, 4300, 4480, 640),
    ("mixed", 7680, 8192, 1280, 7900, 8500, 640),
]


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_win(path: Path, col: int, row: int, n: int) -> np.ndarray:
    with rasterio.open(path) as ds:
        return np.transpose(ds.read(window=Window(col, row, n, n))[:3], (1, 2, 0))


def write_rgb_geotiff(path: Path, rgb: np.ndarray, transform, crs) -> None:
    from tree_seg.io_geotiff import write_geotiff

    write_geotiff(path, np.transpose(rgb, (2, 0, 1)), transform, crs, dtype="uint8")


def write_window_geotiff(path: Path, rgb: np.ndarray, col: int, row: int) -> None:
    from rasterio.windows import transform as window_transform

    with rasterio.open(ORIG) as ds:
        win = Window(col, row, rgb.shape[1], rgb.shape[0])
        write_rgb_geotiff(path, rgb, window_transform(win, ds.transform), ds.crs)


def label_bar(width: int, text: str) -> Image.Image:
    bar = Image.new("RGB", (width, 32), (18, 18, 18))
    ImageDraw.Draw(bar).text((8, 8), text, fill=(235, 235, 235))
    return bar


def hstack_labeled(images: list[np.ndarray], titles: list[str]) -> Image.Image:
    ims = [Image.fromarray(a) for a in images]
    w, h = ims[0].size
    canvas = Image.new("RGB", (w * len(ims), h + 32), (0, 0, 0))
    for i, (im, title) in enumerate(zip(ims, titles)):
        canvas.paste(label_bar(w, title), (i * w, 0))
        canvas.paste(im, (i * w, 32))
    return canvas


def save_previews(paths: dict[str, Path]) -> None:
    rows = []
    zooms = []
    titles = ["original", "MAT", "OmniEraser", "landcover"]
    for name, r, c, n, zr, zc, zn in CROPS:
        orig = read_win(ORIG, c, r, n)
        mat = read_win(paths["mat"], c, r, n)
        land = read_win(paths["landcover"], c, r, n)
        omni_p = paths.get(f"omni_{name}")
        if omni_p and omni_p.is_file():
            with rasterio.open(omni_p) as ds:
                omni = np.transpose(ds.read()[:3], (1, 2, 0))
            if omni.shape[0] != n or omni.shape[1] != n:
                omni = np.array(Image.fromarray(omni).resize((n, n), Image.Resampling.BILINEAR))
        else:
            omni = np.zeros_like(orig)
        rows.append(hstack_labeled([orig, mat, omni, land], [f"{name} {t}" for t in titles]))
        zo = read_win(ORIG, zc, zr, zn)
        zm = read_win(paths["mat"], zc, zr, zn)
        zl = read_win(paths["landcover"], zc, zr, zn)
        if omni_p and omni_p.is_file():
            # zoom is offset inside the 1280 crop
            dy, dx = zr - r, zc - c
            zz = omni[dy : dy + zn, dx : dx + zn] if omni.shape[0] >= dy + zn else omni[:zn, :zn]
            if zz.shape[0] != zn or zz.shape[1] != zn:
                zz = np.array(Image.fromarray(omni).resize((n, n), Image.Resampling.BILINEAR))[
                    dy : dy + zn, dx : dx + zn
                ]
        else:
            zz = np.zeros_like(zo)
        zooms.append(hstack_labeled([zo, zm, zz, zl], [f"{name} zoom {t}" for t in titles]))

    def vstack(imgs: list[Image.Image]) -> Image.Image:
        w = max(i.width for i in imgs)
        hh = sum(i.height for i in imgs)
        out = Image.new("RGB", (w, hh), (0, 0, 0))
        y = 0
        for i in imgs:
            out.paste(i, (0, y))
            y += i.height
        return out

    overview = vstack(rows)
    overview.resize((overview.width // 2, overview.height // 2), Image.Resampling.BILINEAR).save(
        OUT / "_preview.jpg", quality=90
    )
    vstack(zooms).save(OUT / "_preview_zoom.jpg", quality=92)
    log("wrote JPEG previews")


def ensure_omnieraser_src() -> Path:
    if (OMNI_SRC / "pipeline_flux_control_removal.py").is_file():
        return OMNI_SRC
    OMNI_SRC.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    log(f"clone OmniEraser into {OMNI_SRC}")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "https://github.com/PRIS-CV/Omnieraser.git", str(OMNI_SRC)]
    )
    return OMNI_SRC


def load_omnieraser_pipe():
    """OmniEraser ControlNet release (public weights). Base LoRA is still gated."""
    import torch

    src = OMNI_SRC / "ControlNet_version"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from controlnet_flux import FluxControlNetModel
    from transformer_flux import FluxTransformer2DModel
    from pipeline_flux_controlnet_removal import FluxControlNetInpaintingPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    cache = str(HF_CACHE)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kw = dict(torch_dtype=dtype, cache_dir=cache, token=token)
    log("load Alimama FLUX inpaint ControlNet")
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        **kw,
    )
    log("load FLUX.1-dev transformer")
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        **kw,
    )
    log("build OmniEraser ControlNet pipeline")
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        **kw,
    )
    pipe.load_lora_weights(
        "theSure/Omnieraser_Controlnet_version",
        weight_name="controlnet_flux_pytorch_lora_weights.safetensors",
        token=token,
    )
    pipe.transformer.to(dtype)
    pipe.controlnet.to(dtype)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    else:
        pipe.to("cpu")
    pipe.set_progress_bar_config(disable=False)
    return pipe


def run_omnieraser(pipe, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    h, w = rgb.shape[:2]
    side = 1024
    img = Image.fromarray(rgb).convert("RGB").resize((side, side), Image.Resampling.BILINEAR)
    msk = Image.fromarray((mask > 0).astype(np.uint8) * 255).convert("RGB").resize(
        (side, side), Image.Resampling.NEAREST
    )
    gen = torch.Generator(device="cpu").manual_seed(24)
    result = pipe(
        prompt="There is nothing here.",
        control_image=img,
        control_mask=msk,
        num_inference_steps=20,
        guidance_scale=3.5,
        true_guidance_scale=1.0,
        controlnet_conditioning_scale=0.9,
        generator=gen,
        max_sequence_length=512,
        height=side,
        width=side,
    ).images[0]
    out = np.array(result.resize((w, h), Image.Resampling.BILINEAR))
    return np.where((mask > 0)[..., None], out, rgb)


def run_omnieraser_windows(rgb: np.ndarray | None, obj: np.ndarray | None) -> dict[str, Path]:
    written: dict[str, Path] = {}
    pending = []
    for name, r, c, n, *_ in CROPS:
        out_p = OUT / f"omnieraser_{name}.tif"
        if out_p.is_file():
            log(f"skip OmniEraser {name} (exists)")
            written[f"omni_{name}"] = out_p
        else:
            pending.append((name, r, c, n, out_p))
    if not pending:
        return written
    pipe = load_omnieraser_pipe()
    for name, r, c, n, out_p in pending:
        log(f"OmniEraser crop {name} {n}x{n} at r={r} c={c}")
        if rgb is None:
            crop = read_win(ORIG, c, r, n)
            with rasterio.open(MASK) as mds:
                hole = mds.read(1, window=Window(c, r, n, n)) > 0
        else:
            crop = rgb[r : r + n, c : c + n]
            hole = obj[r : r + n, c : c + n]
        filled = run_omnieraser(pipe, crop, hole)
        write_window_geotiff(out_p, filled, c, r)
        written[f"omni_{name}"] = out_p
        log(f"wrote {out_p.name}")
    del pipe
    import torch

    torch.cuda.empty_cache()
    return written


def save_omni_preview(omni_paths: dict[str, Path]) -> None:
    rows = []
    zooms = []
    for name, r, c, n, zr, zc, zn in CROPS:
        orig = read_win(ORIG, c, r, n)
        omni_p = omni_paths.get(f"omni_{name}")
        with rasterio.open(omni_p) as ds:
            omni = np.transpose(ds.read()[:3], (1, 2, 0))
        if omni.shape[:2] != (n, n):
            omni = np.array(Image.fromarray(omni).resize((n, n), Image.Resampling.BILINEAR))
        rows.append(hstack_labeled([orig, omni], [f"{name} original", f"{name} OmniEraser"]))
        zo = orig[zr - r : zr - r + zn, zc - c : zc - c + zn]
        zz = omni[zr - r : zr - r + zn, zc - c : zc - c + zn]
        zooms.append(hstack_labeled([zo, zz], [f"{name} zoom original", f"{name} zoom OmniEraser"]))

    def vstack(imgs: list[Image.Image]) -> Image.Image:
        w = max(i.width for i in imgs)
        hh = sum(i.height for i in imgs)
        out = Image.new("RGB", (w, hh), (0, 0, 0))
        y = 0
        for i in imgs:
            out.paste(i, (0, y))
            y += i.height
        return out

    overview = vstack(rows)
    overview.resize((overview.width // 2, overview.height // 2), Image.Resampling.BILINEAR).save(
        OUT / "_preview_omnieraser.jpg", quality=90
    )
    vstack(zooms).save(OUT / "_preview_omnieraser_zoom.jpg", quality=92)
    log("wrote OmniEraser JPEG previews")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-only", action="store_true", help="Only run OmniEraser on the two preview crops")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from tree_seg.config import load_config
        from tree_seg.ground_fill import (
            MatFiller,
            inpaint_per_object,
            landcover_fill,
            lift_context_shadows,
            load_landcover,
            segment_landcover,
        )
        from tree_seg.hf_auth import ensure_hf_auth
        from tree_seg.io_geotiff import open_rgb_geotiff, read_rgb

        ensure_hf_auth(verbose=True)
        os.environ.setdefault(
            "TORCH_HOME", str(ROOT / "outputs" / "checkpoints" / "ground_fill" / "torch")
        )
        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))

        if args.omni_only:
            log("=== OmniEraser crops only")
            omni_paths = run_omnieraser_windows(None, None)
            save_omni_preview(omni_paths)
            mat_p = OUT / "mat.tif"
            land_p = OUT / "landcover.tif"
            if mat_p.is_file() and land_p.is_file():
                save_previews({"mat": mat_p, "landcover": land_p, **omni_paths})
            log("=== OMNIERASER DONE")
            return 0

        cfg = load_config(ROOT / "configs" / "ground_fill.yaml")
        m = cfg.get("model", {})
        inp = cfg.get("inpaint", {})
        processor, model, id2label, object_ids, device = load_landcover(m.get("hf_id"))
        log(f"device={device} object_ids={sorted(object_ids)}")

        with open_rgb_geotiff(ORIG) as ds:
            rgb = read_rgb(ds)
            transform = ds.transform
            crs = ds.crs
        import rasterio as rio

        with rio.open(MASK) as mds:
            obj = mds.read(1) > 0
        log(f"loaded rgb {rgb.shape} hole={float(obj.mean()):.3f}")

        if PM.is_file():
            shutil.copy2(PM, OUT / "patchmatch.tif")
            log("copied patchmatch.tif (previous test, not a new method)")

        # --- land cover ---
        classes_p = OUT / "classes.tif"
        if classes_p.is_file():
            with rio.open(classes_p) as cds:
                classes = cds.read(1)
            log("reused classes.tif")
        else:
            log("segment land-cover classes")
            _, classes = segment_landcover(
                rgb,
                processor,
                model,
                object_ids,
                device,
                tile_size=int(m.get("tile_size", 512)),
                overlap=float(m.get("overlap", 0.125)),
            )
            from tree_seg.io_geotiff import write_geotiff

            write_geotiff(classes_p, classes, transform, crs, nodata=255, dtype="uint8")
            log(f"wrote {classes_p.name}")

        land_p = OUT / "landcover.tif"
        if land_p.is_file():
            log("skip landcover (exists)")
        else:
            log("land-cover fill")
            context = lift_context_shadows(rgb, obj.astype(np.uint8), luma_ratio=float(inp.get("shadow_luma_ratio", 0.55)))
            land = landcover_fill(context, obj.astype(np.uint8), classes)
            land = np.where(obj[..., None], land, rgb)
            write_rgb_geotiff(land_p, land, transform, crs)
            log(f"wrote {land_p.name}")
            del land

        # --- MAT ---
        mat_p = OUT / "mat.tif"
        if mat_p.is_file():
            log("skip MAT (exists)")
        else:
            log("load MAT")
            filler = MatFiller(device=device)
            context = lift_context_shadows(rgb, obj.astype(np.uint8), luma_ratio=float(inp.get("shadow_luma_ratio", 0.55)))
            log("MAT per-object fill")
            bare = inpaint_per_object(
                context,
                obj.astype(np.uint8),
                filler,
                pad_px=int(inp.get("object_pad_px", 96)),
                tile_size=512,
                overlap=0.125,
                split_erode_px=int(inp.get("split_erode_px", 12)),
            )
            bare = np.where(obj[..., None], bare, rgb)
            write_rgb_geotiff(mat_p, bare, transform, crs)
            log(f"wrote {mat_p.name}")
            del bare, filler
            import torch

            torch.cuda.empty_cache()

        # --- OmniEraser (comparison windows only) ---
        omni_paths: dict[str, Path] = {}
        try:
            omni_paths = run_omnieraser_windows(rgb, obj.astype(np.uint8))
        except Exception:
            log("OmniEraser failed:\n" + traceback.format_exc())

        paths = {"mat": mat_p, "landcover": land_p, **omni_paths}
        try:
            save_previews(paths)
        except Exception:
            log("preview failed:\n" + traceback.format_exc())

        log("=== COMPARE DONE")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
