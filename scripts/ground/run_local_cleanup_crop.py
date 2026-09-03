#!/usr/bin/env python
"""Local PatchMatch/LaMa cleanup on a Stability-erased crop.

Re-runs OEM tree/building segmentation on the erased RGB, then fills leftovers
with the existing ground-fill backend. No Stability API call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = (
    ROOT
    / "outputs"
    / "ground_fill"
    / "Fort_Riley"
    / "stability_erase"
    / "FortRiley_r81920_c61440_r96_c768_d9_erase.tif"
)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    red = out.copy()
    red[..., 0] = np.maximum(red[..., 0], 220)
    return np.where((mask > 0)[..., None], (0.45 * out + 0.55 * red).astype(np.uint8), out)


def save_strip(parts: list[np.ndarray], labels: list[str], path: Path) -> None:
    h, w = parts[0].shape[:2]
    bar = 48
    gap = 8
    n = len(parts)
    canvas = Image.new("RGB", (n * w + (n - 1) * gap, bar + h), (18, 18, 18))
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    x = 0
    for img, title in zip(parts, labels):
        draw.text((x + 10, 12), title, fill=(240, 240, 240), font=font)
        canvas.paste(Image.fromarray(img), (x, bar))
        x += w + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ground_fill.yaml")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from tree_seg.config import load_config
    from tree_seg.ground_fill import (
        blend_local_grain,
        dilate_mask,
        grow_attached_shadows,
        inpaint_per_object,
        lift_context_shadows,
        load_landcover,
        make_filler,
        segment_objects,
    )
    from tree_seg.hf_auth import ensure_hf_auth
    from tree_seg.io_geotiff import open_rgb_geotiff, read_rgb, write_geotiff

    ensure_hf_auth(verbose=True)
    src = Path(args.input)
    if not src.is_file():
        raise FileNotFoundError(src)

    cfg = load_config(args.config)
    m = cfg.get("model", {})
    inp = cfg.get("inpaint", {})
    processor, model, id2label, object_ids, device = load_landcover(m.get("hf_id"))
    filler = make_filler(inp, device)
    print(f"device={device} backend={filler.backend} object_ids={sorted(object_ids)}", flush=True)

    with open_rgb_geotiff(src) as ds:
        rgb = read_rgb(ds)
        transform = ds.transform
        crs = ds.crs

    obj = segment_objects(
        rgb,
        processor,
        model,
        object_ids,
        device,
        tile_size=int(m.get("tile_size", 512)),
        overlap=float(m.get("overlap", 0.125)),
    )
    raw_frac = float(obj.mean())
    obj = dilate_mask(obj, int(inp.get("dilate_px", 12)))
    obj = grow_attached_shadows(
        rgb, obj, max_px=int(inp.get("shadow_grow_px", 96)), luma_ratio=float(inp.get("shadow_luma_ratio", 0.55))
    )
    print(f"pass2 raw_frac={raw_frac:.4f} after_dilate_shadow={float(obj.mean()):.4f} px={int(obj.sum())}", flush=True)

    context = rgb
    if bool(inp.get("lift_context", True)):
        context = lift_context_shadows(rgb, obj, luma_ratio=float(inp.get("shadow_luma_ratio", 0.55)))
    if obj.any():
        bare = inpaint_per_object(
            context,
            obj,
            filler,
            pad_px=int(inp.get("object_pad_px", 96)),
            tile_size=int(inp.get("tile_size", 1024)),
            overlap=float(inp.get("overlap", 0.125)),
            split_erode_px=int(inp.get("split_erode_px", 12)),
        )
        bare = np.where(obj[..., None].astype(bool), bare, rgb)
        bare = blend_local_grain(bare, rgb, obj)
    else:
        bare = rgb

    out_dir = src.parent
    stem = src.stem.replace("_erase", "") + "_local"
    mask_path = out_dir / f"{stem}_objects.tif"
    bare_path = out_dir / f"{stem}_bare.tif"
    preview = out_dir / f"_preview_{stem}.jpg"
    seg_preview = out_dir / f"_preview_{stem}_segmentation.jpg"
    write_geotiff(mask_path, obj, transform, crs, nodata=0, dtype="uint8")
    write_geotiff(bare_path, np.transpose(bare, (2, 0, 1)), transform, crs, dtype="uint8")
    ov = overlay_mask(rgb, obj)
    Image.fromarray(ov).save(seg_preview, quality=90)
    Image.fromarray(bare).save(out_dir / f"_preview_{stem}_final.jpg", quality=90)
    save_strip(
        [rgb, ov, bare],
        ["after Stability erase", "pass-2 OEM trees+buildings", "after local PatchMatch fill"],
        preview,
    )
    meta = {
        "input": str(src),
        "backend": filler.backend,
        "raw_object_frac": raw_frac,
        "mask_frac": float(obj.mean()),
        "mask": str(mask_path),
        "bare": str(bare_path),
        "preview": str(preview),
        "segmentation_jpg": str(seg_preview),
        "final_jpg": str(out_dir / f"_preview_{stem}_final.jpg"),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
