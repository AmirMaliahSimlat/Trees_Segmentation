#!/usr/bin/env python
"""One Stability Erase call on a <=4 MP crop of a Fort Riley tile.

Uses OEM Mask2Former (trees + buildings) for the hole mask.
Does not run the full 10240 tile (that would be many paid calls).
Does not call the Inpaint endpoint. Does not overwrite LaMa/PatchMatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.windows import Window, transform as window_transform

ROOT = Path(__file__).resolve().parents[2]
TILE_NAME = "FortRiley_r81920_c61440.tif"
IMG = ROOT / "data" / "Fort_Riley" / "Imagery" / TILE_NAME
HINT_MASK = ROOT / "outputs" / "ground_fill" / "Fort_Riley" / "objects" / "FortRiley_r81920_c61440_objects.tif"
OUT_DIR = ROOT / "outputs" / "ground_fill" / "Fort_Riley" / "stability_erase"
LOG = ROOT / "outputs" / "logs" / "stability_erase_sample.log"
CROP_SIDE = 1920  # 3.69 MP, under the 4.19 MP API cap


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_preview(rgb: np.ndarray, mask: np.ndarray, erased: np.ndarray, path: Path) -> None:
    h, w = rgb.shape[:2]
    overlay = rgb.copy()
    red = overlay.copy()
    red[..., 0] = np.maximum(red[..., 0], 220)
    overlay = np.where((mask > 0)[..., None], (0.45 * overlay + 0.55 * red).astype(np.uint8), overlay)
    gap = np.full((h, 8, 3), 30, dtype=np.uint8)
    strip = np.concatenate([rgb, gap, overlay, gap, erased], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", type=Path, default=IMG)
    parser.add_argument("--side", type=int, default=CROP_SIDE, help="Crop edge in pixels (keep side^2 < 4.19e6)")
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--col", type=int, default=None)
    parser.add_argument("--dilate", type=int, default=6, help="Grow each hole (px); overlaps stay as context")
    parser.add_argument("--union-dilate", type=int, default=0, help="Dilate combined mask (may merge instances)")
    parser.add_argument("--shadow-grow", type=int, default=0)
    parser.add_argument("--open", type=int, default=1, dest="open_px")
    parser.add_argument("--split-min-dist", type=int, default=10, help="Peak spacing in px for instance split")
    parser.add_argument("--gap", type=int, default=2, help="Unmasked pixels between instances before expand")
    parser.add_argument("--tag", default="mid", help="Output filename tag (avoids overwriting prior run)")
    parser.add_argument("--mask", type=Path, default=None, help="Reuse a saved hole mask; skip OEM")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from tree_seg.config import load_config
    from tree_seg.ground_fill import (
        expand_instances_keep_gaps,
        grow_attached_shadows,
        load_landcover,
        object_holes_with_context_gaps,
        segment_objects,
        union_dilate_mask,
    )
    from tree_seg.hf_auth import ensure_hf_auth
    from tree_seg.io_geotiff import open_rgb_geotiff, read_rgb, write_geotiff
    from tree_seg.stability_erase import StabilityEraseFiller, pick_object_window, stability_api_key

    if args.mask is None:
        ensure_hf_auth(verbose=True)
    key = stability_api_key()
    log(f"STABILITY_API_KEY set len={len(key)}")

    tile = Path(args.tile)
    if not tile.is_file():
        raise FileNotFoundError(tile)
    side = int(args.side)
    if side * side > 4_000_000:
        raise SystemExit(f"side={side} is {side * side} px; keep under 4e6 for the API cap")

    import rasterio

    with rasterio.open(tile) as ds:
        th, tw = ds.height, ds.width
        full_transform = ds.transform
        crs = ds.crs
        log(f"tile {tile.name} {tw}x{th} px={tw * th} gsd={abs(ds.transform.a):.4f}m")

    if args.row is not None and args.col is not None:
        row, col = int(args.row), int(args.col)
        frac = float("nan")
    elif HINT_MASK.is_file():
        with rasterio.open(HINT_MASK) as mds:
            hint = mds.read(1)
        row, col, frac = pick_object_window(hint, min(side, hint.shape[0], hint.shape[1]))
        log(f"crop from existing objects hint row={row} col={col} obj_frac={frac:.3f}")
        del hint
    else:
        row = max(0, (th - side) // 2)
        col = max(0, (tw - side) // 2)
        log(f"no objects hint; center crop row={row} col={col}")

    row = min(max(0, row), max(0, th - side))
    col = min(max(0, col), max(0, tw - side))
    win = Window(col, row, min(side, tw - col), min(side, th - row))
    with open_rgb_geotiff(tile) as ds:
        rgb = read_rgb(ds, window=win)
        crop_transform = window_transform(win, full_transform)

    h, w = rgb.shape[:2]
    log(f"crop {w}x{h} = {w * h} px (API max 4194304)")

    import cv2
    import rasterio as rio

    if args.mask is not None:
        mask_src = Path(args.mask)
        if not mask_src.is_file():
            raise FileNotFoundError(mask_src)
        with rio.open(mask_src) as mds:
            obj = mds.read(1)
        if obj.shape[:2] != (h, w):
            raise SystemExit(f"mask {obj.shape} does not match crop {(h, w)}")
        log(f"reuse mask {mask_src.name} object_frac={float((obj > 0).mean()):.4f}")
    else:
        cfg = load_config(ROOT / "configs" / "ground_fill.yaml")
        m = cfg.get("model", {})
        processor, model, id2label, object_ids, device = load_landcover(m.get("hf_id"))
        log(f"OEM device={device} object_ids={sorted(object_ids)} labels={id2label}")
        obj = segment_objects(
            rgb,
            processor,
            model,
            object_ids,
            device,
            tile_size=int(m.get("tile_size", 512)),
            overlap=float(m.get("overlap", 0.125)),
        )
        log(f"raw OEM object_frac={float(obj.mean()):.4f}")
        obj = object_holes_with_context_gaps(
            obj,
            min_distance_px=int(args.split_min_dist),
            gap_px=int(args.gap),
            open_px=int(args.open_px),
        )
        if int(args.dilate) > 0:
            obj = expand_instances_keep_gaps(obj, int(args.dilate))
        if int(args.union_dilate) > 0:
            obj = union_dilate_mask(obj, int(args.union_dilate))
        if int(args.shadow_grow) > 0:
            obj = grow_attached_shadows(rgb, obj, max_px=int(args.shadow_grow), luma_ratio=0.55)

    n_cc, _lab, st, _ = cv2.connectedComponentsWithStats((obj > 0).astype("uint8"), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA] if n_cc > 1 else []
    largest = int(areas.max()) if len(areas) else 0
    log(
        f"mask object_frac={float(obj.mean()):.4f} px={int(obj.sum())} "
        f"components={max(n_cc - 1, 0)} largest={largest}"
    )
    if not obj.any():
        log("no trees/buildings in crop; abort (no API call)")
        return 1

    filler = StabilityEraseFiller(allow_tile=False, grow_mask=0, timeout=180)
    log("POST erase (1 call, 5 credits)")
    erased = filler.fill(rgb, obj)
    log(f"erase done calls={filler.calls}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{tile.stem}_r{row}_c{col}_{args.tag}"
    mask_path = OUT_DIR / f"{stem}_objects.tif"
    bare_path = OUT_DIR / f"{stem}_erase.tif"
    preview = OUT_DIR / f"_preview_{stem}.jpg"
    write_geotiff(mask_path, obj, crop_transform, crs, nodata=0, dtype="uint8")
    write_geotiff(bare_path, np.transpose(erased, (2, 0, 1)), crop_transform, crs, dtype="uint8")
    write_preview(rgb, obj, erased, preview)
    meta = {
        "tile": str(tile),
        "row": row,
        "col": col,
        "width": w,
        "height": h,
        "pixels": w * h,
        "object_frac": float(obj.mean()),
        "n_components": int(max(n_cc - 1, 0)),
        "largest_component_px": largest,
        "dilate": int(args.dilate),
        "union_dilate": int(args.union_dilate),
        "shadow_grow": int(args.shadow_grow),
        "gap": int(args.gap),
        "split_min_dist": int(args.split_min_dist),
        "api_calls": filler.calls,
        "mask": str(mask_path),
        "erase": str(bare_path),
        "preview": str(preview),
        "backend": filler.backend,
    }
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
