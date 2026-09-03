#!/usr/bin/env python
"""Build orchard-block train/val PNG pairs from NAIP RGB + USDA CDL labels.

Labels are USDA Cropland Data Layer classes for orchards, vineyards, citrus,
nuts, and other tree crops. Imagery is NAIP aerial RGB (~0.6 m). Both come
from Microsoft Planetary Computer (no API key required).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# Faster / more reliable remote COG reads (Azure NAIP / CDL)
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform as warp_xy
from rasterio.warp import transform_bounds
from tqdm import tqdm

# USDA CDL codes for planted tree/vine blocks (not forest / not row crops).
ORCHARD_CDL = {
    66,  # Cherries
    67,  # Peaches
    68,  # Apples
    69,  # Grapes (vineyards)
    70,  # Christmas Trees
    71,  # Other Tree Crops
    72,  # Citrus
    74,  # Pecans
    75,  # Almonds
    76,  # Walnuts
    77,  # Pears
    204,  # Pistachios
    211,  # Olives
    212,  # Oranges
    223,  # Caneberries
}

# Orchard-dense AOIs (lon/lat): Central Valley, Napa, Yakima, Florida citrus, Kern pistachios.
AOIS: list[tuple[str, list[float]]] = [
    ("ca_fresno", [-119.85, 36.45, -119.35, 36.85]),
    ("ca_napa", [-122.48, 38.22, -122.22, 38.52]),
    ("ca_kern", [-119.35, 35.15, -118.85, 35.55]),
    ("wa_yakima", [-120.55, 46.42, -120.15, 46.72]),
    ("fl_citrus", [-81.75, 27.85, -81.35, 28.25]),
]


def _open_catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def _latest_item(catalog, collection: str, bbox: list[float], datetime: str, *, asset: str | None = None):
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime,
        max_items=50,
    )
    items = list(search.items())
    if asset:
        items = [it for it in items if asset in it.assets]
    if not items:
        return None

    def _key(it):
        return str(it.properties.get("start_datetime") or it.properties.get("datetime") or it.id)

    items.sort(key=_key, reverse=True)
    return items[0]


def _read_rgb_window(item, col: int, row: int, size: int) -> np.ndarray | None:
    href = item.assets["image"].href
    with rasterio.open(href) as src:
        if col < 0 or row < 0 or col + size > src.width or row + size > src.height:
            return None
        data = src.read([1, 2, 3], window=Window(col, row, size, size))
        if data.size == 0:
            return None
        rgb = np.transpose(data, (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb


def _window_bounds(item, col: int, row: int, size: int):
    href = item.assets["image"].href
    with rasterio.open(href) as src:
        win = Window(col, row, size, size)
        bounds = rasterio.windows.bounds(win, src.transform)
        return bounds, src.crs


def _cdl_href(cdl_item) -> str:
    for key in ("cropland", "data", "cdl"):
        if key in cdl_item.assets:
            return cdl_item.assets[key].href
    raise KeyError(f"No cropland asset on {cdl_item.id}: {list(cdl_item.assets)}")


def prepare(
    output_dir: Path,
    *,
    chip_size: int = 512,
    max_train: int = 600,
    max_val: int = 80,
    seed: int = 42,
    naip_years: str = "2021-01-01/2023-12-31",
    cdl_year: str = "2021-01-01/2021-12-31",
) -> dict:
    rng = random.Random(seed)
    catalog = _open_catalog()
    output_dir = Path(output_dir)
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    target_pos = int(round(0.75 * (max_train + max_val)))
    target_neg = (max_train + max_val) - target_pos
    positives: list[tuple[np.ndarray, np.ndarray]] = []
    negatives: list[tuple[np.ndarray, np.ndarray]] = []
    orchard_codes = np.array(sorted(ORCHARD_CDL), dtype=np.uint16)

    for aoi_name, bbox in AOIS:
        if len(positives) >= target_pos and len(negatives) >= target_neg:
            break
        naip = _latest_item(catalog, "naip", bbox, naip_years, asset="image")
        cdl = _latest_item(catalog, "usda-cdl", bbox, cdl_year, asset="cropland")
        if naip is None or cdl is None:
            print(f"skip {aoi_name}: missing NAIP or CDL", flush=True)
            continue
        print(f"{aoi_name}: opening {naip.id}", flush=True)
        try:
            naip_ds = rasterio.open(naip.assets["image"].href)
            cdl_ds = rasterio.open(_cdl_href(cdl))
        except Exception as exc:
            print(f"skip {aoi_name}: open failed {exc}", flush=True)
            continue
        width, height = naip_ds.width, naip_ds.height
        print(f"{aoi_name}: NAIP {width}x{height}", flush=True)

        def read_pair(col: int, row: int) -> tuple[np.ndarray, np.ndarray] | None:
            if col < 0 or row < 0 or col + chip_size > width or row + chip_size > height:
                return None
            win = Window(col, row, chip_size, chip_size)
            data = naip_ds.read([1, 2, 3], window=win)
            rgb = np.transpose(data, (1, 2, 0))
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if float(rgb.mean()) < 8:
                return None
            bounds = rasterio.windows.bounds(win, naip_ds.transform)
            left, bottom, right, top = transform_bounds(naip_ds.crs, cdl_ds.crs, *bounds, densify_pts=8)
            cwin = from_bounds(left, bottom, right, top, transform=cdl_ds.transform)
            cdata = cdl_ds.read(1, window=cwin, boundless=True, fill_value=0)
            mask = np.isin(cdata, orchard_codes).astype(np.uint8)
            mask = cv2.resize(mask, (chip_size, chip_size), interpolation=cv2.INTER_NEAREST)
            return rgb, mask

        # Prefer windows that overlap orchard CDL cells
        naip_bounds = transform_bounds(naip_ds.crs, cdl_ds.crs, *naip_ds.bounds, densify_pts=8)
        overview = cdl_ds.read(1, window=from_bounds(*naip_bounds, transform=cdl_ds.transform), boundless=True, fill_value=0)
        ys, xs = np.where(np.isin(overview, orchard_codes))
        print(f"{aoi_name}: orchard CDL cells {len(ys)}", flush=True)

        n_pos_before = len(positives)
        n_neg_before = len(negatives)
        for _ in range(200):
            if len(positives) >= target_pos:
                break
            if len(ys) == 0:
                break
            i = rng.randrange(len(ys))
            # map CDL pixel to NAIP pixel via geographic center
            cwin_full = from_bounds(*naip_bounds, transform=cdl_ds.transform)
            ccol = int(cwin_full.col_off) + int(xs[i])
            crow = int(cwin_full.row_off) + int(ys[i])
            cx, cy = cdl_ds.xy(crow, ccol)
            (nx,), (ny,) = warp_xy(cdl_ds.crs, naip_ds.crs, [cx], [cy])
            row, col = naip_ds.index(nx, ny)
            col = int(col - chip_size // 2 + rng.randint(-32, 32))
            row = int(row - chip_size // 2 + rng.randint(-32, 32))
            try:
                pair = read_pair(col, row)
            except Exception:
                continue
            if pair is None:
                continue
            rgb, mask = pair
            if float(mask.mean()) >= 0.02:
                positives.append((rgb, mask))

        for _ in range(30):
            if len(negatives) >= target_neg:
                break
            col = rng.randint(0, max(0, width - chip_size))
            row = rng.randint(0, max(0, height - chip_size))
            try:
                pair = read_pair(col, row)
            except Exception:
                continue
            if pair is None:
                continue
            rgb, mask = pair
            if float(mask.mean()) < 0.01:
                negatives.append((rgb, mask))

        naip_ds.close()
        cdl_ds.close()
        print(
            f"{aoi_name}: +pos {len(positives) - n_pos_before} +neg {len(negatives) - n_neg_before} "
            f"(tot pos={len(positives)} neg={len(negatives)})",
            flush=True,
        )

    pairs = positives + negatives
    rng.shuffle(pairs)
    n_val = min(max_val, max(1, int(round(0.12 * len(pairs)))))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:][:max_train]
    val_pairs = val_pairs[:max_val]

    def _write(split: str, items: list[tuple[np.ndarray, np.ndarray]]) -> int:
        img_dir = output_dir / split / "images"
        msk_dir = output_dir / split / "masks"
        for i, (rgb, mask) in enumerate(tqdm(items, desc=f"write {split}")):
            stem = f"orch_{i:04d}"
            cv2.imwrite(str(img_dir / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(msk_dir / f"{stem}.png"), (mask * 255).astype(np.uint8))
        return len(items)

    n_train = _write("train", train_pairs)
    n_val_w = _write("val", val_pairs)
    meta = {
        "source": "NAIP + USDA CDL via Planetary Computer",
        "cdl_orchard_classes": sorted(ORCHARD_CDL),
        "n_train": n_train,
        "n_val": n_val_w,
        "n_positive_sampled": len(positives),
        "n_negative_sampled": len(negatives),
        "chip_size": chip_size,
        "aois": [a[0] for a in AOIS],
    }
    (output_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if n_train < 20:
        raise RuntimeError(f"Too few training chips ({n_train}). Check Planetary Computer access.")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare orchard-block dataset from NAIP+CDL")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/shared/datasets/orchard_blocks"))
    parser.add_argument("--max-train", type=int, default=600)
    parser.add_argument("--max-val", type=int, default=80)
    parser.add_argument("--chip-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(
        args.output,
        chip_size=args.chip_size,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
