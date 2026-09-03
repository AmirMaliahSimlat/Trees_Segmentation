#!/usr/bin/env python
"""Build orchard-block chips from EuroCrops parcels + IGN PNOA ~25 cm RGB.

Portugal DGT WMS is not reachable from this environment; Spain PNOA is.
Uses EuroCrops Navarra (ES_NA) farmer-declared olive / fruit / vineyard polygons.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
ZENODO_ES_NA = "https://zenodo.org/api/records/14094196/files/ES_NA_2020.zip/content"
PNOA_WMS = "https://www.ign.es/wms-inspire/pnoa-ma"

# HCAT names that are planted tree/vine blocks (not forest, not annual crops).
ORCHARD_NAMES = {
    "olive_plantations",
    "vineyards_wine_vine_rebland_grapes",
    "orchards_fruits",
    "unspecified_orchards_fruits",
    "citrus_plantations",
    "almond",
    "apples",
    "pears",
    "peach",
    "cherry_cherries",
    "nuts",
    "fig",
    "hazelnuts_hazel",
    "avocado",
    "pistachio",
    "sweet_chestnuts",
    "walnuts",
    "oranges",
    "olives",
}

NAME_SUBSTR = (
    "olive",
    "vineyard",
    "orchard",
    "citrus",
    "almond",
    "fruit",
    "grape",
    "walnut",
    "pistachio",
    "chestnut",
    "fig",
    "peach",
    "apple",
    "pear",
    "cherry",
    "avocado",
    "hazel",
    "nut",
)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    print(f"downloading {url} -> {dest}", flush=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 tree-seg"})
    with urlopen(req, timeout=600) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def _ensure_es_na(raw_dir: Path) -> Path:
    zpath = _download(ZENODO_ES_NA, raw_dir / "ES_NA_2020.zip")
    shp = next(raw_dir.rglob("ES_NA_2020_EC21.shp"), None)
    if shp is None:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(raw_dir / "ES_NA_2020")
        shp = next((raw_dir / "ES_NA_2020").rglob("ES_NA_2020_EC21.shp"))
    return shp


def _is_orchard_row(name: str) -> bool:
    n = (name or "").strip().lower()
    if n in ORCHARD_NAMES:
        return True
    return any(k in n for k in NAME_SUBSTR)


def _wms_rgb(xmin: float, ymin: float, xmax: float, ymax: float, size: int) -> np.ndarray | None:
    url = (
        f"{PNOA_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS=OI.OrthoimageCoverage&STYLES=&CRS=EPSG:3857"
        f"&WIDTH={size}&HEIGHT={size}&FORMAT=image/jpeg"
        f"&BBOX={xmin},{ymin},{xmax},{ymax}"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 tree-seg"})
    try:
        with urlopen(req, timeout=15) as resp:
            data = resp.read()
    except Exception:
        return None
    if not data or data[:2] != b"\xff\xd8":
        return None
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None or arr.size == 0:
        return None
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if float(rgb.mean()) < 5 or float(rgb.mean()) > 250:
        return None
    return rgb


def _rasterize(geoms_3857, xmin, ymin, xmax, ymax, size: int) -> np.ndarray:
    transform = from_bounds(xmin, ymin, xmax, ymax, size, size)
    if not geoms_3857:
        return np.zeros((size, size), dtype=np.uint8)
    mask = rasterize(
        [(g, 1) for g in geoms_3857 if g is not None and not g.is_empty],
        out_shape=(size, size),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return mask


def prepare(
    output_dir: Path,
    *,
    raw_dir: Path,
    chip_size: int = 512,
    gsd_m: float = 0.25,
    max_train: int = 500,
    max_val: int = 70,
    seed: int = 42,
) -> dict:
    import geopandas as gpd

    rng = random.Random(seed)
    output_dir = Path(output_dir)
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    shp = _ensure_es_na(raw_dir)
    print(f"reading {shp}", flush=True)
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    name_col = "EC_hcat_n" if "EC_hcat_n" in gdf.columns else None
    if name_col is None:
        raise RuntimeError(f"No EC_hcat_n in {list(gdf.columns)}")

    orchard = gdf[gdf[name_col].astype(str).map(_is_orchard_row)].copy()
    other = gdf[~gdf.index.isin(orchard.index)]
    print(f"parcels={len(gdf)} orchard={len(orchard)} other={len(other)}", flush=True)
    print(orchard[name_col].value_counts().head(20).to_string(), flush=True)

    orchard_3857 = orchard.to_crs(3857)
    neg_sample = min(len(other), max(4000, (max_train + max_val) * 8))
    other_sample = other.sample(n=neg_sample, random_state=seed) if len(other) else other
    other_3857 = other_sample.to_crs(3857) if len(other_sample) else other_sample

    half = chip_size * gsd_m / 2.0
    target_pos = int(round(0.8 * (max_train + max_val)))
    target_neg = (max_train + max_val) - target_pos

    def _take(frame, n: int) -> list:
        ids = list(frame.index)
        rng.shuffle(ids)
        return ids[:n]

    names = orchard_3857[name_col].astype(str).str.lower()
    olive_idx = _take(orchard_3857[names.str.contains("olive")], max(1, int(0.5 * (max_train + max_val) * 3)))
    vine_idx = _take(orchard_3857[names.str.contains("vine")], max(1, int(0.2 * (max_train + max_val) * 3)))
    rest_idx = _take(
        orchard_3857[~names.str.contains("olive") & ~names.str.contains("vine")],
        max(1, int(0.5 * (max_train + max_val) * 3)),
    )
    pos_idx = olive_idx + vine_idx + rest_idx
    rng.shuffle(pos_idx)
    neg_idx = list(other_3857.index) if len(other_3857) else []
    rng.shuffle(neg_idx)

    sindex = orchard_3857.sindex

    def chip_at(cx: float, cy: float) -> tuple[np.ndarray, np.ndarray] | None:
        xmin, ymin, xmax, ymax = cx - half, cy - half, cx + half, cy + half
        rgb = _wms_rgb(xmin, ymin, xmax, ymax, chip_size)
        if rgb is None:
            return None
        hits = list(sindex.intersection((xmin, ymin, xmax, ymax)))
        geoms = []
        for i in hits:
            geom = orchard_3857.geometry.iloc[i]
            clipped = geom.intersection(box(xmin, ymin, xmax, ymax))
            if not clipped.is_empty:
                geoms.append(clipped)
        mask = _rasterize(geoms, xmin, ymin, xmax, ymax, chip_size)
        return rgb, mask

    def _pos_job(idx):
        geom = orchard_3857.geometry.loc[idx]
        if geom is None or geom.is_empty:
            return None
        c = geom.centroid
        jitter = random.uniform(-half * 0.25, half * 0.25)
        pair = chip_at(c.x + jitter, c.y + jitter)
        if pair is None:
            return None
        rgb, mask = pair
        if float(mask.mean()) < 0.03:
            return None
        return rgb, mask

    def _neg_job(idx):
        geom = other_3857.geometry.loc[idx]
        if geom is None or geom.is_empty:
            return None
        c = geom.centroid
        pair = chip_at(c.x, c.y)
        if pair is None:
            return None
        rgb, mask = pair
        if float(mask.mean()) >= 0.01:
            return None
        return rgb, mask

    def _collect(jobs, worker, need: int, desc: str) -> list:
        out: list[tuple[np.ndarray, np.ndarray]] = []
        pool = ThreadPoolExecutor(max_workers=12)
        try:
            futs = [pool.submit(worker, j) for j in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                try:
                    pair = fut.result()
                except Exception:
                    continue
                if pair is None:
                    continue
                out.append(pair)
                if len(out) >= need:
                    break
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return out

    positives = _collect(pos_idx[:900], _pos_job, target_pos, "positive parcels")
    negatives = _collect(neg_idx[: max(target_neg * 8, 200)], _neg_job, target_neg, "negative parcels")

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
            stem = f"ec_{i:04d}"
            cv2.imwrite(str(img_dir / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(msk_dir / f"{stem}.png"), (mask * 255).astype(np.uint8))
        return len(items)

    n_train = _write("train", train_pairs)
    n_val_w = _write("val", val_pairs)
    meta = {
        "source": "EuroCrops ES_NA_2020 + IGN PNOA WMS ~25cm",
        "n_train": n_train,
        "n_val": n_val_w,
        "n_positive_sampled": len(positives),
        "n_negative_sampled": len(negatives),
        "chip_size": chip_size,
        "gsd_m": gsd_m,
        "orchard_classes": orchard[name_col].value_counts().to_dict(),
    }
    (output_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    if n_train < 20:
        raise RuntimeError(f"Too few training chips ({n_train}).")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EuroCrops orchard-block chips")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "data" / "shared" / "datasets" / "orchard_eurocrops")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "shared" / "sources" / "eurocrops")
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-val", type=int, default=70)
    parser.add_argument("--chip-size", type=int, default=512)
    parser.add_argument("--gsd", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(
        args.output,
        raw_dir=args.raw_dir,
        chip_size=args.chip_size,
        gsd_m=args.gsd,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
