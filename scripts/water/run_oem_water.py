"""OEM Mask2Former water-only on Fort Riley tiles (no dilation, no luma filter).

Writes mask, shapefile, and preview. Does not touch tree / orchard / roof /
building / SpaceNet / ground_fill checkpoints. Does not overwrite an existing
_preview_water_*.jpg (OEM baseline).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "Fort_Riley" / "Imagery"
OUT = ROOT / "outputs" / "footprints" / "Fort_Riley" / "water_oem"
LOG = ROOT / "outputs" / "logs" / "oem_water.log"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _to_pixels(gdf: gpd.GeoDataFrame, transform: Affine, out_w: int, out_h: int) -> list[list[tuple[int, int]]]:
    rings: list[list[tuple[int, int]]] = []
    if gdf.empty:
        return rings
    inv = ~transform

    def add_ring(coords) -> None:
        pts = []
        for x, y in coords:
            c, r = inv * (x, y)
            pts.append((int(round(c)), int(round(r))))
        if len(pts) >= 3:
            rings.append(pts)

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            add_ring(geom.exterior.coords)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                add_ring(part.exterior.coords)
    return [pts for pts in rings if any(0 <= x < out_w and 0 <= y < out_h for x, y in pts)]


def write_preview(
    rgb: np.ndarray,
    transform: Affine,
    water: gpd.GeoDataFrame,
    path: Path,
    size: int = 2048,
) -> None:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    base = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    ds_t = Affine(transform.a * w / nw, transform.b, transform.c, transform.d, transform.e * h / nh, transform.f)
    im = Image.fromarray(base.copy())
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ring in _to_pixels(water, ds_t, nw, nh):
        draw.polygon(ring, fill=(0, 160, 255, 90), outline=(0, 200, 255, 230))
    mid = np.asarray(Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB"))
    gap = np.full((nh, 8, 3), 30, dtype=np.uint8)
    sheet = np.concatenate([base, gap, mid], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=88)
    log(f"preview {path.name} n_water={len(water)}")


def water_class_ids(id2label: dict) -> set[int]:
    ids = {i for i, name in id2label.items() if "water" in str(name).lower()}
    return ids or {5}


def run_tile(
    tif: Path,
    processor,
    model,
    device,
    water_ids: set[int],
    tile_size: int,
    overlap: float,
    min_area: float,
    *,
    force: bool = False,
) -> dict:
    from tree_seg.ground_fill import segment_objects
    from tree_seg.io_geotiff import read_rgb, write_geotiff
    from tree_seg.postprocess import mask_to_polygons

    stem = tif.stem
    log(f"segment water-only {stem} ids={sorted(water_ids)}")
    with rasterio.open(tif) as ds:
        rgb = read_rgb(ds)
        transform = ds.transform
        crs = ds.crs
        gsd = float(max(abs(ds.transform.a), abs(ds.transform.e)))

    mask_path = OUT / f"{stem}_water.tif"
    raw_path = OUT / f"{stem}_water_raw.tif"
    reuse = (not force) and (raw_path.is_file() or mask_path.is_file())
    if reuse and raw_path.is_file():
        with rasterio.open(raw_path) as ds:
            mask = ds.read(1)
        log(f"  reuse {raw_path.name}")
        if not mask_path.is_file():
            write_geotiff(mask_path, mask.astype(np.uint8), transform, crs, nodata=0, dtype="uint8")
    elif reuse:
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1)
        write_geotiff(raw_path, mask.astype(np.uint8), transform, crs, nodata=0, dtype="uint8")
        log(f"  reuse {mask_path.name}")
    else:
        if processor is None or model is None:
            raise RuntimeError("model not loaded; cannot segment")
        mask = segment_objects(rgb, processor, model, water_ids, device, tile_size=tile_size, overlap=overlap)
        write_geotiff(mask_path, mask.astype(np.uint8), transform, crs, nodata=0, dtype="uint8")
        write_geotiff(raw_path, mask.astype(np.uint8), transform, crs, nodata=0, dtype="uint8")

    frac = float((mask > 0).mean())
    log(f"  water_frac={frac:.4f} gsd={gsd:.3f}")
    gdf = mask_to_polygons(mask, transform, min_area_m2=min_area, pixel_size_m=gsd)
    if crs is not None:
        gdf = gdf.set_crs(crs)
    if not gdf.empty:
        gdf["tile_id"] = stem
        gdf["source"] = "oem_water"
        shp_out = OUT / f"{stem}_water.shp"
        try:
            gdf.to_file(shp_out, driver="ESRI Shapefile")
        except PermissionError:
            shp_out = OUT / f"{stem}_water_oem.shp"
            gdf.to_file(shp_out, driver="ESRI Shapefile")
            log(f"  shapefile locked; wrote {shp_out.name}")
    else:
        shp_out = OUT / f"{stem}_water.shp"
        log(f"  no polygons above {min_area} m2")

    preview = OUT / f"_preview_water_{stem}.jpg"
    if preview.is_file():
        log(f"  keep existing OEM preview {preview.name}")
    else:
        write_preview(rgb, transform, gdf, preview)
    return {
        "tile": stem,
        "water_frac": frac,
        "n_water": int(len(gdf)),
        "water_area_m2": float(gdf["area_m2"].sum()) if len(gdf) else 0.0,
        "mask": str(mask_path),
        "shp": str(shp_out) if shp_out.is_file() else None,
        "preview": str(preview),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OEM water-only on Fort Riley tiles")
    parser.add_argument("--tiles", nargs="+", default=["FortRiley_r51200_c40960"])
    parser.add_argument("--min-area", type=float, default=25.0)
    parser.add_argument("--force", action="store_true", help="Re-run OEM even if a water mask already exists")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from tree_seg.config import load_config
    from tree_seg.ground_fill import load_landcover
    from tree_seg.hf_auth import ensure_hf_auth

    ensure_hf_auth(verbose=True)
    cfg = load_config(ROOT / "configs" / "ground_fill.yaml")
    m = cfg.get("model", {})
    hf_id = str(m.get("hf_id", "mfaytin/mask2former-satellite"))
    tile_size = int(m.get("tile_size", 512))
    overlap = float(m.get("overlap", 0.125))

    paths = []
    for name in args.tiles:
        p = Path(name)
        if not p.is_file():
            p = IMG_DIR / name
        if p.suffix == "":
            p = IMG_DIR / f"{name}.tif"
        if not p.is_file():
            raise FileNotFoundError(name)
        paths.append(p)

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== OEM water-only tiles={[p.stem for p in paths]} force={args.force} ===")
    need_model = args.force or any(
        not (OUT / f"{p.stem}_water_raw.tif").is_file() and not (OUT / f"{p.stem}_water.tif").is_file()
        for p in paths
    )
    processor = model = device = None
    water_ids: set[int] = {5}
    if need_model:
        processor, model, id2label, _object_ids, device = load_landcover(hf_id)
        water_ids = water_class_ids(id2label)
        log(f"model={hf_id} device={device} labels={id2label} water_ids={sorted(water_ids)}")
    else:
        log("reuse existing OEM water masks; skip model load")

    rows = []
    for tif in paths:
        rows.append(run_tile(tif, processor, model, device, water_ids, tile_size, overlap, args.min_area, force=args.force))

    summary = OUT / "summary.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
