"""OEM Mask2Former buildings-only on Fort Riley tiles (no trees, no dilation).

Writes per-tile masks, shapefiles, and previews vs OSM. Does not touch
tree / orchard / roof / ground_fill checkpoints.
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
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import Affine
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "Fort_Riley" / "Imagery"
BLDG = ROOT / "data" / "Fort_Riley" / "Buildings" / "1.shp"
OEM_ROOT = ROOT / "outputs" / "footprints" / "Fort_Riley" / "buildings_oem"
OUT = OEM_ROOT / "runs"
LOG = ROOT / "outputs" / "logs" / "oem_buildings.log"
TEST_TILE = "FortRiley_r81920_c61440.tif"
# Frozen 2026-08-23 reference previews — never write these names in OEM_ROOT.
PROTECTED_PREVIEWS = {
    "_preview_FortRiley_r81920_c61440.jpg",
    "_preview_FortRiley_r20480_c120032.jpg",
    "_preview_FortRiley_r10240_c120032.jpg",
}


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pick_tiles(n: int) -> list[Path]:
    tifs = sorted(IMG_DIR.glob("FortRiley_*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No FortRiley_*.tif in {IMG_DIR}")
    forced = IMG_DIR / TEST_TILE
    bldg = gpd.read_file(BLDG)
    with rasterio.open(tifs[0]) as ds:
        bldg = bldg.to_crs(ds.crs)
    sindex = bldg.sindex
    scored: list[tuple[float, Path]] = []
    for tif in tifs:
        with rasterio.open(tif) as ds:
            geom = box(*ds.bounds)
        idx = list(sindex.intersection(geom.bounds))
        if not idx:
            area = 0.0
        else:
            hit = bldg.iloc[idx]
            hit = hit[hit.intersects(geom)]
            area = float(hit.geometry.intersection(geom).area.sum()) if len(hit) else 0.0
        scored.append((area, tif))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen: list[Path] = []
    if forced.is_file():
        chosen.append(forced)
    for _area, path in scored:
        if path.resolve() == forced.resolve():
            continue
        chosen.append(path)
        if len(chosen) >= n:
            break
    return chosen[:n]


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
    clipped = []
    for pts in rings:
        if any(0 <= x < out_w and 0 <= y < out_h for x, y in pts):
            clipped.append(pts)
    return clipped


def write_preview(
    rgb: np.ndarray,
    transform: Affine,
    oem: gpd.GeoDataFrame,
    osm: gpd.GeoDataFrame,
    path: Path,
    size: int = 2048,
) -> None:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    base = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    ds_t = Affine(transform.a * w / nw, transform.b, transform.c, transform.d, transform.e * h / nh, transform.f)

    def panel(polys: list[list[tuple[int, int]]], color: tuple[int, int, int], fill_a: int) -> np.ndarray:
        im = Image.fromarray(base.copy())
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for pts in polys:
            draw.polygon(pts, fill=(*color, fill_a), outline=(*color, 230))
        out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        return np.asarray(out)

    oem_pts = _to_pixels(oem, ds_t, nw, nh)
    osm_pts = _to_pixels(osm, ds_t, nw, nh)
    left = base
    mid = panel(oem_pts, (0, 220, 80), 70)
    right = panel(osm_pts, (230, 40, 40), 70)
    gap = np.full((nh, 8, 3), 30, dtype=np.uint8)
    sheet = np.concatenate([left, gap, mid, gap, right], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=88)
    log(f"preview {path.name} oem={len(oem)} osm={len(osm)}")


def run_tile(tif: Path, processor, model, device, tile_size: int, overlap: float, min_area: float) -> dict:
    from tree_seg.ground_fill import segment_objects
    from tree_seg.io_geotiff import read_rgb, write_geotiff
    from tree_seg.postprocess import mask_to_polygons

    stem = tif.stem
    log(f"segment buildings-only {stem}")
    with rasterio.open(tif) as ds:
        rgb = read_rgb(ds)
        transform = ds.transform
        crs = ds.crs
        bounds = box(*ds.bounds)
        gsd = float(max(abs(ds.transform.a), abs(ds.transform.e)))

    mask = segment_objects(
        rgb, processor, model, {7}, device, tile_size=tile_size, overlap=overlap
    )
    frac = float(mask.mean())
    log(f"  building_frac={frac:.4f} gsd={gsd:.3f}")

    tif_out = OUT / f"{stem}_buildings.tif"
    write_geotiff(tif_out, mask.astype(np.uint8), transform, crs, nodata=0, dtype="uint8")

    gdf = mask_to_polygons(mask, transform, min_area_m2=min_area, pixel_size_m=gsd)
    if crs is not None:
        gdf = gdf.set_crs(crs)
    gdf["tile_id"] = stem
    gdf["source"] = "oem_building"
    shp_out = OUT / f"{stem}_buildings.shp"
    if not gdf.empty:
        gdf.to_file(shp_out, driver="ESRI Shapefile")
    else:
        log(f"  no polygons above {min_area} m2")

    osm = gpd.GeoDataFrame(geometry=[], crs=crs)
    if BLDG.is_file():
        osm = gpd.read_file(BLDG).to_crs(crs)
        osm = osm[osm.intersects(bounds)].copy()
        if not osm.empty:
            osm.geometry = osm.geometry.intersection(bounds)
            osm = osm[~osm.geometry.is_empty]

    preview = OUT / f"_preview_{stem}.jpg"
    if preview.name in PROTECTED_PREVIEWS or (OEM_ROOT / preview.name).exists():
        preview = OUT / f"_preview_{stem}.jpg"
    write_preview(rgb, transform, gdf, osm, preview)
    return {
        "tile": stem,
        "building_frac": frac,
        "n_oem": int(len(gdf)),
        "n_osm": int(len(osm)),
        "oem_area_m2": float(gdf["area_m2"].sum()) if len(gdf) else 0.0,
        "mask": str(tif_out),
        "shp": str(shp_out) if shp_out.is_file() else None,
        "preview": str(preview),
    }


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description="OEM buildings-only on Fort Riley tiles")
    parser.add_argument("--tiles", nargs="+", default=None, help="Tile filenames or stems")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--min-area", type=float, default=15.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="Output folder (default: buildings_oem/runs; does not overwrite baseline_v1 previews)",
    )
    args = parser.parse_args()
    OUT = args.out

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

    if args.tiles:
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
    else:
        paths = pick_tiles(args.count)

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== OEM buildings-only tiles={[p.stem for p in paths]} ===")
    processor, model, id2label, _object_ids, device = load_landcover(hf_id)
    log(f"model={hf_id} device={device} class7={id2label.get(7)} (buildings only, no dilate)")

    rows = []
    frames = []
    for tif in paths:
        row = run_tile(tif, processor, model, device, tile_size, overlap, args.min_area)
        rows.append(row)
        shp = OUT / f"{row['tile']}_buildings.shp"
        if shp.is_file():
            frames.append(gpd.read_file(shp))

    if frames:
        merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
        merged["id"] = range(1, len(merged) + 1)
        merged_path = OUT / "Fort_Riley_oem_buildings_3tiles.shp"
        merged.to_file(merged_path, driver="ESRI Shapefile")
        log(f"merged features={len(merged)} {merged_path}")

    summary = OUT / "summary.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
