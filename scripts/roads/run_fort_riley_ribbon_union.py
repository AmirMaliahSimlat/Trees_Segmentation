"""Union touching cover-run road ribbons into one polygon per connected piece."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tree_seg.io_geotiff import open_rgb_geotiff, pixel_size_m, read_rgb, write_geotiff
from tree_seg.road_ribbon import merge_touching_ribbons, rasterize_ribbons

STEM = "FortRiley_r61440_c81920"
IMG = ROOT / "outputs" / "ground_fill" / "Fort_Riley" / "bare" / f"{STEM}_bare.tif"
EDIT = ROOT / "outputs" / "footprints" / "Fort_Riley" / "roads_edit" / f"{STEM}_edit.shp"
COVER = ROOT / "outputs" / "footprints" / "Fort_Riley" / "ribbons without parking cleaned"
SRC = COVER / f"{STEM}_ribbons.shp"
OUT = COVER
LOG = ROOT / "outputs" / "logs" / "roads_ribbon_cleaned_union.log"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_shp(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdf.to_file(path, driver="ESRI Shapefile")
    except PermissionError:
        path = path.with_name(path.stem + "_new.shp")
        gdf.to_file(path, driver="ESRI Shapefile")
        log(f"shapefile locked; wrote {path.name}")
    return path


def _overlay(rgb: np.ndarray, mask: np.ndarray, polys: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, transform) -> np.ndarray:
    vis = rgb.copy()
    tint = vis.astype(np.float32)
    hit = mask > 0
    tint[hit, 0] = tint[hit, 0] * 0.45 + 255 * 0.55
    tint[hit, 1] = tint[hit, 1] * 0.55 + 80 * 0.45
    tint[hit, 2] = tint[hit, 2] * 0.75
    vis = np.clip(tint, 0, 255).astype(np.uint8)
    inv = ~transform

    def draw_ring(coords, color, width: int) -> None:
        pts = []
        for x, y in coords:
            c, r = inv * (x, y)
            pts.append((int(round(c)), int(round(r))))
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], True, color, width, cv2.LINE_AA)

    for geom in polys.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            draw_ring(part.exterior.coords, (255, 230, 40), 2)
    for geom in lines.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            pts = [(int(round((inv * (x, y))[0])), int(round((inv * (x, y))[1]))) for x, y in part.coords]
            if len(pts) >= 2:
                cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
    return vis


def main() -> int:
    if not SRC.is_file():
        log(f"missing {SRC}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    ribbons = gpd.read_file(SRC)
    log(f"input ribbons {len(ribbons)}")
    merged = merge_touching_ribbons(ribbons, snap_m=0.35)
    log(f"merged polygons {len(merged)}  (connected components)")
    if len(merged):
        log(merged[["n_rib", "kind", "width_m", "area_m2"]].describe(include="all").to_string())
    shp = _write_shp(merged, OUT / f"{STEM}_union.shp")
    log(f"loading {IMG.name}")
    with open_rgb_geotiff(IMG) as ds:
        rgb = read_rgb(ds)
        transform = ds.transform
        crs = ds.crs
        h, w = rgb.shape[:2]
    mask = rasterize_ribbons(merged, (h, w), transform)
    write_geotiff(OUT / f"{STEM}_union_mask.tif", mask.astype(np.uint8), transform, crs, dtype="uint8")
    lines = gpd.read_file(EDIT).to_crs(crs) if EDIT.is_file() else gpd.GeoDataFrame(geometry=[], crs=crs)
    vis = _overlay(rgb, mask, merged, lines, transform)
    im = Image.fromarray(vis)
    im.thumbnail((2560, 2560), Image.Resampling.BILINEAR)
    im.save(OUT / f"{STEM}_union_overlay.jpg", quality=88)
    windows = [(1539, 7055, 1280, "nbhd"), (1761, 5454, 1280, "long"), (2384, 7722, 1280, "yards")]
    for i, (x0, y0, size, name) in enumerate(windows, start=1):
        x0 = int(np.clip(x0, 0, w - size))
        y0 = int(np.clip(y0, 0, h - size))
        crop = vis[y0 : y0 + size, x0 : x0 + size]
        bar = Image.new("RGB", (crop.shape[1], 28), (18, 18, 18))
        ImageDraw.Draw(bar).text((8, 6), f"union {name}  yellow=merged outline  cyan=edit line", fill=(230, 230, 230))
        canvas = Image.new("RGB", (crop.shape[1], crop.shape[0] + 28))
        canvas.paste(bar, (0, 0))
        canvas.paste(Image.fromarray(crop), (0, 28))
        canvas.save(OUT / f"{STEM}_union_crop{i}.jpg", quality=90)
        log(f"crop {i} {name} x{x0} y{y0}")
    log(f"wrote {shp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
