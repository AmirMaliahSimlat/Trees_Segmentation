"""Union touching cover-run road ribbons into one polygon per connected piece."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tree_seg.io_geotiff import open_rgb_geotiff, read_rgb, write_geotiff
from tree_seg.road_ribbon import merge_touching_ribbons, rasterize_ribbons
from fort_riley_ribbon_runs import EDIT, RUNS, STEM

LOG: Path | None = None


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    if LOG is None:
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_shp(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [path] + [path.with_name(f"{path.stem}_v{i}{path.suffix}") for i in range(2, 21)]
    last_err: PermissionError | None = None
    for dest in candidates:
        try:
            gdf.to_file(dest, driver="ESRI Shapefile")
            if dest != path:
                log(f"shapefile locked; wrote {dest.name}")
            return dest
        except PermissionError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
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
    global LOG
    rc = 0
    for run in RUNS:
        LOG = run["log_union"]
        img: Path = run["img"]
        out: Path = run["out"]
        src_candidates = [
            out / f"{STEM}_ribbons.shp",
            out / f"{STEM}_ribbons_new.shp",
            out / f"{STEM}_ribbons_run.shp",
            out / f"{STEM}_ribbons_run2.shp",
        ]
        existing = [p for p in src_candidates if p.is_file()]
        if not existing:
            log(f"[{run['name']}] missing {src_candidates[0]}")
            rc = 1
            continue
        src = max(existing, key=lambda p: p.stat().st_mtime)
        out.mkdir(parents=True, exist_ok=True)
        ribbons = gpd.read_file(src)
        log(f"[{run['name']}] input ribbons {len(ribbons)}")
        merged = merge_touching_ribbons(ribbons, snap_m=0.35)
        n_junc = int((ribbons["method"] == "junction").sum()) if "method" in ribbons.columns else 0
        log(f"[{run['name']}] merged (junction astroid corners: {n_junc})")
        log(f"[{run['name']}] merged polygons {len(merged)}  (connected components)")
        if len(merged):
            log(merged[["n_rib", "kind", "width_m", "area_m2"]].describe(include="all").to_string())
        shp = _write_shp(merged, out / f"{STEM}_union.shp")
        if "method" in ribbons.columns:
            astro = ribbons[ribbons["method"] == "junction"].copy()
            if len(astro):
                ash = _write_shp(astro, out / f"{STEM}_astroids.shp")
                astro_u = merge_touching_ribbons(astro, snap_m=0.35)
                aush = _write_shp(astro_u, out / f"{STEM}_astroids_union.shp")
                log(f"[{run['name']}] astroids {len(astro)} wrote {ash.name}; union {len(astro_u)} wrote {aush.name}")
        log(f"[{run['name']}] loading {img.name}")
        with open_rgb_geotiff(img) as ds:
            rgb = read_rgb(ds)
            transform = ds.transform
            crs = ds.crs
            h, w = rgb.shape[:2]
        mask = rasterize_ribbons(merged, (h, w), transform)
        write_geotiff(out / f"{STEM}_union_mask.tif", mask.astype(np.uint8), transform, crs, dtype="uint8")
        lines = gpd.read_file(EDIT).to_crs(crs) if EDIT.is_file() else gpd.GeoDataFrame(geometry=[], crs=crs)
        vis = _overlay(rgb, mask, merged, lines, transform)
        im = Image.fromarray(vis)
        im.thumbnail((2560, 2560), Image.Resampling.BILINEAR)
        im.save(out / f"{STEM}_union_overlay.jpg", quality=88)
        log(f"[{run['name']}] wrote {shp}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
