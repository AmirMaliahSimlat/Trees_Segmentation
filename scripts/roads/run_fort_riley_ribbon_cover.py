"""Road ribbons with not-road seed skip + object-cleaned remeasure."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tree_seg.io_geotiff import open_rgb_geotiff, pixel_size_m, read_rgb, write_geotiff
from tree_seg.road_ribbon import RibbonParams, rasterize_ribbons, ribbons_for_gdf

STEM = "FortRiley_r61440_c81920"
IMG = ROOT / "outputs" / "ground_fill" / "Fort_Riley" / "bare" / f"{STEM}_bare.tif"
SHP = ROOT / "outputs" / "footprints" / "Fort_Riley" / "roads_edit" / f"{STEM}_edit.shp"
OUT = ROOT / "outputs" / "footprints" / "Fort_Riley" / "ribbons without parking cleaned"
LOG = ROOT / "outputs" / "logs" / "roads_ribbon_cleaned.log"


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


def _overlay(rgb: np.ndarray, mask: np.ndarray, lines: gpd.GeoDataFrame, transform) -> np.ndarray:
    vis = rgb.copy()
    tint = vis.astype(np.float32)
    hit = mask > 0
    tint[hit, 0] = tint[hit, 0] * 0.45 + 255 * 0.55
    tint[hit, 1] = tint[hit, 1] * 0.55 + 80 * 0.45
    tint[hit, 2] = tint[hit, 2] * 0.75
    vis = np.clip(tint, 0, 255).astype(np.uint8)
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    vis[edge > 0] = (255, 230, 40)
    inv = ~transform
    for geom in lines.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            pts = []
            for x, y in part.coords:
                c, r = inv * (x, y)
                pts.append((int(round(c)), int(round(r))))
            if len(pts) >= 2:
                cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
    return vis


def _save_jpg(path: Path, rgb: np.ndarray, max_side: int | None = None) -> None:
    im = Image.fromarray(rgb)
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=88)


def _world_to_px(x: float, y: float, transform) -> tuple[int, int]:
    c, r = ~transform * (x, y)
    return int(round(c)), int(round(r))


def _line_window(line, transform, shape, size: int = 960) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    mid = line.interpolate(0.5, normalized=True)
    cx, cy = _world_to_px(float(mid.x), float(mid.y), transform)
    x0 = int(np.clip(cx - size // 2, 0, max(0, w - size)))
    y0 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    return x0, y0, x1, y1


def _draw_geom(vis: np.ndarray, geom, transform, box_xy: tuple[int, int, int, int], color, width: int) -> None:
    if geom is None or geom.is_empty:
        return
    c0, r0, _, _ = box_xy
    parts = geom.geoms if geom.geom_type in ("MultiPolygon", "MultiLineString") else [geom]
    closed = geom.geom_type in ("Polygon", "MultiPolygon")
    for part in parts:
        closed = part.geom_type == "Polygon"
        coords_list = [part.exterior.coords] if closed else [part.coords]
        for coords in coords_list:
            pts = []
            for x, y in coords:
                c, r = _world_to_px(float(x), float(y), transform)
                pts.append((c - c0, r - r0))
            if len(pts) >= 2:
                cv2.polylines(vis, [np.array(pts, dtype=np.int32)], closed, color, width, cv2.LINE_AA)


def _caption(rgb: np.ndarray, text: str) -> np.ndarray:
    bar = Image.new("RGB", (rgb.shape[1], 36), (18, 18, 18))
    ImageDraw.Draw(bar).text((8, 10), text, fill=(230, 230, 230))
    canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + 36))
    canvas.paste(bar, (0, 0))
    canvas.paste(Image.fromarray(rgb), (0, 36))
    return np.asarray(canvas)


def _hstack(imgs: list[np.ndarray]) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    w = sum(im.shape[1] for im in imgs)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    x = 0
    for im in imgs:
        out[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1]
    return out


def _gain_idea1(ex: dict) -> float:
    wn, ws = ex.get("w_naive"), ex.get("w_skip")
    if wn is None or ws is None:
        return -1.0
    gain = float(ws) - float(wn)
    if gain < 0.45:
        return -1.0
    if float(ws) > 14.0 or float(wn) > 12.0:
        return gain * 0.05
    thin = max(0.0, 6.0 - float(wn))
    return gain + thin


def _gain_idea2(ex: dict) -> float:
    wc = ex.get("w_clean")
    if wc is None:
        return -1.0
    base = ex.get("w_skip")
    if base is None:
        base = ex.get("w_naive")
    if base is None:
        return -1.0
    gain = float(wc) - float(base)
    if gain < 0.45:
        return -1.0
    if float(wc) > 12.0 or float(base) > 8.0:
        return gain * 0.08
    thin = max(0.0, 6.0 - float(base))
    return gain + thin


def _pick_diverse(items: list[dict], key, n: int = 6, min_dist: float = 90.0) -> list[dict]:
    ordered = sorted(items, key=key, reverse=True)
    picked: list[dict] = []
    for it in ordered:
        line = it.get("line")
        if line is None or line.is_empty:
            continue
        mid = line.interpolate(0.5, normalized=True)
        if any(mid.distance(p["line"].interpolate(0.5, normalized=True)) < min_dist for p in picked):
            continue
        picked.append(it)
        if len(picked) >= n:
            return picked
    for it in ordered:
        if it in picked:
            continue
        picked.append(it)
        if len(picked) >= n:
            break
    return picked


def _write_idea1(ex: dict, rgb: np.ndarray, transform, path: Path) -> None:
    line = ex["line"]
    box_xy = _line_window(line, transform, rgb.shape, size=900)
    x0, y0, x1, y1 = box_xy
    left = rgb[y0:y1, x0:x1].copy()
    right = rgb[y0:y1, x0:x1].copy()
    _draw_geom(left, ex.get("naive_geom"), transform, box_xy, (255, 80, 80), 2)
    _draw_geom(left, line, transform, box_xy, (0, 220, 255), 1)
    _draw_geom(right, ex.get("skip_geom"), transform, box_xy, (255, 170, 40), 2)
    _draw_geom(right, line, transform, box_xy, (0, 220, 255), 1)
    left = _caption(left, f"naive W={ex.get('w_naive')} m  (red)")
    right = _caption(right, f"skip-not-road W={ex.get('w_skip')} m  (orange)  +{_gain_idea1(ex):.1f} m")
    _save_jpg(path, _hstack([left, right]))


def _write_idea2(ex: dict, rgb: np.ndarray, transform, stem: Path) -> None:
    crops = ex.get("crops") or []
    if crops:
        crop = crops[0]
        raw = crop["rgb"]
        cleaned = crop["clean"]
        box_xy = crop["box"]
    else:
        box_xy = _line_window(ex["line"], transform, rgb.shape, size=900)
        x0, y0, x1, y1 = box_xy
        raw = rgb[y0:y1, x0:x1].copy()
        cleaned = raw
        box_xy = (x0, y0, x1, y1)
    _save_jpg(stem.with_name(stem.stem + "_crop.jpg"), raw, max_side=1400)
    _save_jpg(stem.with_name(stem.stem + "_clean.jpg"), cleaned, max_side=1400)
    a = raw.copy()
    b = cleaned.copy()
    c = cleaned.copy()
    _draw_geom(a, ex.get("line"), transform, box_xy, (0, 220, 255), 1)
    _draw_geom(b, ex.get("line"), transform, box_xy, (0, 220, 255), 1)
    _draw_geom(c, ex.get("skip_geom"), transform, box_xy, (80, 180, 255), 2)
    _draw_geom(c, ex.get("clean_geom"), transform, box_xy, (255, 170, 40), 2)
    _draw_geom(c, ex.get("line"), transform, box_xy, (0, 220, 255), 1)
    a = _caption(a, "original crop")
    b = _caption(b, "object-cleaned (temporary)")
    c = _caption(
        c,
        f"skip W={ex.get('w_skip')}  clean W={ex.get('w_clean')} m  cyan=line  blue=before  orange=after",
    )
    _save_jpg(stem.with_suffix(".jpg"), _hstack([a, b, c]))


def main() -> int:
    if not IMG.is_file():
        log(f"missing imagery {IMG}")
        return 1
    if not SHP.is_file():
        log(f"missing edited shapefile {SHP}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"loading {IMG.name}")
    with open_rgb_geotiff(IMG) as ds:
        rgb = read_rgb(ds)
        transform = ds.transform
        crs = ds.crs
        gsd = pixel_size_m(ds)
        bounds = box(*ds.bounds)
        h, w = rgb.shape[:2]
    log(f"rgb {w}x{h} gsd={gsd:.3f}m  (cleaned/bare)")
    gdf = gpd.read_file(SHP).to_crs(crs)
    gdf = gdf[gdf.intersects(bounds)].copy()
    log(f"edited lines {len(gdf)}")
    ribbons = ribbons_for_gdf(gdf, rgb, transform, gsd, RibbonParams(), cleaner=None)
    n_clean = int((ribbons["method"] == "clean").sum()) if len(ribbons) else 0
    n_skip = int((ribbons["method"] == "skip").sum()) if len(ribbons) else 0
    wider = 0
    if len(ribbons):
        wn = ribbons["w_naive"].astype(float)
        wf = ribbons["width_m"].astype(float)
        wider = int((wf > wn + 0.44).sum())
    log(
        f"ribbons {len(ribbons)}  method skip={n_skip} clean={n_clean}  "
        f"wider than naive={wider}  median W={ribbons['width_m'].median() if len(ribbons) else 'n/a'}"
    )
    shp = _write_shp(ribbons, OUT / f"{STEM}_ribbons.shp")
    cols = [c for c in ("name", "fclass", "length_m", "w_naive", "w_skip", "w_clean", "width_m", "skip_pct", "method") if c in ribbons.columns]
    if len(ribbons) and cols:
        ribbons[cols].to_csv(OUT / f"{STEM}_widths.csv", index=False)
    mask = rasterize_ribbons(ribbons, (h, w), transform)
    write_geotiff(OUT / f"{STEM}_mask.tif", mask.astype(np.uint8), transform, crs, dtype="uint8")
    vis = _overlay(rgb, mask, gdf, transform)
    _save_jpg(OUT / f"{STEM}_overlay.jpg", vis, max_side=2560)
    log(f"wrote {shp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
