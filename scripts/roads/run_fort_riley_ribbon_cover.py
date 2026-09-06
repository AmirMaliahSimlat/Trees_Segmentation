"""Road ribbons with not-road seed skip + object-cleaned remeasure."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tree_seg.config import load_config
from tree_seg.ground_fill import load_landcover, make_filler
from tree_seg.hf_auth import ensure_hf_auth
from tree_seg.io_geotiff import open_rgb_geotiff, pixel_size_m, read_rgb, write_geotiff
from tree_seg.road_ribbon import CorridorCleaner, RibbonParams, rasterize_ribbons, ribbons_for_gdf
from fort_riley_ribbon_runs import EDIT as SHP, RUNS, STEM

LOG: Path | None = None


def _build_cleaner() -> CorridorCleaner:
    ensure_hf_auth(verbose=True)
    cfg = load_config(ROOT / "configs" / "ground_fill.yaml")
    m = cfg.get("model", {})
    inp = cfg.get("inpaint", {})
    processor, model, _id2label, object_ids, device = load_landcover(m.get("hf_id"))
    filler = make_filler(inp, device)
    return CorridorCleaner(
        processor=processor,
        model=model,
        object_ids=object_ids,
        device=device,
        filler=filler,
        tile_size=int(m.get("tile_size", 512)),
        overlap=float(m.get("overlap", 0.125)),
    )


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
    candidates = [path]
    for extra in ("_new", "_run", "_run2"):
        candidates.append(path.with_name(path.stem + extra + path.suffix))
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


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dyn-width",
        action="store_true",
        help="let a long stretch of a road override the road's single width (off by default)",
    )
    return ap.parse_args()


def main() -> int:
    global LOG
    args = _args()
    params = RibbonParams(dyn_width=args.dyn_width)
    if not SHP.is_file():
        print(f"missing edited shapefile {SHP}", flush=True)
        return 1
    rc = 0
    for run in RUNS:
        LOG = run["log_cover"]
        img: Path = run["img"]
        out: Path = run["out"]
        if not img.is_file():
            log(f"[{run['name']}] missing imagery {img}")
            rc = 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        log(f"[{run['name']}] loading {img.name}")
        with open_rgb_geotiff(img) as ds:
            rgb = read_rgb(ds)
            transform = ds.transform
            crs = ds.crs
            gsd = pixel_size_m(ds)
            bounds = box(*ds.bounds)
            h, w = rgb.shape[:2]
        log(f"[{run['name']}] rgb {w}x{h} gsd={gsd:.3f}m")
        log(f"[{run['name']}] dynamic width {'on' if params.dyn_width else 'off'}")
        gdf = gpd.read_file(SHP).to_crs(crs)
        gdf = gdf[gdf.intersects(bounds)].copy()
        log(f"[{run['name']}] edited lines {len(gdf)}")
        cleaner = _build_cleaner()
        log(f"[{run['name']}] corridor cleaner device={cleaner.device} backend={getattr(cleaner.filler, 'backend', '?')}")
        ribbons = ribbons_for_gdf(gdf, rgb, transform, gsd, params, cleaner=cleaner)
        n_clean = int((ribbons["method"] == "clean").sum()) if len(ribbons) else 0
        n_skip = int((ribbons["method"] == "skip").sum()) if len(ribbons) else 0
        n_junc = int((ribbons["method"] == "junction").sum()) if len(ribbons) else 0
        wider = 0
        if len(ribbons):
            wn = ribbons["w_naive"].astype(float)
            wf = ribbons["width_m"].astype(float)
            wider = int((wf > wn + 0.44).sum())
        log(
            f"[{run['name']}] ribbons {len(ribbons)}  method skip={n_skip} clean={n_clean} "
            f"junction={n_junc}  "
            f"wider than naive={wider}  median W={ribbons['width_m'].median() if len(ribbons) else 'n/a'}"
        )
        if params.dyn_width and len(ribbons) and "n_seg" in ribbons.columns:
            seg = ribbons.loc[ribbons["method"].isin(["skip", "naive", "clean"]), ["n_seg", "w_min", "w_max"]].astype(float)
            multi = seg[seg["n_seg"] > 1]
            log(
                f"[{run['name']}] dynamic width: {len(multi)}/{len(seg)} lines have >1 segment  "
                f"median segments={seg['n_seg'].median()}  max segments={seg['n_seg'].max():.0f}  "
                f"widest stretch={seg['w_max'].max():.1f} m"
            )
        shp = _write_shp(ribbons, out / f"{STEM}_ribbons.shp")
        if len(ribbons) and "method" in ribbons.columns:
            astro = ribbons[ribbons["method"] == "junction"].copy()
            if len(astro):
                ash = _write_shp(astro, out / f"{STEM}_astroids.shp")
                log(f"[{run['name']}] astroids {len(astro)} wrote {ash}")
        cols = [c for c in ("name", "fclass", "length_m", "w_naive", "w_skip", "w_clean", "width_m", "w_min", "w_max", "n_seg", "skip_pct", "method") if c in ribbons.columns]
        if len(ribbons) and cols:
            ribbons[cols].to_csv(out / f"{STEM}_widths.csv", index=False)
        mask = rasterize_ribbons(ribbons, (h, w), transform)
        write_geotiff(out / f"{STEM}_mask.tif", mask.astype(np.uint8), transform, crs, dtype="uint8")
        vis = _overlay(rgb, mask, gdf, transform)
        _save_jpg(out / f"{STEM}_overlay.jpg", vis, max_side=2560)
        log(f"[{run['name']}] wrote {shp}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
