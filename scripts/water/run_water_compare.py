"""Predict LandCover.ai water SegFormer on the Fort Riley water tile vs OEM.

Does not overwrite water_oem/_preview_water_*.jpg or other task checkpoints.
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
OEM = ROOT / "outputs" / "footprints" / "Fort_Riley" / "water_oem"
OUT = ROOT / "outputs" / "footprints" / "Fort_Riley" / "water_landcoverai"
CKPT = ROOT / "outputs" / "checkpoints" / "water_landcoverai" / "best"
LOG = ROOT / "outputs" / "logs" / "water_landcoverai.log"
TILES = ["FortRiley_r51200_c40960"]
PROTECTED = {"_preview_water_FortRiley_r51200_c40960.jpg"}


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


def _overlay(base: np.ndarray, rings, color: tuple[int, int, int]) -> np.ndarray:
    im = Image.fromarray(base.copy())
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ring in rings:
        draw.polygon(ring, fill=(*color, 90), outline=(*color, 230))
    return np.asarray(Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB"))


def write_preview(rgb, transform, oem, pred, path: Path, size: int = 2048) -> None:
    if path.name in PROTECTED:
        raise RuntimeError(f"refusing to overwrite OEM preview {path.name}")
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    base = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    ds_t = Affine(transform.a * w / nw, transform.b, transform.c, transform.d, transform.e * h / nh, transform.f)
    mid = _overlay(base, _to_pixels(oem, ds_t, nw, nh), (0, 160, 255))
    right = _overlay(base, _to_pixels(pred, ds_t, nw, nh), (220, 40, 40))
    gap = np.full((nh, 8, 3), 30, dtype=np.uint8)
    sheet = np.concatenate([base, gap, mid, gap, right], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=88)
    log(f"preview {path.name} oem={len(oem)} landcoverai={len(pred)}")


def load_oem_mask(stem: str, shape: tuple[int, int]) -> np.ndarray:
    for name in (f"{stem}_water_raw.tif", f"{stem}_water.tif"):
        p = OEM / name
        if p.is_file():
            with rasterio.open(p) as ds:
                return ds.read(1)
    return np.zeros(shape, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--tiles", nargs="+", default=TILES)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from tree_seg.config import load_config
    from tree_seg.hf_auth import ensure_hf_auth
    from tree_seg.io_geotiff import read_rgb, write_geotiff
    from tree_seg.model import load_segformer
    from tree_seg.postprocess import mask_to_polygons
    from tree_seg.predict import predict_geotiff

    ensure_hf_auth(verbose=True)
    cfg = load_config(ROOT / "configs" / "water.yaml")
    m = cfg.get("model", {})
    p = cfg.get("predict", {})
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint {args.checkpoint} — train first")

    bundle = load_segformer(
        str(m.get("name", "nvidia/segformer-b5-finetuned-ade-640-640")),
        num_labels=int(m.get("num_labels", 2)),
        id2label={int(k): v for k, v in m.get("id2label", {0: "background", 1: "water"}).items()},
        local_checkpoint=str(args.checkpoint),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== LandCover.ai water compare ckpt={args.checkpoint} ===")

    rows = []
    for stem in args.tiles:
        tif = IMG_DIR / f"{stem}.tif"
        log(f"predict {stem}")
        paths = predict_geotiff(
            tif,
            OUT,
            bundle,
            tile_size=int(p.get("tile_size", 1024)),
            overlap=float(p.get("overlap", 0.25)),
            threshold=float(p.get("threshold", 0.5)),
            match_train_gsd=False,
            native_gsd_m=0.30,
            stem=stem,
            write_proba=False,
        )
        with rasterio.open(tif) as ds:
            rgb = read_rgb(ds)
            transform = ds.transform
            crs = ds.crs
            gsd = float(max(abs(ds.transform.a), abs(ds.transform.e)))
        with rasterio.open(paths["mask"]) as ds:
            pred_mask = ds.read(1)
        write_geotiff(OUT / f"{stem}_water.tif", pred_mask.astype(np.uint8), transform, crs, nodata=0, dtype="uint8")
        pred = mask_to_polygons(pred_mask, transform, min_area_m2=25.0, pixel_size_m=gsd)
        if crs is not None:
            pred = pred.set_crs(crs)
        if not pred.empty:
            pred["source"] = "landcoverai_segformer"
            pred.to_file(OUT / f"{stem}_water.shp", driver="ESRI Shapefile")
        oem_mask = load_oem_mask(stem, pred_mask.shape)
        oem = mask_to_polygons(oem_mask, transform, min_area_m2=25.0, pixel_size_m=gsd)
        if crs is not None:
            oem = oem.set_crs(crs)
        preview = OUT / f"_preview_water_lcai_vs_oem_{stem}.jpg"
        write_preview(rgb, transform, oem, pred, preview)
        row = {"tile": stem, "n_oem": int(len(oem)), "n_landcoverai": int(len(pred)), "preview": str(preview)}
        rows.append(row)
        log(f"  oem={row['n_oem']} landcoverai={row['n_landcoverai']} {preview.name}")

    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log("wrote LandCover.ai vs OEM water previews (cyan=OEM, red=LandCover.ai)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
