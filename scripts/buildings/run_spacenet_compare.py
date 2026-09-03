"""Predict SpaceNet-trained buildings on the 3 frozen Fort Riley tiles.

Writes new previews (original | OEM baseline | SpaceNet). Does not overwrite
outputs/footprints/Fort_Riley/buildings_oem/_preview_*.jpg or baseline_v1/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "Fort_Riley" / "Imagery"
OEM_BASE = ROOT / "outputs" / "footprints" / "Fort_Riley" / "buildings_oem" / "baseline_v1"
OUT = ROOT / "outputs" / "footprints" / "Fort_Riley" / "buildings_spacenet"
CKPT = ROOT / "outputs" / "checkpoints" / "building_footprints" / "best"
LOG = ROOT / "outputs" / "logs" / "spacenet_buildings.log"
TILES = [
    "FortRiley_r81920_c61440",
    "FortRiley_r20480_c120032",
    "FortRiley_r10240_c120032",
]


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts" / "buildings"))
    from tree_seg.config import load_config
    from tree_seg.hf_auth import ensure_hf_auth
    from tree_seg.io_geotiff import read_rgb
    from tree_seg.model import load_segformer
    from tree_seg.postprocess import mask_to_polygons
    from tree_seg.predict import predict_geotiff

    from run_oem_buildings import write_preview

    ensure_hf_auth(verbose=True)
    cfg = load_config(ROOT / "configs" / "building_footprints.yaml")
    m = cfg.get("model", {})
    p = cfg.get("predict", {})
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint {args.checkpoint} — train first")

    bundle = load_segformer(
        str(m.get("name", "nvidia/segformer-b5-finetuned-ade-640-640")),
        num_labels=int(m.get("num_labels", 2)),
        id2label={int(k): v for k, v in m.get("id2label", {0: "background", 1: "building"}).items()},
        local_checkpoint=str(args.checkpoint),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== SpaceNet compare ckpt={args.checkpoint} ===")

    rows = []
    for stem in TILES:
        tif = IMG_DIR / f"{stem}.tif"
        oem_shp = OEM_BASE / f"{stem}_buildings.shp"
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
        mask_path = paths["mask"]
        # predict_geotiff writes *_tree_mask.tif; rename conceptually in shp name only
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1)
            transform = ds.transform
            crs = ds.crs
            gsd = float(max(abs(ds.transform.a), abs(ds.transform.e)))
            rgb = read_rgb(ds) if ds.count >= 3 else None
        if rgb is None:
            with rasterio.open(tif) as ds:
                rgb = read_rgb(ds)
                transform = ds.transform
                crs = ds.crs
        gdf = mask_to_polygons(mask, transform, min_area_m2=15.0, pixel_size_m=gsd)
        if crs is not None:
            gdf = gdf.set_crs(crs)
        gdf["tile_id"] = stem
        gdf["source"] = "spacenet_segformer"
        shp = OUT / f"{stem}_buildings.shp"
        if not gdf.empty:
            gdf.to_file(shp, driver="ESRI Shapefile")
        oem = gpd.read_file(oem_shp) if oem_shp.is_file() else gpd.GeoDataFrame(geometry=[], crs=crs)
        preview = OUT / f"_preview_spacenet_vs_oem_{stem}.jpg"
        if preview.name.startswith("_preview_FortRiley_"):
            raise RuntimeError("refusing to write OEM baseline preview name")
        write_preview(rgb, transform, oem, gdf, preview)
        row = {
            "tile": stem,
            "n_spacenet": int(len(gdf)),
            "n_oem": int(len(oem)),
            "preview": str(preview),
        }
        rows.append(row)
        log(f"  spacenet={row['n_spacenet']} oem={row['n_oem']} {preview.name}")

    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log("wrote SpaceNet vs OEM previews (green=OEM baseline, red=SpaceNet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
