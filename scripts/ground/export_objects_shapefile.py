#!/usr/bin/env python
"""Polygonize *_objects.tif masks and merge into one shapefile.

Does not touch tree / orchard / roof checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge object-mask GeoTIFFs into one shapefile")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Folder of *_objects.tif")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output .shp path")
    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Pixel size in metres (required for geographic CRS, e.g. Nablus 0.24)",
    )
    parser.add_argument("--min-area", type=float, default=5.0, help="Drop polygons smaller than this m^2")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from tqdm import tqdm

    from tree_seg.postprocess import mask_to_polygons

    mask_dir = args.input
    masks = sorted(mask_dir.glob("*_objects.tif"))
    if not masks:
        masks = sorted((mask_dir / "objects").glob("*_objects.tif"))
    if not masks:
        raise FileNotFoundError(f"No *_objects.tif in {mask_dir}")

    frames: list[gpd.GeoDataFrame] = []
    for path in tqdm(masks, desc="polygonize"):
        with rasterio.open(path) as ds:
            data = ds.read(1)
            mask = (data > 0).astype("uint8")
            crs = ds.crs
            geographic = crs is not None and getattr(crs, "is_geographic", False)
            if geographic:
                gsd = args.gsd if args.gsd is not None else 0.24
            else:
                gsd = args.gsd if args.gsd is not None else float(max(abs(ds.transform.a), abs(ds.transform.e)))
            gdf = mask_to_polygons(mask, ds.transform, min_area_m2=args.min_area, pixel_size_m=gsd)
            if crs is not None:
                gdf = gdf.set_crs(crs)
        if gdf.empty:
            continue
        gdf["tile_id"] = path.stem.replace("_objects", "")
        frames.append(gdf)

    if not frames:
        raise ValueError(f"No polygons from masks in {mask_dir}")

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    merged["id"] = range(1, len(merged) + 1)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(out, driver="ESRI Shapefile")
    print(f"tiles={len(masks)} features={len(merged)} wrote={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
