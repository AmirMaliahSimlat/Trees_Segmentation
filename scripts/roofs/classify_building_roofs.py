#!/usr/bin/env python
"""Classify roof type for each polygon in a building-footprint shapefile.

Crops a buffered window from RGB GeoTIFF tile(s), runs the Bonn-trained
ResNet-50, writes a copy of the shapefile with roof_type + roof_conf.

Does not modify tree/orchard checkpoints.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def _metric_buffer(gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Building shapefile has no CRS")
    if gdf.crs.is_projected:
        out = gdf.copy()
        out.geometry = out.geometry.buffer(buffer_m)
        return out
    utm = gdf.estimate_utm_crs()
    proj = gdf.to_crs(utm)
    proj.geometry = proj.geometry.buffer(buffer_m)
    return proj.to_crs(gdf.crs)


def _load_model(ckpt_path: Path, device):
    import torch
    from torchvision import models, transforms

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = blob["classes"]
    image_size = int(blob.get("image_size", 224))
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, classes, tf


def _list_rasters(raster: Path) -> list[Path]:
    if raster.is_dir():
        tifs = sorted(raster.glob("*.tif")) + sorted(raster.glob("*.tiff"))
        if not tifs:
            raise FileNotFoundError(f"No GeoTIFFs in {raster}")
        return tifs
    if not raster.is_file():
        raise FileNotFoundError(raster)
    return [raster]


def _tile_index(paths: list[Path]) -> list[dict]:
    index = []
    crs = None
    for p in paths:
        with rasterio.open(p) as src:
            if crs is None:
                crs = src.crs
            elif src.crs != crs:
                raise ValueError(f"Mixed tile CRS: {crs} vs {src.crs} ({p.name})")
            index.append({"path": p, "crs": src.crs, "poly": box(*src.bounds)})
    return index


def _assign_tiles(gdf: gpd.GeoDataFrame, index: list[dict]) -> list[int]:
    """Tile index per building, or -1 if none covers the centroid."""
    assigned = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            assigned.append(-1)
            continue
        pt = geom.centroid
        hit = -1
        for i, tile in enumerate(index):
            if tile["poly"].covers(pt) or tile["poly"].intersects(geom):
                hit = i
                if tile["poly"].covers(pt):
                    break
        assigned.append(hit)
    return assigned


def _crop_rgb(src, geom, fill: int = 114) -> np.ndarray | None:
    """Mask to the footprint so yards/roads do not dominate the chip."""
    try:
        data, _ = rio_mask(src, [mapping(geom)], crop=True, filled=True, nodata=0)
    except ValueError:
        return None
    if data.size == 0:
        return None
    if data.shape[0] >= 3:
        rgb = np.transpose(data[:3], (1, 2, 0))
    else:
        rgb = np.transpose(np.repeat(data[:1], 3, axis=0), (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return None
    inside = rgb.reshape(-1, 3).max(axis=1) > 0
    if float(inside.mean()) < 0.05:
        return None
    # Neutral pad to a square so Resize does not squash the roof.
    side = max(h, w, 16)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = rgb
    return canvas


def classify(
    shapefile: Path,
    raster: Path,
    checkpoint: Path,
    output: Path,
    *,
    buffer_m: float = 4.0,
) -> Path:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, tf = _load_model(checkpoint, device)

    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise ValueError(f"No features in {shapefile}")

    paths = _list_rasters(raster)
    index = _tile_index(paths)
    tile_crs = index[0]["crs"]
    if gdf.crs and tile_crs and gdf.crs != tile_crs:
        gdf_work = gdf.to_crs(tile_crs)
    else:
        gdf_work = gdf.copy()

    buffered = _metric_buffer(gdf_work, buffer_m)
    assigned = _assign_tiles(gdf_work, index)

    labels = ["unknown"] * len(gdf_work)
    confs = [0.0] * len(gdf_work)

    by_tile: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(assigned):
        by_tile[t].append(i)

    n_miss = len(by_tile.pop(-1, []))
    print(
        f"device={device} buildings={len(gdf_work)} tiles={len(paths)} "
        f"uncovered={n_miss}",
        flush=True,
    )

    for tile_i, rows in tqdm(by_tile.items(), desc="tiles"):
        with rasterio.open(index[tile_i]["path"]) as src:
            for i in rows:
                geom = buffered.geometry.iloc[i]
                if geom is None or geom.is_empty:
                    continue
                rgb = _crop_rgb(src, geom)
                if rgb is None:
                    continue
                tensor = tf(rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    prob = torch.softmax(logits, 1)[0]
                    idx = int(prob.argmax())
                labels[i] = classes[idx]
                confs[i] = float(prob[idx])

    out = gdf.copy()
    if "roof_type" in out.columns:
        out = out.rename(columns={"roof_type": "roof_src"})
    out["roof_type"] = labels
    out["roof_conf"] = confs
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(output, driver="ESRI Shapefile")
    summary = {
        "n": len(out),
        "uncovered": n_miss,
        "counts": out["roof_type"].value_counts().to_dict(),
        "output": str(output),
        "classes": classes,
        "device": str(device),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapefile",
        type=Path,
        required=True,
        help="Building footprints .shp (drop under data/<map>/Buildings/)",
    )
    parser.add_argument(
        "--raster",
        type=Path,
        required=True,
        help="RGB GeoTIFF, or a folder of GeoTIFF tiles covering the buildings",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "checkpoints" / "roof_types" / "best.pt",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=1.5,
        help="Small footprint buffer (m) to absorb offset; keep low so yards are masked out",
    )
    args = parser.parse_args()
    classify(args.shapefile, args.raster, args.checkpoint, args.output, buffer_m=args.buffer_m)


if __name__ == "__main__":
    main()
