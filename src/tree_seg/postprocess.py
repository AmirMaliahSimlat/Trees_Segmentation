"""Post-process canopy masks and export tree-area footprint polygons (shapefile)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape

from tree_seg.io_geotiff import write_geotiff


def morphological_cleanup(
    mask: np.ndarray,
    *,
    open_px: int = 3,
    close_px: int = 5,
) -> np.ndarray:
    import cv2

    out = (mask > 0).astype(np.uint8) * 255
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return (out > 0).astype(np.uint8)


def mask_to_polygons(
    mask: np.ndarray,
    transform,
    *,
    min_area_m2: float = 5.0,
    pixel_size_m: float | None = None,
) -> gpd.GeoDataFrame:
    """Polygonize binary mask; filter by approximate ground area in m^2."""
    mask_bool = mask.astype(bool)
    gsd = pixel_size_m if pixel_size_m is not None else float(max(abs(transform.a), abs(transform.e)))
    px_area_m2 = gsd * gsd
    transform_px_area = max(abs(transform.a * transform.e), 1e-12)

    geoms = []
    areas = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask_bool, transform=transform):
        if int(val) == 0:
            continue
        poly = shape(geom)
        pixel_count = float(poly.area) / transform_px_area
        area_m2 = pixel_count * px_area_m2
        if area_m2 < min_area_m2:
            continue
        geoms.append(poly)
        areas.append(area_m2)

    gdf = gpd.GeoDataFrame({"id": list(range(1, len(geoms) + 1)), "area_m2": areas, "geometry": geoms})
    return gdf


def export_polygons(
    mask_or_proba_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 0.5,
    min_area_m2: float = 5.0,
    morph_open_px: int = 3,
    morph_close_px: int = 5,
    native_gsd_m: float | None = None,
    write_clean_mask: bool = True,
    write_geojson: bool = False,
) -> dict[str, Path]:
    """
    From a probability or mask GeoTIFF, write tree-area footprint polygons as a shapefile.

    Primary deliverable: ``*_tree_footprints.shp`` (+ sidecar files).
    Optionally also writes a cleaned mask GeoTIFF and GeoJSON.
    """
    mask_or_proba_path = Path(mask_or_proba_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        mask_or_proba_path.stem.replace("_tree_proba", "")
        .replace("_tree_mask_clean", "")
        .replace("_tree_mask", "")
    )

    with rasterio.open(mask_or_proba_path) as ds:
        data = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        gsd = native_gsd_m if native_gsd_m is not None else float(max(abs(transform.a), abs(transform.e)))

        if np.issubdtype(data.dtype, np.floating):
            mask = (data >= threshold).astype(np.uint8)
        else:
            mask = (data > 0).astype(np.uint8)
            if data.max() > 1:
                mask = (data > 127).astype(np.uint8)

        mask = morphological_cleanup(mask, open_px=morph_open_px, close_px=morph_close_px)

        paths: dict[str, Path] = {}
        if write_clean_mask:
            clean_path = output_dir / f"{stem}_tree_mask_clean.tif"
            write_geotiff(clean_path, mask, transform, crs, nodata=0, dtype="uint8")
            paths["mask_clean"] = clean_path

        gdf = mask_to_polygons(mask, transform, min_area_m2=min_area_m2, pixel_size_m=gsd)
        if crs is not None:
            gdf = gdf.set_crs(crs)
        if len(gdf) == 0:
            gdf = gpd.GeoDataFrame({"id": [], "area_m2": [], "geometry": []}, crs=crs)

        shp_path = output_dir / f"{stem}_tree_footprints.shp"
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        paths["shapefile"] = shp_path

        if write_geojson:
            geojson_path = output_dir / f"{stem}_tree_footprints.geojson"
            gdf.to_file(geojson_path, driver="GeoJSON")
            paths["geojson"] = geojson_path

    return paths


# Backwards-compatible alias
export_for_unreal = export_polygons


def export_from_config(mask_path: str | Path, output_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Path]:
    p = cfg.get("postprocess", {})
    pred = cfg.get("predict", {})
    return export_polygons(
        mask_path,
        output_dir,
        threshold=float(p.get("threshold", pred.get("threshold", 0.5))),
        min_area_m2=float(p.get("min_area_m2", 5.0)),
        morph_open_px=int(p.get("morph_open_px", 3)),
        morph_close_px=int(p.get("morph_close_px", 5)),
        native_gsd_m=pred.get("native_gsd_m"),
        write_clean_mask=bool(p.get("write_clean_mask", True)),
        write_geojson=bool(p.get("write_geojson", False)),
    )
