"""Tiled GeoTIFF inference with overlap-averaged probability stitching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from tree_seg.io_geotiff import (
    iter_tile_specs,
    open_rgb_geotiff,
    pixel_size_m,
    read_tile_rgb,
    write_geotiff,
)
from tree_seg.model import SegFormerBundle, predict_tile_proba


def _resize_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)


def _resize_proba(proba: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    if proba.shape[0] == h and proba.shape[1] == w:
        return proba
    return cv2.resize(proba, (w, h), interpolation=cv2.INTER_LINEAR)


def predict_geotiff(
    input_path: str | Path,
    output_dir: str | Path,
    bundle: SegFormerBundle,
    *,
    tile_size: int = 1024,
    overlap: float = 0.25,
    threshold: float = 0.5,
    match_train_gsd: bool = False,
    train_gsd_m: float = 0.10,
    native_gsd_m: float | None = None,
    stem: str | None = None,
    write_proba: bool = True,
) -> dict[str, Path]:
    """
    Run tiled inference and write probability + binary mask GeoTIFFs.

    Returns paths for ``proba`` and ``mask``.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or input_path.stem

    with open_rgb_geotiff(input_path) as ds:
        height, width = ds.height, ds.width
        transform = ds.transform
        crs = ds.crs
        gsd = native_gsd_m if native_gsd_m is not None else pixel_size_m(ds)

        scale = 1.0
        if match_train_gsd and gsd > 0:
            # Upsample toward pretrained GSD (e.g. 0.30 -> 0.10 => scale 3)
            scale = float(gsd / train_gsd_m)

        proba_acc = np.zeros((height, width), dtype=np.float64)
        weight_acc = np.zeros((height, width), dtype=np.float64)

        specs = list(iter_tile_specs(height, width, tile_size=tile_size, overlap=overlap))
        for spec in tqdm(specs, desc=f"Predict {input_path.name}", unit="tile"):
            rgb = read_tile_rgb(ds, spec, tile_size)
            if match_train_gsd and abs(scale - 1.0) > 1e-6:
                rgb_in = _resize_rgb(rgb, scale)
                proba_tile = predict_tile_proba(bundle, rgb_in, target_size=(tile_size, tile_size))
                # predict_tile_proba already resized to target_size
            else:
                proba_tile = predict_tile_proba(bundle, rgb, target_size=(tile_size, tile_size))

            # Only accumulate the valid (non-padded) region
            h, w = spec.height, spec.width
            tile_valid = proba_tile[:h, :w]
            r0, c0 = spec.row_off, spec.col_off
            proba_acc[r0 : r0 + h, c0 : c0 + w] += tile_valid
            weight_acc[r0 : r0 + h, c0 : c0 + w] += 1.0

        weight_acc = np.maximum(weight_acc, 1e-8)
        proba = (proba_acc / weight_acc).astype(np.float32)
        mask = (proba >= threshold).astype(np.uint8)

        paths: dict[str, Path] = {}
        mask_path = output_dir / f"{stem}_tree_mask.tif"
        write_geotiff(mask_path, mask, transform, crs, nodata=0, dtype="uint8")
        paths["mask"] = mask_path

        if write_proba:
            proba_path = output_dir / f"{stem}_tree_proba.tif"
            write_geotiff(proba_path, proba, transform, crs, nodata=None, dtype="float32")
            paths["proba"] = proba_path

    return paths


def predict_from_config(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: dict[str, Any],
    bundle: SegFormerBundle,
) -> dict[str, Path]:
    p = cfg.get("predict", {})
    return predict_geotiff(
        input_path,
        output_dir,
        bundle,
        tile_size=int(p.get("tile_size", 1024)),
        overlap=float(p.get("overlap", 0.25)),
        threshold=float(p.get("threshold", 0.5)),
        match_train_gsd=bool(p.get("match_train_gsd", False)),
        train_gsd_m=float(p.get("train_gsd_m", 0.10)),
        native_gsd_m=p.get("native_gsd_m"),
        write_proba=bool(p.get("write_proba", True)),
    )
