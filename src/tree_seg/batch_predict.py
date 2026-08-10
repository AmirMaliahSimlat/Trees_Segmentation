"""Batch prediction over a folder of tile GeoTIFFs into a review session."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm

from tree_seg.io_geotiff import open_rgb_geotiff, read_rgb, write_geotiff
from tree_seg.model import SegFormerBundle, predict_tile_proba
from tree_seg.predict import predict_geotiff
from tree_seg.review_store import ReviewSession


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}


def list_input_images(input_dir: str | Path, recursive: bool = True) -> list[Path]:
    input_dir = Path(input_dir)
    paths: list[Path] = []
    pattern_iter: Iterable[Path]
    if recursive:
        pattern_iter = input_dir.rglob("*")
    else:
        pattern_iter = input_dir.glob("*")
    for p in pattern_iter:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return sorted(paths)


def _read_any_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with open_rgb_geotiff(path) as ds:
            return read_rgb(ds)
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def _downscale_rgb(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 0.999:
        return rgb, 1.0
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA), scale


def _predict_rgb_tiled(
    bundle: SegFormerBundle,
    rgb: np.ndarray,
    *,
    tile_size: int = 1024,
    overlap: float = 0.25,
    threshold: float = 0.5,
) -> np.ndarray:
    """Soft-stitch prediction on an in-memory RGB array; returns binary mask."""
    from tree_seg.io_geotiff import iter_tile_specs

    h, w = rgb.shape[:2]
    proba_acc = np.zeros((h, w), dtype=np.float64)
    weight_acc = np.zeros((h, w), dtype=np.float64)
    for spec in iter_tile_specs(h, w, tile_size=tile_size, overlap=overlap):
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        patch = rgb[spec.row_off : spec.row_off + spec.height, spec.col_off : spec.col_off + spec.width]
        tile[: patch.shape[0], : patch.shape[1]] = patch
        proba = predict_tile_proba(bundle, tile, target_size=(tile_size, tile_size))
        hh, ww = spec.height, spec.width
        r0, c0 = spec.row_off, spec.col_off
        proba_acc[r0 : r0 + hh, c0 : c0 + ww] += proba[:hh, :ww]
        weight_acc[r0 : r0 + hh, c0 : c0 + ww] += 1.0
    proba = (proba_acc / np.maximum(weight_acc, 1e-8)).astype(np.float32)
    return (proba >= threshold).astype(np.uint8)


def predict_tile_to_session(
    path: Path,
    session: ReviewSession,
    bundle: SegFormerBundle,
    *,
    pred_dir: Path,
    cfg: dict[str, Any],
    threshold: float = 0.5,
    preview_max_side: int = 2048,
    full_resolution: bool = False,
) -> None:
    """
    Predict one image into the review session.

    For large GeoTIFFs, default is preview-resolution inference (fast, UI-friendly).
    Set ``full_resolution=True`` for full tiled GeoTIFF inference (slow on CPU).
    """
    tile_id = path.stem
    pred_dir.mkdir(parents=True, exist_ok=True)
    pcfg = cfg.get("predict", {})
    tile_size = int(pcfg.get("tile_size", 1024))
    overlap = float(pcfg.get("overlap", 0.25))

    if path.suffix.lower() in {".tif", ".tiff"} and full_resolution:
        paths = predict_geotiff(
            path,
            pred_dir,
            bundle,
            tile_size=tile_size,
            overlap=overlap,
            threshold=threshold,
            match_train_gsd=bool(pcfg.get("match_train_gsd", False)),
            train_gsd_m=float(pcfg.get("train_gsd_m", 0.10)),
            native_gsd_m=pcfg.get("native_gsd_m"),
            stem=tile_id,
        )
        with open_rgb_geotiff(path) as ds:
            rgb_full = read_rgb(ds)
            transform = ds.transform
            crs = ds.crs
        import rasterio

        with rasterio.open(paths["mask"]) as ds:
            mask_full = ds.read(1).astype(np.uint8)
        rgb, _ = _downscale_rgb(rgb_full, preview_max_side)
        mask = cv2.resize(mask_full, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        session.upsert_item(tile_id, path, rgb, mask)
        return

    # Preview-resolution path (default)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with open_rgb_geotiff(path) as ds:
            rgb_full = read_rgb(ds)
            transform = ds.transform
            crs = ds.crs
            full_h, full_w = ds.height, ds.width
    else:
        rgb_full = _read_any_rgb(path)
        transform = crs = None
        full_h, full_w = rgb_full.shape[:2]

    rgb, _ = _downscale_rgb(rgb_full, preview_max_side)
    mask = _predict_rgb_tiled(
        bundle,
        rgb,
        tile_size=min(tile_size, max(rgb.shape[0], rgb.shape[1])),
        overlap=overlap,
        threshold=threshold,
    )

    # Upsample mask to full resolution for GeoTIFF export when source is georeferenced
    if transform is not None:
        mask_full = cv2.resize(mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
        out_mask = pred_dir / f"{tile_id}_tree_mask.tif"
        write_geotiff(out_mask, mask_full.astype(np.uint8), transform, crs, nodata=0, dtype="uint8")
    else:
        from tree_seg.io_geotiff import save_mask_png

        save_mask_png(pred_dir / f"{tile_id}_tree_mask.png", mask)

    session.upsert_item(tile_id, path, rgb, mask)


def batch_predict_folder(
    input_dir: str | Path,
    session_dir: str | Path,
    bundle: SegFormerBundle,
    cfg: dict[str, Any],
    *,
    pred_dir: str | Path | None = None,
    recursive: bool = True,
    skip_existing: bool = True,
    preview_max_side: int = 2048,
    full_resolution: bool = False,
) -> ReviewSession:
    """
    Run prediction on every image under ``input_dir`` and build/update a review session.
    """
    input_dir = Path(input_dir)
    session = ReviewSession(session_dir)
    pred_dir = Path(pred_dir or (Path(session_dir) / "geotiff_masks"))
    threshold = float(cfg.get("predict", {}).get("threshold", 0.5))

    files = list_input_images(input_dir, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    existing_ids = {i.tile_id for i in session.items} if skip_existing else set()

    for path in tqdm(files, desc="Batch predict", unit="img"):
        if skip_existing and path.stem in existing_ids and (session.masks_dir / f"{path.stem}.png").exists():
            continue
        predict_tile_to_session(
            path,
            session,
            bundle,
            pred_dir=pred_dir,
            cfg=cfg,
            threshold=threshold,
            preview_max_side=preview_max_side,
            full_resolution=full_resolution,
        )

    session.save()
    return session
