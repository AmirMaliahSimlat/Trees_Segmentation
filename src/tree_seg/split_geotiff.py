"""Split a large GeoTIFF into smaller georeferenced tiles (streaming, low RAM)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform
from tqdm import tqdm


def _is_mostly_empty(data: np.ndarray, nodata: float | None, empty_frac: float) -> bool:
    """True if fraction of nodata / all-zero pixels exceeds ``empty_frac``."""
    if data.size == 0:
        return True
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mask = np.all(data == nodata, axis=0) if data.ndim == 3 else data == nodata
    else:
        mask = np.all(data == 0, axis=0) if data.ndim == 3 else data == 0
    return float(mask.mean()) >= empty_frac


def split_geotiff(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 10240,
    overlap: int = 0,
    compress: str = "deflate",
    skip_empty: bool = True,
    empty_frac: float = 0.99,
    prefix: str | None = None,
) -> dict[str, Any]:
    """
    Cut ``input_path`` into ``tile_size``×``tile_size`` GeoTIFF tiles.

    Reads and writes window-by-window so a multi‑GB orthomosaic does not need
    to fit in RAM. Returns a summary dict (also written as ``manifest.json``).
    """
    if tile_size < 64:
        raise ValueError("tile_size must be >= 64")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or input_path.stem
    stride = tile_size - overlap

    written: list[dict[str, Any]] = []
    skipped = 0

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": tile_size,
                "width": tile_size,
                "tiled": True,
                "blockxsize": min(256, tile_size),
                "blockysize": min(256, tile_size),
                "compress": compress,
            }
        )
        nodata = src.nodata

        row_offs = list(range(0, max(height - tile_size, 0) + 1, stride))
        col_offs = list(range(0, max(width - tile_size, 0) + 1, stride))
        if not row_offs or row_offs[-1] + tile_size < height:
            row_offs.append(max(0, height - tile_size))
        if not col_offs or col_offs[-1] + tile_size < width:
            col_offs.append(max(0, width - tile_size))

        # Deduplicate edge tiles
        seen: set[tuple[int, int]] = set()
        specs: list[tuple[int, int, int, int]] = []
        for r in row_offs:
            for c in col_offs:
                key = (r, c)
                if key in seen:
                    continue
                seen.add(key)
                h = min(tile_size, height - r)
                w = min(tile_size, width - c)
                specs.append((r, c, h, w))

        for r, c, h, w in tqdm(specs, desc=f"Split {input_path.name}", unit="tile"):
            window = Window(c, r, w, h)
            data = src.read(window=window)
            if skip_empty and _is_mostly_empty(data, nodata, empty_frac):
                skipped += 1
                continue

            name = f"{prefix}_r{r:05d}_c{c:05d}.tif"
            out_path = output_dir / name
            tile_profile = profile.copy()
            tile_profile.update(
                {
                    "height": h,
                    "width": w,
                    "transform": window_transform(window, src.transform),
                    "blockxsize": min(256, w),
                    "blockysize": min(256, h),
                }
            )
            # rasterio requires block sizes to be multiples of 16 when tiled; fall back if tiny
            if tile_profile["blockxsize"] < 16 or tile_profile["blockysize"] < 16:
                tile_profile["tiled"] = False
                tile_profile.pop("blockxsize", None)
                tile_profile.pop("blockysize", None)
            if src.crs is not None:
                tile_profile["crs"] = src.crs
            if nodata is not None:
                tile_profile["nodata"] = nodata

            with rasterio.open(out_path, "w", **tile_profile) as dst:
                dst.write(data)
                for i in range(1, src.count + 1):
                    tags = src.tags(i)
                    if tags:
                        dst.update_tags(i, **tags)

            written.append(
                {
                    "file": name,
                    "row_off": r,
                    "col_off": c,
                    "height": h,
                    "width": w,
                }
            )

    summary = {
        "source": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "tile_size": tile_size,
        "overlap": overlap,
        "source_height": height,
        "source_width": width,
        "tiles_written": len(written),
        "tiles_skipped_empty": skipped,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tiles": written,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
