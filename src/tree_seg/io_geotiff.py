"""GeoTIFF I/O, tiling windows, and georeferenced raster writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window, transform as window_transform


@dataclass(frozen=True)
class TileSpec:
    """A single tile window inside a GeoTIFF."""

    row_off: int
    col_off: int
    height: int
    width: int
    pad_top: int = 0
    pad_left: int = 0
    pad_bottom: int = 0
    pad_right: int = 0

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)

    @property
    def out_height(self) -> int:
        return self.height + self.pad_top + self.pad_bottom

    @property
    def out_width(self) -> int:
        return self.width + self.pad_left + self.pad_right


def open_rgb_geotiff(path: str | Path):
    """Open a GeoTIFF dataset (caller must close / use as context manager)."""
    return rasterio.open(path)


def read_rgb(dataset: rasterio.DatasetReader, window: Window | None = None) -> np.ndarray:
    """
    Read RGB as uint8 HxWx3.

    Uses the first three bands. Scales 16-bit / float data into 0-255.
    """
    indexes = list(range(1, min(3, dataset.count) + 1))
    while len(indexes) < 3:
        indexes.append(indexes[-1])

    arr = dataset.read(indexes, window=window, boundless=True, fill_value=0)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    return _to_uint8(arr)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    # Common 0-1 float or 0-65535 uint16
    vmax = float(np.nanpercentile(arr[finite], 99.5))
    vmin = float(np.nanpercentile(arr[finite], 0.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[~finite] = 0.0
    return (scaled * 255.0).astype(np.uint8)


def iter_tile_specs(
    height: int,
    width: int,
    tile_size: int = 1024,
    overlap: float = 0.25,
) -> Iterator[TileSpec]:
    """Yield TileSpecs covering the raster with the given overlap fraction."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(tile_size * (1.0 - overlap)))

    row_offs = list(range(0, max(height - tile_size, 0) + 1, stride))
    col_offs = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not row_offs or row_offs[-1] + tile_size < height:
        row_offs.append(max(0, height - tile_size))
    if not col_offs or col_offs[-1] + tile_size < width:
        col_offs.append(max(0, width - tile_size))

    # Deduplicate while preserving order
    seen: set[tuple[int, int]] = set()
    for row_off in row_offs:
        for col_off in col_offs:
            key = (row_off, col_off)
            if key in seen:
                continue
            seen.add(key)

            h = min(tile_size, height - row_off)
            w = min(tile_size, width - col_off)
            pad_bottom = tile_size - h
            pad_right = tile_size - w
            yield TileSpec(
                row_off=row_off,
                col_off=col_off,
                height=h,
                width=w,
                pad_bottom=pad_bottom,
                pad_right=pad_right,
            )


def read_tile_rgb(
    dataset: rasterio.DatasetReader,
    spec: TileSpec,
    tile_size: int,
) -> np.ndarray:
    """Read a tile and pad to tile_size x tile_size."""
    rgb = read_rgb(dataset, spec.window)
    if rgb.shape[0] == tile_size and rgb.shape[1] == tile_size:
        return rgb
    out = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    out[: rgb.shape[0], : rgb.shape[1]] = rgb
    return out


def write_geotiff(
    path: str | Path,
    data: np.ndarray,
    transform: Affine,
    crs,
    *,
    nodata: float | None = None,
    dtype: str | None = None,
    compress: str = "deflate",
) -> None:
    """
    Write a single-band or multi-band GeoTIFF.

    data: HxW or CxHxW
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        count = 1
        height, width = data.shape
        bands = data[np.newaxis, ...]
    elif data.ndim == 3:
        count, height, width = data.shape
        bands = data
    else:
        raise ValueError("data must be HxW or CxHxW")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype or str(bands.dtype),
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands.astype(profile["dtype"]))


def pixel_size_m(dataset: rasterio.DatasetReader) -> float:
    """Approximate ground sample distance in metres from affine transform."""
    transform = dataset.transform
    return float(max(abs(transform.a), abs(transform.e)))


def window_geotransform(dataset: rasterio.DatasetReader, window: Window) -> Affine:
    return window_transform(window, dataset.transform)


def save_tile_png(path: str | Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    elif mask.max() <= 1:
        mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


def load_mask_png(path: str | Path) -> np.ndarray:
    from PIL import Image

    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)
