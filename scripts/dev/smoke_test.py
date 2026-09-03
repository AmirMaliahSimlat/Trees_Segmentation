"""Create a small synthetic GeoTIFF and exercise IO / postprocess / metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from tree_seg.io_geotiff import iter_tile_specs, open_rgb_geotiff, read_tile_rgb, write_geotiff
from tree_seg.metrics import binary_confusion
from tree_seg.postprocess import export_polygons


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)


def make_synthetic_ortho(path: Path, size: int = 768, gsd: float = 0.3) -> Path:
    """RGB ortho with green 'grass' and darker green circular 'trees'."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    # grass-like background
    rgb[..., 0] = 40
    rgb[..., 1] = 140
    rgb[..., 2] = 40
    # dirt patch
    rgb[100:200, 100:250] = (120, 90, 60)
    # tree blobs
    yy, xx = np.ogrid[:size, :size]
    for cy, cx, r in [(200, 400, 60), (500, 200, 80), (550, 550, 50)]:
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        rgb[disk] = (20, 90, 25)

    transform = from_origin(500000.0, 3500000.0, gsd, gsd)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32636",
        "transform": transform,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.transpose(rgb, (2, 0, 1)))
    return path


def make_gt_mask(path: Path, size: int = 768, gsd: float = 0.3) -> Path:
    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    for cy, cx, r in [(200, 400, 60), (500, 200, 80), (550, 550, 50)]:
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        mask[disk] = 1
    transform = from_origin(500000.0, 3500000.0, gsd, gsd)
    write_geotiff(path, mask, transform, "EPSG:32636", nodata=0, dtype="uint8")
    return path


def main() -> None:
    ortho = make_synthetic_ortho(OUT / "synthetic_ortho.tif")
    gt = make_gt_mask(OUT / "synthetic_gt_mask.tif")

    with open_rgb_geotiff(ortho) as ds:
        specs = list(iter_tile_specs(ds.height, ds.width, tile_size=256, overlap=0.25))
        assert len(specs) >= 4, specs
        tile = read_tile_rgb(ds, specs[0], 256)
        assert tile.shape == (256, 256, 3), tile.shape

    with rasterio.open(gt) as ds:
        gt_mask = ds.read(1)

    # Pretend model output = GT with a bit of grass FP
    noisy = gt_mask.copy()
    noisy[120:180, 120:220] = 1
    write_geotiff(
        OUT / "synthetic_tree_mask.tif",
        noisy,
        from_origin(500000.0, 3500000.0, 0.3, 0.3),
        "EPSG:32636",
        nodata=0,
        dtype="uint8",
    )

    paths = export_polygons(
        OUT / "synthetic_tree_mask.tif",
        OUT / "footprints",
        min_area_m2=2.0,
        native_gsd_m=0.3,
    )
    m = binary_confusion(noisy, gt_mask)
    print("tiles:", len(specs))
    print("metrics on noisy mask:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    print("export:", {k: str(v) for k, v in paths.items()})
    print("SMOKE_OK", ortho)


if __name__ == "__main__":
    main()
