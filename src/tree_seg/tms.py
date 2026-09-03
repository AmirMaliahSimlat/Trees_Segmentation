"""OSGeo TMS (global-geodetic) helpers: GeoTIFF mosaic <-> PNG tiles + TMS.xml."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from tqdm import tqdm


@dataclass(frozen=True)
class TmsMap:
    """Parsed TileMap metadata."""

    path: Path
    title: str
    srs: str
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    origin: tuple[float, float]
    tile_width: int
    tile_height: int
    extension: str


def parse_tms_xml(path: str | Path) -> TmsMap:
    path = Path(path)
    root = ET.parse(path).getroot()
    bbox_el = root.find("BoundingBox")
    origin_el = root.find("Origin")
    fmt = root.find("TileFormat")
    if bbox_el is None or origin_el is None or fmt is None:
        raise ValueError(f"Incomplete TMS.xml: {path}")

    return TmsMap(
        path=path,
        title=(root.findtext("Title") or path.parent.name).strip(),
        srs=(root.findtext("SRS") or "EPSG:4326").strip(),
        bbox=(
            float(bbox_el.attrib["minx"]),
            float(bbox_el.attrib["miny"]),
            float(bbox_el.attrib["maxx"]),
            float(bbox_el.attrib["maxy"]),
        ),
        origin=(float(origin_el.attrib["x"]), float(origin_el.attrib["y"])),
        tile_width=int(fmt.attrib.get("width", 256)),
        tile_height=int(fmt.attrib.get("height", 256)),
        extension=(fmt.attrib.get("extension") or "png").lstrip("."),
    )


def tms_resolution_deg(zoom: int, tile_size: int = 256) -> float:
    """Degrees per pixel for OSGeo TMS global-geodetic (180° / tile / 2^z)."""
    return 180.0 / float(tile_size) / float(2**zoom)


def tile_bounds_geodetic(
    zoom: int,
    x: int,
    y: int,
    *,
    tile_size: int = 256,
    origin: tuple[float, float] = (-180.0, -90.0),
) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in EPSG:4326 for TMS tile z/x/y."""
    res = tms_resolution_deg(zoom, tile_size)
    minx = origin[0] + x * tile_size * res
    miny = origin[1] + y * tile_size * res
    return minx, miny, minx + tile_size * res, miny + tile_size * res


def list_tms_png_tiles(tiles_root: str | Path, zoom: int) -> list[tuple[int, int, Path]]:
    """List ``(x, y, path)`` for PNG tiles under ``tiles_root/{zoom}/{x}/{y}.png``."""
    zoom_dir = Path(tiles_root) / str(zoom)
    if not zoom_dir.is_dir():
        raise FileNotFoundError(f"Zoom folder not found: {zoom_dir}")
    tiles: list[tuple[int, int, Path]] = []
    for x_dir in sorted(zoom_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not x_dir.is_dir() or not x_dir.name.isdigit():
            continue
        x = int(x_dir.name)
        for png in x_dir.glob("*.png"):
            if not png.stem.isdigit():
                continue
            tiles.append((x, int(png.stem), png))
    if not tiles:
        raise FileNotFoundError(f"No PNG tiles under {zoom_dir}")
    return tiles


def mosaic_tms_pngs_to_geotiff(
    tiles_root: str | Path,
    output_path: str | Path,
    *,
    zoom: int,
    tms_xml: str | Path | None = None,
    compress: str = "deflate",
) -> dict[str, Any]:
    """
    Mosaic TMS PNG tiles at ``zoom`` into one north-up GeoTIFF (EPSG:4326).

    Missing tiles stay nodata (0). Uses OSGeo global-geodetic indexing.
    """
    tiles_root = Path(tiles_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tms = parse_tms_xml(tms_xml or (tiles_root / "tms.xml"))
    tiles = list_tms_png_tiles(tiles_root, zoom)
    xs = sorted({x for x, _, _ in tiles})
    ys = sorted({y for _, y, _ in tiles})
    min_x, max_x = xs[0], xs[-1]
    min_y, max_y = ys[0], ys[-1]
    tw, th = tms.tile_width, tms.tile_height

    width = (max_x - min_x + 1) * tw
    height = (max_y - min_y + 1) * th
    west, south, _, _ = tile_bounds_geodetic(zoom, min_x, min_y, tile_size=tw, origin=tms.origin)
    _, _, east, north = tile_bounds_geodetic(zoom, max_x, max_y, tile_size=tw, origin=tms.origin)
    transform = from_bounds(west, south, east, north, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": tms.srs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": compress,
        # Do NOT set nodata=0: pure black is valid in shadows/water and QGIS
        # would render those pixels as transparent (white canvas speckles).
        "photometric": "RGB",
    }

    tile_index = {(x, y): path for x, y, path in tiles}
    blank = np.zeros((3, th, tw), dtype=np.uint8)

    with rasterio.open(output_path, "w", **profile) as dst:
        for x in tqdm(xs, desc=f"Mosaic z{zoom}", unit="col"):
            for y in ys:
                col = (x - min_x) * tw
                row = (max_y - y) * th
                window = rasterio.windows.Window(col, row, tw, th)
                path = tile_index.get((x, y))
                if path is None:
                    dst.write(blank, window=window)
                    continue
                img = Image.open(path).convert("RGBA")
                arr = np.asarray(img)
                rgb = arr[:, :, :3].copy()
                alpha = arr[:, :, 3]
                rgb[alpha == 0] = 0
                dst.write(np.transpose(rgb, (2, 0, 1)), window=window)

    res_deg = tms_resolution_deg(zoom, tw)
    mid_lat = 0.5 * (south + north)
    gsd_m = float(res_deg * 111_320.0 * abs(np.cos(np.deg2rad(mid_lat))))

    summary = {
        "source": str(tiles_root.resolve()),
        "tms_xml": str((tms_xml or tiles_root / "tms.xml")),
        "output": str(output_path.resolve()),
        "zoom": zoom,
        "crs": tms.srs,
        "width": width,
        "height": height,
        "tiles_used": len(tiles),
        "x_range": [min_x, max_x],
        "y_range": [min_y, max_y],
        "bounds": [west, south, east, north],
        "approx_gsd_m": gsd_m,
    }
    return summary


def iter_tms_blocks(
    tiles: list[tuple[int, int, Path]],
    *,
    block_tiles: int = 40,
) -> list[dict[str, Any]]:
    """
    Group TMS tiles into square blocks of ``block_tiles`` x ``block_tiles`` cells.

    Only blocks that contain at least one PNG are returned.
    """
    if block_tiles < 1:
        raise ValueError("block_tiles must be >= 1")
    xs = sorted({x for x, _, _ in tiles})
    ys = sorted({y for _, y, _ in tiles})
    min_x, max_x = xs[0], xs[-1]
    min_y, max_y = ys[0], ys[-1]
    tile_set = {(x, y) for x, y, _ in tiles}

    blocks: list[dict[str, Any]] = []
    x0 = min_x
    while x0 <= max_x:
        y0 = min_y
        while y0 <= max_y:
            x1 = min(x0 + block_tiles - 1, max_x)
            y1 = min(y0 + block_tiles - 1, max_y)
            n = sum(
                1
                for x in range(x0, x1 + 1)
                for y in range(y0, y1 + 1)
                if (x, y) in tile_set
            )
            if n > 0:
                blocks.append(
                    {
                        "name": f"x{x0}_y{y0}",
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "n_tiles": n,
                    }
                )
            y0 += block_tiles
        x0 += block_tiles
    return blocks


def mosaic_tms_block_to_geotiff(
    tiles_root: str | Path,
    output_path: str | Path,
    *,
    zoom: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    tms_xml: str | Path | None = None,
    compress: str = "deflate",
) -> dict[str, Any]:
    """Mosaic one TMS block [x0..x1] x [y0..y1] inclusive to a GeoTIFF (EPSG:4326)."""
    tiles_root = Path(tiles_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tms = parse_tms_xml(tms_xml or (tiles_root / "tms.xml"))
    tw, th = tms.tile_width, tms.tile_height
    zoom_dir = tiles_root / str(zoom)

    width = (x1 - x0 + 1) * tw
    height = (y1 - y0 + 1) * th
    west, south, _, _ = tile_bounds_geodetic(zoom, x0, y0, tile_size=tw, origin=tms.origin)
    _, _, east, north = tile_bounds_geodetic(zoom, x1, y1, tile_size=tw, origin=tms.origin)
    transform = from_bounds(west, south, east, north, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": tms.srs,
        "transform": transform,
        "tiled": True,
        "blockxsize": min(256, tw),
        "blockysize": min(256, th),
        "compress": compress,
        "photometric": "RGB",
    }
    blank = np.zeros((3, th, tw), dtype=np.uint8)
    used = 0

    with rasterio.open(output_path, "w", **profile) as dst:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                col = (x - x0) * tw
                row = (y1 - y) * th  # north-up
                window = rasterio.windows.Window(col, row, tw, th)
                path = zoom_dir / str(x) / f"{y}.png"
                if not path.is_file():
                    dst.write(blank, window=window)
                    continue
                img = Image.open(path).convert("RGBA")
                arr = np.asarray(img)
                rgb = arr[:, :, :3].copy()
                alpha = arr[:, :, 3]
                rgb[alpha == 0] = 0
                dst.write(np.transpose(rgb, (2, 0, 1)), window=window)
                used += 1

    return {
        "output": str(output_path.resolve()),
        "zoom": zoom,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "width": width,
        "height": height,
        "tiles_used": used,
        "bounds": [west, south, east, north],
    }


def warp_geotiff_to_utm(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dst_crs: str = "EPSG:32614",
    compress: str = "deflate",
) -> Path:
    """Reproject a GeoTIFF to a metric CRS (default UTM 14N for Grand Forks ND)."""
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": compress,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )
        # Drop nodata so valid black pixels are not treated as empty in QGIS
        profile.pop("nodata", None)
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    return output_path


def _tile_range(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    *,
    tile_size: int = 256,
    origin: tuple[float, float] = (-180.0, -90.0),
) -> tuple[int, int, int, int]:
    res = tms_resolution_deg(zoom, tile_size)
    span = tile_size * res
    x0 = int(np.floor((west - origin[0]) / span))
    y0 = int(np.floor((south - origin[1]) / span))
    x1 = int(np.ceil((east - origin[0]) / span)) - 1
    y1 = int(np.ceil((north - origin[1]) / span)) - 1
    return x0, y0, x1, y1


def _write_tms_xml(
    path: Path,
    *,
    title: str,
    bbox: tuple[float, float, float, float],
    zooms: list[int],
    tile_size: int = 256,
) -> None:
    west, south, east, north = bbox
    sets = []
    for z in zooms:
        upp = 180.0 / float(tile_size) / float(2**z)
        sets.append(
            f'    <TileSet href="{z}" order="{z}" units-per-pixel="{upp:.10f}" />'
        )
    xml = (
        '<?xml version="1.0" ?>\n'
        '<TileMap tilemapservice="http://tms.osgeo.org/1.0.0" version="1.0.0">\n'
        f"  <Title>{title}</Title>\n"
        "  <abstract></abstract>\n"
        "  <SRS>EPSG:4326</SRS>\n"
        "  <vsrs></vsrs>\n"
        f'  <BoundingBox maxx="{east:.7f}" maxy="{north:.7f}" minx="{west:.7f}" miny="{south:.7f}" />\n'
        '  <Origin x="-180.0000000" y="-90.0000000" />\n'
        f'  <TileFormat extension="png" height="{tile_size}" mime-type="image/png" width="{tile_size}" />\n'
        '  <TileSets profile="global-geodetic">\n'
        + "\n".join(sets)
        + "\n"
        "  </TileSets>\n"
        "  <dataextents>\n"
        f'    <dataextent maxlevel="{max(zooms)}" maxx="{east:.7f}" maxy="{north:.7f}" '
        f'minlevel="{min(zooms)}" minx="{west:.7f}" miny="{south:.7f}" />\n'
        "  </dataextents>\n"
        "</TileMap>\n"
    )
    path.write_text(xml, encoding="utf-8")


def _bbox_intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _native_zoom(
    bounds4326: tuple[float, float, float, float],
    pixel_size: float,
    crs_epsg: int | None,
    *,
    tile_size: int = 256,
) -> int:
    west, south, east, north = bounds4326
    mid_lat = 0.5 * (south + north)
    native_deg = abs(pixel_size) / (111_320.0 * max(abs(np.cos(np.deg2rad(mid_lat))), 0.2))
    if crs_epsg == 4326:
        native_deg = abs(pixel_size)
    z_native = int(round(np.log2((180.0 / tile_size) / max(native_deg, 1e-12))))
    return int(np.clip(z_native, 0, 22))


def _iter_tms_blocks(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    block_tiles: int,
) -> list[tuple[int, int, int, int]]:
    blocks: list[tuple[int, int, int, int]] = []
    x = x0
    while x <= x1:
        y = y0
        while y <= y1:
            blocks.append((x, y, min(x + block_tiles - 1, x1), min(y + block_tiles - 1, y1)))
            y += block_tiles
        x += block_tiles
    return blocks


def _save_tms_png(path: Path, rgba: np.ndarray) -> None:
    """Write a TMS PNG. Drop alpha when the tile is fully opaque (much smaller)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = rgba[:, :, 3]
    if int(alpha.min()) == 255:
        img = Image.fromarray(rgba[:, :, :3], "RGB")
    else:
        img = Image.fromarray(rgba, "RGBA")
    img.save(path, format="PNG", optimize=False, compress_level=6)


def _write_zmax_block(
    sources: list[tuple[Path, tuple[float, float, float, float]]],
    output_dir: Path,
    *,
    zmax: int,
    bx0: int,
    by0: int,
    bx1: int,
    by1: int,
    tile_size: int,
    skip_existing: bool,
) -> int:
    """Warp overlapping GeoTIFFs into one TMS block and write zmax PNGs."""
    origin = (-180.0, -90.0)
    needed: list[tuple[int, int, Path]] = []
    for x in range(bx0, bx1 + 1):
        for y in range(by0, by1 + 1):
            out_path = output_dir / str(zmax) / str(x) / f"{y}.png"
            if skip_existing and out_path.is_file():
                continue
            needed.append((x, y, out_path))
    if not needed:
        return 0

    west, south, _, _ = tile_bounds_geodetic(zmax, bx0, by0, tile_size=tile_size, origin=origin)
    _, _, east, north = tile_bounds_geodetic(zmax, bx1, by1, tile_size=tile_size, origin=origin)
    dest_bounds = (west, south, east, north)
    hits = [(path, b) for path, b in sources if _bbox_intersects(b, dest_bounds)]
    if not hits:
        return 0

    width = (bx1 - bx0 + 1) * tile_size
    height = (by1 - by0 + 1) * tile_size
    dst_transform = from_bounds(west, south, east, north, width, height)
    rgb = np.zeros((3, height, width), dtype=np.uint8)
    cover = np.zeros((height, width), dtype=np.uint8)
    tmp = np.zeros((3, height, width), dtype=np.uint8)

    for path, src_bounds in hits:
        with rasterio.open(path) as src:
            tmp.fill(0)
            for b in range(1, min(3, src.count) + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=tmp[b - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear,
                )
        sw, ss, se, sn = src_bounds
        footprint = {
            "type": "Polygon",
            "coordinates": [[(sw, ss), (se, ss), (se, sn), (sw, sn), (sw, ss)]],
        }
        tmp_cover = rasterize(
            [(footprint, 255)],
            out_shape=(height, width),
            transform=dst_transform,
            fill=0,
            dtype="uint8",
        )
        mask = tmp_cover > 0
        if not mask.any():
            continue
        rgb[:, mask] = tmp[:, mask]
        cover[mask] = tmp_cover[mask]

    mosaic = np.transpose(rgb, (1, 2, 0))
    n_written = 0
    for x, y, out_path in needed:
        row = (by1 - y) * tile_size
        col = (x - bx0) * tile_size
        patch = mosaic[row : row + tile_size, col : col + tile_size]
        alpha = cover[row : row + tile_size, col : col + tile_size]
        if int(alpha.max()) == 0:
            continue
        _save_tms_png(out_path, np.dstack([patch, alpha]))
        n_written += 1
    return n_written


def _write_pyramid_tile(
    output_dir: Path,
    z: int,
    x: int,
    y: int,
    tile_size: int,
) -> tuple[int, int] | None:
    ts = tile_size
    canvas = np.zeros((ts * 2, ts * 2, 4), dtype=np.uint8)
    children = (
        (0, 0, 2 * x, 2 * y + 1),
        (0, ts, 2 * x + 1, 2 * y + 1),
        (ts, 0, 2 * x, 2 * y),
        (ts, ts, 2 * x + 1, 2 * y),
    )
    found = False
    for rr, cc, cx, cy in children:
        child = output_dir / str(z + 1) / str(cx) / f"{cy}.png"
        if not child.is_file():
            continue
        canvas[rr : rr + ts, cc : cc + ts] = np.asarray(Image.open(child).convert("RGBA"))
        found = True
    if not found:
        return None
    img = Image.fromarray(canvas, "RGBA").resize((ts, ts), Image.Resampling.BILINEAR)
    _save_tms_png(output_dir / str(z) / str(x) / f"{y}.png", np.asarray(img))
    return (x, y)


def _build_tms_pyramid(
    output_dir: Path,
    *,
    zmax: int,
    zmin: int,
    tile_size: int,
    workers: int = 8,
) -> int:
    """Build zmax-1 .. zmin from 2x2 children. Always overwrites lower zooms."""
    zdir = output_dir / str(zmax)
    if not zdir.is_dir():
        return 0
    current: list[tuple[int, int]] = []
    x_dirs = [p for p in zdir.iterdir() if p.is_dir() and p.name.isdigit()]
    for x_dir in tqdm(x_dirs, desc=f"TMS index z{zmax}", leave=False):
        x = int(x_dir.name)
        for png in x_dir.glob("*.png"):
            if png.stem.isdigit():
                current.append((x, int(png.stem)))
    if not current:
        return 0

    n_written = 0
    workers = max(1, int(workers))
    for z in range(zmax - 1, zmin - 1, -1):
        parents = sorted({(x // 2, y // 2) for x, y in current})
        next_level: list[tuple[int, int]] = []

        def _job(xy: tuple[int, int]) -> tuple[int, int] | None:
            return _write_pyramid_tile(output_dir, z, xy[0], xy[1], tile_size)

        if workers == 1:
            iterator = (_job(xy) for xy in parents)
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            iterator = pool.map(_job, parents)
        try:
            for result in tqdm(iterator, total=len(parents), desc=f"TMS z{z}"):
                if result is not None:
                    next_level.append(result)
                    n_written += 1
        finally:
            if workers != 1:
                pool.shutdown(wait=True)
        current = next_level
        if not current:
            break
    return n_written


def export_geotiffs_to_tms(
    inputs: list[str | Path] | str | Path,
    output_dir: str | Path,
    *,
    title: str | None = None,
    tile_size: int = 256,
    min_zoom: int | None = None,
    max_zoom: int | None = None,
    skip_existing: bool = True,
    block_tiles: int = 16,
    workers: int = 1,
) -> dict[str, Any]:
    """Warp one or more GeoTIFFs to OSGeo TMS (global-geodetic, EPSG:4326).

    Streams 256 px tiles in blocks so a full map does not need to fit in RAM.
    Writes ``{output_dir}/{z}/{x}/{y}.png`` and ``TMS.xml`` (same layout as GFK).
    """
    if isinstance(inputs, (str, Path)):
        paths = [Path(inputs)]
    else:
        paths = [Path(p) for p in inputs]
    if not paths:
        raise FileNotFoundError("No GeoTIFFs to export")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    origin = (-180.0, -90.0)

    sources: list[tuple[Path, tuple[float, float, float, float]]] = []
    west = south = east = north = None
    zooms_native: list[int] = []
    for path in paths:
        with rasterio.open(path) as src:
            b = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            epsg = src.crs.to_epsg() if src.crs else None
            zooms_native.append(_native_zoom(b, abs(src.transform.a), epsg, tile_size=tile_size))
        sources.append((path, b))
        if west is None:
            west, south, east, north = b
        else:
            west = min(west, b[0])
            south = min(south, b[1])
            east = max(east, b[2])
            north = max(north, b[3])
    assert west is not None and south is not None and east is not None and north is not None

    z_native = int(round(float(np.median(zooms_native))))
    zmax = int(max_zoom) if max_zoom is not None else z_native
    zmin = int(min_zoom) if min_zoom is not None else 0
    zmin = max(0, min(zmin, zmax))
    x0, y0, x1, y1 = _tile_range(west, south, east, north, zmax, tile_size=tile_size, origin=origin)
    blocks = _iter_tms_blocks(x0, y0, x1, y1, block_tiles)
    workers = max(1, int(workers))

    def _job(block: tuple[int, int, int, int]) -> int:
        bx0, by0, bx1, by1 = block
        return _write_zmax_block(
            sources,
            output_dir,
            zmax=zmax,
            bx0=bx0,
            by0=by0,
            bx1=bx1,
            by1=by1,
            tile_size=tile_size,
            skip_existing=skip_existing,
        )

    n_zmax = 0
    if workers == 1:
        for block in tqdm(blocks, desc=f"TMS z{zmax}"):
            n_zmax += _job(block)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n in tqdm(pool.map(_job, blocks), total=len(blocks), desc=f"TMS z{zmax}"):
                n_zmax += n

    n_pyr = _build_tms_pyramid(
        output_dir, zmax=zmax, zmin=zmin, tile_size=tile_size, workers=max(8, workers)
    )
    xml_path = output_dir / "TMS.xml"
    _write_tms_xml(
        xml_path,
        title=title or paths[0].stem,
        bbox=(west, south, east, north),
        zooms=list(range(zmin, zmax + 1)),
        tile_size=tile_size,
    )
    return {
        "sources": [str(p.resolve()) for p in paths],
        "n_sources": len(paths),
        "output": str(output_dir.resolve()),
        "tms_xml": str(xml_path.resolve()),
        "crs": "EPSG:4326",
        "bbox": [west, south, east, north],
        "min_zoom": zmin,
        "max_zoom": zmax,
        "zmax_tiles": [x0, y0, x1, y1],
        "png_tiles": n_zmax + n_pyr,
        "png_zmax": n_zmax,
        "png_pyramid": n_pyr,
    }


def export_geotiff_to_tms(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 256,
    min_zoom: int | None = None,
    max_zoom: int | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Warp a GeoTIFF to OSGeo TMS (global-geodetic, EPSG:4326) with a zoom pyramid."""
    input_path = Path(input_path)
    result = export_geotiffs_to_tms(
        [input_path],
        output_dir,
        title=input_path.stem,
        tile_size=tile_size,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        skip_existing=skip_existing,
        workers=1,
    )
    result["source"] = str(input_path.resolve())
    return result

