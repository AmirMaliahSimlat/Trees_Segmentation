"""Re-clip OSM + TIGER roads to the full Fort Riley imagery extent.

The previous shapefiles stopped near lat 39.049, so the southern ~2.5 tile rows
had imagery but no roads in the editor.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, box

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "Fort_Riley" / "Imagery"
ROADS_DIR = ROOT / "data" / "Fort_Riley" / "Roads"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
GEOFABRIK_ROADS = "https://download.geofabrik.de/north-america/us/kansas-latest-free.shp.zip"
TIGER_URLS = {
    "Riley": "https://www2.census.gov/geo/tiger/TIGER2024/ROADS/tl_2024_20161_roads.zip",
    "Geary": "https://www2.census.gov/geo/tiger/TIGER2024/ROADS/tl_2024_20061_roads.zip",
}
HEADERS = {"User-Agent": "tree-seg-road-editor/0.1 (local GIS clip)"}


def imagery_bounds_wgs84() -> tuple[float, float, float, float]:
    minx = miny = 1e30
    maxx = maxy = -1e30
    crs = None
    for path in IMG_DIR.glob("FortRiley_r*_c*.tif"):
        with rasterio.open(path) as ds:
            w, s, e, n = array_bounds(ds.height, ds.width, ds.transform)
            minx, miny, maxx, maxy = min(minx, w), min(miny, s), max(maxx, e), max(maxy, n)
            crs = ds.crs
    west, south, east, north = transform_bounds(crs, "EPSG:4326", minx, miny, maxx, maxy)
    pad = 0.002
    return west - pad, south - pad, east + pad, north + pad


def _backup(path: Path) -> None:
    if not path.is_file():
        return
    bak = path.with_name(path.stem + "_old" + path.suffix)
    if not bak.is_file():
        path.replace(bak)


def fetch_osm(west: float, south: float, east: float, north: float) -> gpd.GeoDataFrame:
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"]({south},{west},{north},{east});
    );
    out geom;
    """
    last_err: Exception | None = None
    for url in OVERPASS_URLS:
        print(f"Downloading OSM highways from {url}…", flush=True)
        try:
            res = requests.post(
                url,
                data={"data": query.strip()},
                headers=HEADERS,
                timeout=240,
            )
            if res.status_code == 406:
                res = requests.get(url, params={"data": query.strip()}, headers=HEADERS, timeout=240)
            res.raise_for_status()
            payload = res.json()
            break
        except Exception as exc:
            last_err = exc
            print(f"  failed: {exc}", flush=True)
            payload = None
    else:
        payload = None
    rows = []
    if payload:
        for el in payload.get("elements") or []:
            geom = el.get("geometry") or []
            if el.get("type") != "way" or len(geom) < 2:
                continue
            tags = el.get("tags") or {}
            coords = [(p["lon"], p["lat"]) for p in geom]
            rows.append(
                {
                    "osm_id": int(el["id"]),
                    "fclass": str(tags.get("highway") or "")[:24],
                    "name": str(tags.get("name") or "")[:80],
                    "ref": str(tags.get("ref") or "")[:20],
                    "oneway": str(tags.get("oneway") or "")[:1],
                    "maxspeed": str(tags.get("maxspeed") or "")[:12],
                    "bridge": str(tags.get("bridge") or "")[:1],
                    "tunnel": str(tags.get("tunnel") or "")[:1],
                    "geometry": LineString(coords),
                }
            )
        gdf = gpd.GeoDataFrame(rows, crs=4326)
        print(f"  OSM ways {len(gdf)}", flush=True)
        if len(gdf):
            return gdf
    print("Overpass empty; falling back to Geofabrik Kansas roads extract…", flush=True)
    if last_err and not payload:
        print(f"  last Overpass error: {last_err}", flush=True)
    res = requests.get(GEOFABRIK_ROADS, headers=HEADERS, timeout=300)
    res.raise_for_status()
    tmp = ROADS_DIR / "_osm_dl"
    tmp.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(io.BytesIO(res.content))
    zf.extractall(tmp)
    shp = next(tmp.rglob("*roads*.shp"))
    gdf = gpd.read_file(shp)
    print(f"  Geofabrik roads {len(gdf)} from {shp.name}", flush=True)
    return gdf


def fetch_tiger() -> gpd.GeoDataFrame:
    frames = []
    for county, url in TIGER_URLS.items():
        print(f"Downloading TIGER {county}…", flush=True)
        res = requests.get(url, headers=HEADERS, timeout=180)
        res.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(res.content))
        shp_name = next(n for n in zf.namelist() if n.lower().endswith(".shp"))
        tmp = ROADS_DIR / "_tiger_dl"
        tmp.mkdir(parents=True, exist_ok=True)
        zf.extractall(tmp)
        gdf = gpd.read_file(tmp / shp_name)
        gdf["COUNTY"] = county
        keep = [c for c in ["FULLNAME", "RTTYP", "MTFCC", "LINEARID", "COUNTY", "geometry"] if c in gdf.columns]
        frames.append(gdf[keep])
        print(f"  {county} lines {len(gdf)}", flush=True)
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    if out.crs is None:
        out = out.set_crs(4326)
    elif str(out.crs) != "EPSG:4326":
        out = out.to_crs(4326)
    return out


def clip_and_write(gdf: gpd.GeoDataFrame, poly, dest: Path) -> Path:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    layer = gdf.to_crs(4326)
    hit = layer[layer.intersects(poly)]
    clipped = gpd.clip(hit, poly) if not hit.empty else hit
    clipped = clipped[~clipped.geometry.is_empty]
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            _backup(dest.with_suffix(ext))
        clipped.to_file(dest, driver="ESRI Shapefile")
    except PermissionError:
        dest = dest.with_name(dest.stem.replace("_clipped", "_full") + dest.suffix)
        clipped.to_file(dest, driver="ESRI Shapefile")
        print(f"  original locked; wrote {dest.name}", flush=True)
        return dest
    print(f"  wrote {dest.name} n={len(clipped)} bounds={tuple(round(x, 5) for x in clipped.total_bounds)}", flush=True)
    return dest


def main() -> int:
    ROADS_DIR.mkdir(parents=True, exist_ok=True)
    west, south, east, north = imagery_bounds_wgs84()
    print(f"imagery bbox WGS84 {west:.5f},{south:.5f},{east:.5f},{north:.5f}", flush=True)
    poly = box(west, south, east, north)
    osm = fetch_osm(west, south, east, north)
    osm_path = clip_and_write(osm, poly, ROADS_DIR / "osm_roads_clipped.shp")
    tiger = fetch_tiger()
    tiger_path = clip_and_write(tiger, poly, ROADS_DIR / "tiger_roads_clipped.shp")
    print(f"OSM  {osm_path}")
    print(f"TIGER {tiger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
