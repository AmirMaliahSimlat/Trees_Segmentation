"""Snap existing road centerlines onto OEM road/pavement pixels.

Does not train. Does not merge OSM and TIGER. UI is separate.
"""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import LineString, mapping
from shapely.ops import unary_union

from tree_seg.ground_fill import OEM8


def road_class_ids(id2label: dict[int, str] | None = None) -> set[int]:
    labels = id2label or OEM8
    ids = {i for i, name in labels.items() if str(name).lower() == "road"}
    return ids or {3}


def pavement_class_ids(id2label: dict[int, str] | None = None) -> set[int]:
    labels = id2label or OEM8
    ids = {i for i, name in labels.items() if "pavement" in str(name).lower()}
    return ids or {2}


def evidence_from_classes(
    classes: np.ndarray,
    *,
    road_ids: set[int],
    pavement_ids: set[int],
    include_pavement: bool = True,
    morph_close_px: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (road, pavement, union) uint8 masks."""
    road = np.isin(classes, list(road_ids)).astype(np.uint8)
    pave = np.isin(classes, list(pavement_ids)).astype(np.uint8)
    union = np.maximum(road, pave if include_pavement else np.zeros_like(road))
    if morph_close_px > 0:
        k = 2 * int(morph_close_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)
    return road, pave, (union > 0).astype(np.uint8)


def _explode_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    rows: list[dict[str, Any]] = []
    for _, rec in gdf.iterrows():
        geom = rec.geometry
        if geom is None or geom.is_empty:
            continue
        parts: Iterable
        if geom.geom_type == "MultiLineString":
            parts = geom.geoms
        elif geom.geom_type == "LineString":
            parts = (geom,)
        elif geom.geom_type == "GeometryCollection":
            parts = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
            extra: list = []
            for p in parts:
                if p.geom_type == "MultiLineString":
                    extra.extend(list(p.geoms))
                else:
                    extra.append(p)
            parts = extra
        else:
            continue
        for part in parts:
            if part is None or part.is_empty or part.length < 1e-6:
                continue
            row = rec.drop(labels="geometry").to_dict()
            row["geometry"] = part
            rows.append(row)
    if not rows:
        return gdf.iloc[0:0]
    return gpd.GeoDataFrame(rows, crs=gdf.crs)


def clip_roads(
    gdf: gpd.GeoDataFrame,
    bounds_poly,
    *,
    skip_fclass: set[str] | None = None,
    fclass_field: str | None = None,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    hit = gdf[gdf.intersects(bounds_poly)].copy()
    if skip_fclass and fclass_field and fclass_field in hit.columns:
        skip = {s.lower() for s in skip_fclass}
        hit = hit[~hit[fclass_field].astype(str).str.lower().isin(skip)]
    if hit.empty:
        return hit
    hit = gpd.clip(hit, bounds_poly)
    return _explode_lines(hit)


def _world_to_colrow(x: float, y: float, transform) -> tuple[float, float]:
    inv = ~transform
    c, r = inv * (x, y)
    return float(c), float(r)


def _colrow_to_world(c: float, r: float, transform) -> tuple[float, float]:
    x, y = transform * (c, r)
    return float(x), float(y)


def _sample(arr: np.ndarray, c: float, r: float) -> float:
    h, w = arr.shape[:2]
    ci, ri = int(round(c)), int(round(r))
    if ci < 0 or ri < 0 or ci >= w or ri >= h:
        return 0.0
    return float(arr[ri, ci])


def snap_linestring(
    line: LineString,
    evidence: np.ndarray,
    dist: np.ndarray,
    transform,
    *,
    search_px: int,
    densify_m: float,
    gsd: float,
    min_width_m: float,
    max_width_m: float,
) -> dict[str, Any] | None:
    if line is None or line.is_empty or line.length < gsd:
        return None
    dense = line.segmentize(max_segment_length=max(float(densify_m), gsd * 2))
    coords = list(dense.coords)
    if len(coords) < 2:
        return None
    h, w = evidence.shape[:2]
    snapped: list[tuple[float, float]] = []
    supported: list[bool] = []
    widths: list[float] = []
    shifts: list[float] = []

    for i, (x, y) in enumerate(coords):
        c0, r0 = _world_to_colrow(x, y, transform)
        if i == 0:
            x1, y1 = coords[min(i + 1, len(coords) - 1)]
        elif i == len(coords) - 1:
            x1, y1 = coords[i - 1]
        else:
            x1, y1 = coords[i + 1]
        c1, r1 = _world_to_colrow(x1, y1, transform)
        tx, ty = c1 - c0, r1 - r0
        nrm = (tx * tx + ty * ty) ** 0.5
        if nrm < 1e-3:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = -ty / nrm, tx / nrm
        best_score = -1.0
        best_c, best_r = c0, r0
        found = False
        for t in range(-int(search_px), int(search_px) + 1):
            cc = c0 + t * nx
            rr = r0 + t * ny
            ci, ri = int(round(cc)), int(round(rr))
            if ci < 0 or ri < 0 or ci >= w or ri >= h:
                continue
            if evidence[ri, ci] == 0:
                continue
            score = float(dist[ri, ci])
            if score > best_score:
                best_score = score
                best_c, best_r = float(ci) + 0.5, float(ri) + 0.5
                found = True
        if found:
            sx, sy = _colrow_to_world(best_c, best_r, transform)
            snapped.append((sx, sy))
            supported.append(True)
            width = float(np.clip(2.0 * best_score * gsd, min_width_m, max_width_m))
            widths.append(width)
            shifts.append(((sx - x) ** 2 + (sy - y) ** 2) ** 0.5)
        else:
            snapped.append((x, y))
            supported.append(False)
            widths.append(float("nan"))
            shifts.append(0.0)

    pts = [list(p) for p in snapped]
    for i in range(1, len(pts) - 1):
        if supported[i - 1] and supported[i] and supported[i + 1]:
            pts[i][0] = 0.25 * pts[i - 1][0] + 0.5 * pts[i][0] + 0.25 * pts[i + 1][0]
            pts[i][1] = 0.25 * pts[i - 1][1] + 0.5 * pts[i][1] + 0.25 * pts[i + 1][1]
    out_line = LineString(pts)
    if out_line.length < gsd:
        return None
    n = max(len(supported), 1)
    support = float(sum(supported) / n)
    valid_w = [x for x in widths if np.isfinite(x)]
    width_m = float(np.median(valid_w)) if valid_w else float("nan")
    snap_m = float(np.mean(shifts)) if shifts else 0.0
    if support >= 0.55:
        status = "snapped"
    elif support >= 0.2:
        status = "low"
    else:
        status = "none"
    return {
        "geometry": out_line,
        "width_m": None if not np.isfinite(width_m) else round(width_m, 2),
        "support": round(support, 3),
        "snap_m": round(snap_m, 2),
        "status": status,
        "n_vert": n,
    }


def snap_gdf(
    gdf: gpd.GeoDataFrame,
    evidence: np.ndarray,
    transform,
    *,
    gsd: float,
    search_m: float,
    densify_m: float,
    min_width_m: float,
    max_width_m: float,
    src: str,
    id_field: str | None,
    name_field: str | None,
    class_field: str | None,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["src", "src_id", "name", "fclass", "width_m", "support", "snap_m", "status", "geometry"],
            crs=gdf.crs,
        )
    dist = cv2.distanceTransform(evidence.astype(np.uint8), cv2.DIST_L2, 5)
    search_px = max(2, int(round(search_m / max(gsd, 1e-6))))
    rows: list[dict[str, Any]] = []
    for _, rec in gdf.iterrows():
        snapped = snap_linestring(
            rec.geometry,
            evidence,
            dist,
            transform,
            search_px=search_px,
            densify_m=densify_m,
            gsd=gsd,
            min_width_m=min_width_m,
            max_width_m=max_width_m,
        )
        if snapped is None:
            continue
        src_id = ""
        if id_field and id_field in rec.index and rec[id_field] is not None:
            src_id = str(rec[id_field])[:24]
        name = ""
        if name_field and name_field in rec.index and rec[name_field] is not None:
            name = str(rec[name_field])[:80]
        fclass = ""
        if class_field and class_field in rec.index and rec[class_field] is not None:
            fclass = str(rec[class_field])[:24]
        rows.append(
            {
                "src": src[:10],
                "src_id": src_id,
                "name": name,
                "fclass": fclass,
                **snapped,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(columns=["src", "src_id", "name", "fclass", "width_m", "support", "snap_m", "status", "geometry"], crs=gdf.crs)
    out = gpd.GeoDataFrame(rows, crs=gdf.crs)
    return out


def _rasterize_lines(gdf: gpd.GeoDataFrame, shape: tuple[int, int], transform, width_px: np.ndarray | float) -> np.ndarray:
    if gdf.empty:
        return np.zeros(shape, dtype=np.uint8)
    shapes = []
    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        if isinstance(width_px, np.ndarray):
            buf = float(max(width_px[i], 1.5))
        else:
            buf = float(width_px)
        poly = geom.buffer(buf, cap_style="flat", join_style="round")
        if poly.is_empty:
            continue
        shapes.append((mapping(poly), 1))
    if not shapes:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8)


def _skeleton_polylines(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    ys, xs = np.where(skel > 0)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    if not pixels:
        return []

    def nbrs(p: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = p
        out: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                q = (r + dr, c + dc)
                if q in pixels:
                    out.append(q)
        return out

    deg = {p: len(nbrs(p)) for p in pixels}

    def ekey(a: tuple[int, int], b: tuple[int, int]):
        return (a, b) if a < b else (b, a)

    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    lines: list[list[tuple[int, int]]] = []
    seeds = [p for p, d in deg.items() if d != 2]
    if not seeds:
        seeds = [next(iter(pixels))]

    for seed in seeds:
        for n in nbrs(seed):
            ek = ekey(seed, n)
            if ek in used:
                continue
            used.add(ek)
            chain = [seed, n]
            prev, cur = seed, n
            while deg.get(cur, 0) == 2:
                opts = [q for q in nbrs(cur) if q != prev]
                if not opts:
                    break
                nxt = opts[0]
                ek2 = ekey(cur, nxt)
                if ek2 in used:
                    break
                used.add(ek2)
                chain.append(nxt)
                prev, cur = cur, nxt
            if len(chain) >= 2:
                lines.append(chain)
    return lines


def propose_missing(
    evidence: np.ndarray,
    snapped: gpd.GeoDataFrame,
    transform,
    *,
    gsd: float,
    cover_extra_m: float,
    min_length_m: float,
    max_mean_width_m: float,
    min_width_m: float,
    max_width_m: float,
) -> gpd.GeoDataFrame:
    from skimage.morphology import skeletonize

    h, w = evidence.shape[:2]
    if snapped.empty:
        cover_m = np.full(len(snapped), 8.0 * gsd)
        covered = np.zeros((h, w), dtype=np.uint8)
    else:
        half = []
        for _, rec in snapped.iterrows():
            wm = rec.get("width_m")
            if wm is None or not np.isfinite(float(wm)):
                half.append(cover_extra_m)
            else:
                half.append(0.5 * float(wm) + cover_extra_m)
        covered = _rasterize_lines(snapped, (h, w), transform, np.asarray(half))
    remain = ((evidence > 0) & (covered == 0)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    remain = cv2.morphologyEx(remain, cv2.MORPH_OPEN, k)
    if not remain.any():
        return gpd.GeoDataFrame(columns=["src", "width_m", "length_m", "geometry"], crs=snapped.crs)

    dist = cv2.distanceTransform(remain, cv2.DIST_L2, 5)
    skel = skeletonize(remain > 0).astype(np.uint8)
    rows: list[dict[str, Any]] = []
    for chain in _skeleton_polylines(skel):
        if len(chain) < 4:
            continue
        pts = [_colrow_to_world(c + 0.5, r + 0.5, transform) for r, c in chain]
        line = LineString(pts)
        length = float(line.length)
        if length < min_length_m:
            continue
        dts = [dist[r, c] for r, c in chain]
        mean_w = float(2.0 * np.mean(dts) * gsd)
        if mean_w > max_mean_width_m:
            continue
        width_m = float(np.clip(mean_w, min_width_m, max_width_m))
        rows.append(
            {
                "src": "oem",
                "width_m": round(width_m, 2),
                "length_m": round(length, 1),
                "geometry": line,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(columns=["src", "width_m", "length_m", "geometry"], crs=snapped.crs)
    return gpd.GeoDataFrame(rows, crs=snapped.crs)


def lines_union_length(gdf: gpd.GeoDataFrame) -> float:
    if gdf.empty:
        return 0.0
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return 0.0
    return float(unary_union(geoms).length)
