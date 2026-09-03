"""Grow an asymmetric road ribbon from edited polylines.

The line is a seed on pavement, not a centerline. Left and right extents are
measured independently; each feature keeps one total width W. Outlier stations
are retried nearby, then dropped — never blended into W.

Stations whose seed pixel is clearly not pavement (vegetation, water, painted
objects) do not vote for W. Lines with many of those hits are sampled denser.
Remaining thin / heavily covered lines can be remeasured on a temporary
object-cleaned corridor (OEM + LaMa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import LineString, Polygon
from shapely.ops import substring, unary_union
from shapely.validation import make_valid


@dataclass
class RibbonParams:
    step_m: float = 1.0
    min_step_m: float = 0.4
    min_samples: int = 24
    retry_m: tuple[float, ...] = (0.1, 0.2)
    max_search_m: float = 20.0
    lot_search_m: float = 48.0
    lot_width_m: float = 18.0
    max_gap_m: float = 1.4
    min_half_m: float = 0.45
    lab_cut: float = 26.0
    eg_cut: float = 18.0
    outlier_rel: float = 0.55
    outlier_abs_m: float = 2.5
    min_inliers: int = 5
    # Edge placement uses only stations whose measured total is close to W.
    anchor_rel: float = 0.18
    anchor_abs_m: float = 0.7
    smooth_m: float = 10.0
    # Reject a width-anchor whose left/right split jumps vs nearby anchors
    # (crossroad on one side, tree on the other).
    split_jump_m: float = 1.3
    split_jump_rel: float = 0.16
    split_radius_m: float = 22.0
    skip_not_road: bool = True
    dense_skip_rate: float = 0.18
    clean_skip_rate: float = 0.22
    clean_thin_m: float = 3.6
    clean_margin_m: float = 10.0
    clean_max_side: int = 1400
    clean_chunk_m: float = 380.0
    min_widen_m: float = 0.45


@dataclass
class CorridorCleaner:
    """OEM object mask + LaMa fill, used only for temporary road remeasure."""

    processor: Any
    model: Any
    object_ids: set[int]
    device: Any
    filler: Any
    tile_size: int = 512
    overlap: float = 0.125
    dilate_px: int = 8
    shadow_grow_px: int = 16


def _sample_nn(arr: np.ndarray, c: float, r: float) -> np.ndarray | None:
    h, w = arr.shape[:2]
    ci = int(round(c))
    ri = int(round(r))
    if ci < 0 or ri < 0 or ci >= w or ri >= h:
        return None
    return arr[ri, ci]


def _eg(rgb: np.ndarray) -> float:
    r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return 2.0 * g - r - b


def _not_road_rgb(rgb: np.ndarray) -> bool:
    """True when the pixel is almost certainly not pavement.

    Yellow/white paint and tan/gray asphalt are kept. Green canopy, water,
    and strongly colored objects (cars, red roofs) are skipped.
    """
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    eg = 2.0 * g - r - b
    if g >= r + 6.0 and g >= b + 6.0 and eg >= 12.0:
        return True
    if b >= r + 12.0 and b >= g + 8.0 and b >= 48.0:
        return True
    if r >= g + 28.0 and r >= b + 28.0 and r >= 70.0:
        return True
    if r >= g + 22.0 and b >= g + 22.0 and max(r, b) >= 70.0:
        return True
    return False


def _world_to_colrow(x: float, y: float, transform) -> tuple[float, float]:
    c, r = ~transform * (x, y)
    return float(c), float(r)


def _explode_lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom] if geom.length > 1e-6 else []
    if geom.geom_type == "MultiLineString":
        out: list[LineString] = []
        for part in geom.geoms:
            out.extend(_explode_lines(part))
        return out
    return []


def _sample_step_m(length_m: float, p: RibbonParams) -> float:
    if length_m <= 0:
        return p.min_step_m
    step = length_m / max(p.min_samples - 1, 1)
    return float(np.clip(step, p.min_step_m, p.step_m))


def _stations(line: LineString, step_m: float) -> list[tuple[float, float, float]]:
    length = float(line.length)
    n = max(3, int(round(length / step_m)) + 1)
    out: list[tuple[float, float, float]] = []
    for i in range(n):
        s = 0.0 if n == 1 else length * i / (n - 1)
        pt = line.interpolate(s)
        out.append((s, float(pt.x), float(pt.y)))
    return out


def _px_offset(x: float, y: float, dx: float, dy: float, transform) -> tuple[float, float, float, float]:
    c, r = _world_to_colrow(x, y, transform)
    c1, r1 = _world_to_colrow(x + dx, y + dy, transform)
    return c, r, c1 - c, r1 - r


def _proto(
    lab: np.ndarray,
    x: float,
    y: float,
    tx: float,
    ty: float,
    transform,
    rgb: np.ndarray | None = None,
    skip_not_road: bool = False,
) -> np.ndarray | None:
    labs: list[np.ndarray] = []
    for t in (-0.6, -0.3, 0.0, 0.3, 0.6):
        c, r = _world_to_colrow(x + tx * t, y + ty * t, transform)
        pix = _sample_nn(lab, c, r)
        if pix is None:
            continue
        if skip_not_road and rgb is not None:
            gp = _sample_nn(rgb, c, r)
            if gp is None or _not_road_rgb(gp):
                continue
        labs.append(pix.astype(np.float32))
    if not labs:
        return None
    return np.mean(np.stack(labs, axis=0), axis=0)


def _walk_half(
    rgb: np.ndarray,
    lab: np.ndarray,
    c: float,
    r: float,
    dc_m: float,
    dr_m: float,
    proto: np.ndarray,
    gsd: float,
    max_m: float,
    p: RibbonParams,
) -> float:
    step = max(0.5 * gsd, 0.15)
    last = p.min_half_m
    gap = 0.0
    t = p.min_half_m
    while t <= max_m + 1e-6:
        cc = c + dc_m * t
        rr = r + dr_m * t
        lp = _sample_nn(lab, cc, rr)
        gp = _sample_nn(rgb, cc, rr)
        if lp is None or gp is None:
            return max(last, p.min_half_m)
        dist = float(np.linalg.norm(lp.astype(np.float32) - proto))
        # Fresh/dark asphalt is often blue-teal (G≈B > R), so raw excess-green
        # is high even on pavement. Only treat true green vegetation as a stop.
        like = dist < p.lab_cut and not _not_road_rgb(gp)
        if like:
            last = t
            gap = 0.0
        else:
            gap += step
            if gap > p.max_gap_m:
                return max(last, p.min_half_m)
        t += step
    return max_m


def _measure_station(
    rgb: np.ndarray,
    lab: np.ndarray,
    x: float,
    y: float,
    tx: float,
    ty: float,
    nx: float,
    ny: float,
    transform,
    gsd: float,
    max_m: float,
    p: RibbonParams,
    skip_not_road: bool = False,
) -> tuple[float, float] | None:
    proto = _proto(lab, x, y, tx, ty, transform, rgb=rgb, skip_not_road=skip_not_road)
    if proto is None:
        return None
    c, r, dc_l, dr_l = _px_offset(x, y, -nx, -ny, transform)
    _, _, dc_r, dr_r = _px_offset(x, y, nx, ny, transform)
    left = _walk_half(rgb, lab, c, r, dc_l, dr_l, proto, gsd, max_m, p)
    right = _walk_half(rgb, lab, c, r, dc_r, dr_r, proto, gsd, max_m, p)
    return left, right


def _is_inlier(w: float, w_ref: float, p: RibbonParams) -> bool:
    if not np.isfinite(w) or w <= 0:
        return False
    return abs(w - w_ref) <= max(p.outlier_abs_m, p.outlier_rel * max(w_ref, 1.0))


def _is_anchor(w: float, w_ref: float, p: RibbonParams) -> bool:
    if not np.isfinite(w) or w <= 0 or w_ref <= 0:
        return False
    return abs(w - w_ref) <= max(p.anchor_abs_m, p.anchor_rel * w_ref)


def _local_median_excl(val: np.ndarray, ok: np.ndarray, radius: int) -> np.ndarray:
    """Median of other True samples in [i-radius, i+radius]."""
    n = len(val)
    out = np.full(n, np.nan, dtype=np.float64)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out
    for i in idx:
        lo, hi = i - radius, i + radius
        nb = idx[(idx >= lo) & (idx <= hi) & (idx != i)]
        if len(nb) < 2:
            nb = idx[idx != i]
        if len(nb) == 0:
            continue
        out[i] = float(np.median(val[nb]))
    return out


def _split_consistent(
    left: np.ndarray,
    ok: np.ndarray,
    w_hat: float,
    step: float,
    p: RibbonParams,
) -> np.ndarray:
    """Keep stations whose left offset agrees with nearby anchors."""
    keep = ok.copy()
    if int(keep.sum()) < 3:
        return keep
    radius = max(2, int(round(p.split_radius_m / max(step, 0.4))))
    gate = max(p.split_jump_m, p.split_jump_rel * max(w_hat, 1.0))
    for _ in range(2):
        loc = _local_median_excl(left, keep, radius)
        nxt = keep.copy()
        for i in np.flatnonzero(keep):
            if not np.isfinite(loc[i]):
                continue
            if abs(left[i] - loc[i]) > gate:
                nxt[i] = False
        keep = nxt
        if int(keep.sum()) < 1:
            break
    return keep


def _tangent_at(line: LineString, s: float, length: float) -> tuple[float, float, float, float]:
    s0 = max(0.0, min(length, s) - 0.4)
    s1 = max(0.0, min(length, s + 0.4))
    if s1 - s0 < 1e-4:
        s0, s1 = 0.0, min(length, 0.8)
    a = line.interpolate(s0)
    b = line.interpolate(s1)
    tx, ty = b.x - a.x, b.y - a.y
    ln = (tx * tx + ty * ty) ** 0.5
    if ln < 1e-6:
        tx, ty, ln = 1.0, 0.0, 1.0
    tx, ty = tx / ln, ty / ln
    return tx, ty, -ty, tx


def _interp_from_mask(s: np.ndarray, val: np.ndarray, ok: np.ndarray, fill: float) -> np.ndarray:
    out = np.full(len(s), fill, dtype=np.float64)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return out
    if n_ok == 1:
        out[:] = float(val[ok][0])
        return out
    out[:] = np.interp(s, s[ok], val[ok])
    return out


def _moving_mean(v: np.ndarray, win: int) -> np.ndarray:
    n = len(v)
    if n < 3 or win < 3:
        return v.astype(np.float64, copy=True)
    k = win if win % 2 else win + 1
    k = min(k, n if n % 2 else n - 1)
    if k < 3:
        return v.astype(np.float64, copy=True)
    pad = k // 2
    ext = np.pad(v.astype(np.float64), pad, mode="edge")
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(ext, ker, mode="valid")


def _smooth_open_pts(pts: list[tuple[float, float]], win: int) -> list[tuple[float, float]]:
    if len(pts) < 3 or win < 3:
        return pts
    xs = _moving_mean(np.array([p[0] for p in pts], dtype=np.float64), win)
    ys = _moving_mean(np.array([p[1] for p in pts], dtype=np.float64), win)
    xs[0], ys[0] = pts[0]
    xs[-1], ys[-1] = pts[-1]
    return list(zip(xs.tolist(), ys.tolist()))


def _ribbon_polygon(
    stations: list[tuple[float, float, float]],
    left_m: np.ndarray,
    right_m: np.ndarray,
    line: LineString,
    smooth_win: int = 1,
) -> Polygon | None:
    length = float(line.length)
    left_pts: list[tuple[float, float]] = []
    right_pts: list[tuple[float, float]] = []
    for i, (s, x, y) in enumerate(stations):
        _, _, nx, ny = _tangent_at(line, s, length)
        left_pts.append((x - nx * float(left_m[i]), y - ny * float(left_m[i])))
        right_pts.append((x + nx * float(right_m[i]), y + ny * float(right_m[i])))
    left_pts = _smooth_open_pts(left_pts, smooth_win)
    right_pts = _smooth_open_pts(right_pts, smooth_win)
    ring = left_pts + list(reversed(right_pts))
    if len(ring) < 4:
        return None
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.geom_type != "Polygon":
            return None
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _seed_blocked(
    rgb: np.ndarray,
    lab: np.ndarray,
    x: float,
    y: float,
    tx: float,
    ty: float,
    transform,
) -> bool:
    c, r = _world_to_colrow(x, y, transform)
    gp = _sample_nn(rgb, c, r)
    if gp is None or _not_road_rgb(gp):
        return True
    return _proto(lab, x, y, tx, ty, transform, rgb=rgb, skip_not_road=True) is None


def _count_blocked(line: LineString, stations, rgb, lab, transform) -> int:
    length = float(line.length)
    n = 0
    for s, x, y in stations:
        tx, ty, _, _ = _tangent_at(line, s, length)
        if _seed_blocked(rgb, lab, x, y, tx, ty, transform):
            n += 1
    return n


def _choose_stations(
    line: LineString,
    rgb: np.ndarray,
    lab: np.ndarray,
    transform,
    p: RibbonParams,
    skip_not_road: bool,
) -> tuple[list[tuple[float, float, float]], float, int]:
    step = _sample_step_m(float(line.length), p)
    stations = _stations(line, step)
    n_skip = _count_blocked(line, stations, rgb, lab, transform) if skip_not_road else 0
    if skip_not_road and stations and (n_skip / len(stations)) >= p.dense_skip_rate:
        step = p.min_step_m
        stations = _stations(line, step)
        n_skip = _count_blocked(line, stations, rgb, lab, transform)
    return stations, step, n_skip


def _looks_like_pavement(rgb: np.ndarray) -> bool:
    if _not_road_rgb(rgb):
        return False
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    mx, mn = max(r, g, b), min(r, g, b)
    if g >= r + 3.0 and g >= b + 3.0:
        return False
    if mx >= 30.0 and (mx - mn) >= 50.0:
        return False
    return True


def _paved_frac(rgb: np.ndarray, line: LineString, transform, n: int = 28, *, strict: bool = False) -> float:
    length = float(line.length)
    hits = 0
    ok = 0
    for i in range(max(n, 3)):
        s = 0.0 if n == 1 else length * i / (n - 1)
        pt = line.interpolate(s)
        c, r = _world_to_colrow(float(pt.x), float(pt.y), transform)
        gp = _sample_nn(rgb, c, r)
        if gp is None:
            continue
        ok += 1
        if (_looks_like_pavement if strict else (lambda p: not _not_road_rgb(p)))(gp):
            hits += 1
    return hits / max(ok, 1)


def _needs_clean(row: dict[str, Any] | None, length_m: float, p: RibbonParams) -> bool:
    if length_m < 8.0:
        return False
    if row is None:
        return True
    skip_rate = float(row.get("skip_pct") or 0) / 100.0
    w = float(row.get("width_m") or 0.0)
    n_in = int(row.get("n_in") or 0)
    n_samp = max(int(row.get("n_samp") or 1), 1)
    if skip_rate >= p.clean_skip_rate:
        return True
    if skip_rate >= 0.10 and w <= p.clean_thin_m:
        return True
    if n_in < max(4, int(0.12 * n_samp)) and length_m >= 20.0:
        return True
    return False


def _accept_clean(before: dict[str, Any], after: dict[str, Any], p: RibbonParams) -> bool:
    wb = float(before.get("width_m") or 0.0)
    wa = float(after.get("width_m") or 0.0)
    if wa < wb + p.min_widen_m:
        return False
    if wb >= p.lot_width_m:
        cap = max(p.lot_width_m, wb * 1.35)
    elif wb >= 6.0:
        # Already a plausible road; don't eat yards after fill.
        cap = min(wb + 2.2, 12.0)
    else:
        cap = min(10.5, max(wb + 6.0, 8.5))
    return wa <= cap + 1e-6


def _line_chunks(line: LineString, max_m: float) -> list[LineString]:
    length = float(line.length)
    if max_m <= 0 or length <= max_m:
        return [line]
    n = int(np.ceil(length / max_m))
    out: list[LineString] = []
    for i in range(n):
        part = substring(line, length * i / n, length * (i + 1) / n)
        if part is None or part.is_empty:
            continue
        if part.geom_type == "MultiLineString":
            out.extend(_explode_lines(part))
        elif part.geom_type == "LineString" and part.length > 1e-3:
            out.append(part)
    return out or [line]


def _pixel_window(geom, transform, shape, pad_px: int = 8) -> tuple[int, int, int, int] | None:
    minx, miny, maxx, maxy = geom.bounds
    cols: list[float] = []
    rows: list[float] = []
    for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
        c, r = _world_to_colrow(x, y, transform)
        cols.append(c)
        rows.append(r)
    h, w = shape[:2]
    c0 = max(0, int(np.floor(min(cols))) - pad_px)
    c1 = min(w, int(np.ceil(max(cols))) + pad_px)
    r0 = max(0, int(np.floor(min(rows))) - pad_px)
    r1 = min(h, int(np.ceil(max(rows))) + pad_px)
    if c1 <= c0 or r1 <= r0:
        return None
    return c0, r0, c1, r1


def _clamp_window(
    win: tuple[int, int, int, int],
    shape,
    cx: float,
    cy: float,
    max_side: int,
) -> tuple[int, int, int, int]:
    c0, r0, c1, r1 = win
    h, w = shape[:2]
    side = max(32, int(max_side))
    if c1 - c0 > side:
        c0 = int(np.clip(round(cx) - side // 2, 0, max(0, w - side)))
        c1 = min(w, c0 + side)
    if r1 - r0 > side:
        r0 = int(np.clip(round(cy) - side // 2, 0, max(0, h - side)))
        r1 = min(h, r0 + side)
    return c0, r0, c1, r1


def paste_clean_corridor(
    work: np.ndarray,
    rgb: np.ndarray,
    line: LineString,
    transform,
    gsd: float,
    half_m: float,
    p: RibbonParams,
    cleaner: CorridorCleaner,
) -> list[dict[str, Any]]:
    """Detect+inpaint objects in a corridor around `line`; paste into `work`."""
    from tree_seg.ground_fill import dilate_mask, grow_attached_shadows, segment_objects

    pad_m = max(float(half_m), 2.5) + p.clean_margin_m
    saved: list[dict[str, Any]] = []
    for chunk in _line_chunks(line, p.clean_chunk_m):
        buf = chunk.buffer(pad_m)
        win = _pixel_window(buf, transform, rgb.shape, pad_px=8)
        if win is None:
            continue
        mid = chunk.interpolate(0.5, normalized=True)
        cx, cy = _world_to_colrow(float(mid.x), float(mid.y), transform)
        c0, r0, c1, r1 = _clamp_window(win, rgb.shape, cx, cy, p.clean_max_side)
        crop = rgb[r0:r1, c0:c1]
        if crop.size == 0:
            continue
        obj = segment_objects(
            crop,
            cleaner.processor,
            cleaner.model,
            cleaner.object_ids,
            cleaner.device,
            tile_size=cleaner.tile_size,
            overlap=cleaner.overlap,
        )
        obj = dilate_mask(obj, cleaner.dilate_px)
        obj = grow_attached_shadows(crop, obj, max_px=cleaner.shadow_grow_px, luma_ratio=0.55)
        filled = cleaner.filler.fill(crop, obj)
        hole = obj.astype(bool)[..., None]
        cleaned = np.where(hole, filled, crop)
        work[r0:r1, c0:c1] = np.where(hole, filled, work[r0:r1, c0:c1])
        saved.append(
            {
                "rgb": crop.copy(),
                "clean": cleaned,
                "obj": obj,
                "box": (c0, r0, c1, r1),
                "n_obj": int(obj.sum()),
            }
        )
    return saved


def ribbon_for_line(
    line: LineString,
    rgb: np.ndarray,
    lab: np.ndarray,
    transform,
    gsd: float,
    p: RibbonParams | None = None,
    skip_not_road: bool | None = None,
) -> dict[str, Any] | None:
    p = p or RibbonParams()
    if skip_not_road is None:
        skip_not_road = p.skip_not_road
    if line is None or line.is_empty or line.length < gsd:
        return None
    stations, step, n_skip = _choose_stations(line, rgb, lab, transform, p, skip_not_road)
    length = float(line.length)

    def measure_at(s: float, x: float, y: float, max_m: float) -> tuple[float, float] | None:
        tx, ty, nx, ny = _tangent_at(line, s, length)
        if skip_not_road and _seed_blocked(rgb, lab, x, y, tx, ty, transform):
            return None
        return _measure_station(
            rgb, lab, x, y, tx, ty, nx, ny, transform, gsd, max_m, p, skip_not_road=skip_not_road
        )

    raw: list[tuple[float, float] | None] = []
    for s, x, y in stations:
        raw.append(measure_at(s, x, y, p.max_search_m))

    widths = np.array(
        [((a[0] + a[1]) if a else np.nan) for a in raw],
        dtype=np.float64,
    )
    finite = np.isfinite(widths)
    if int(finite.sum()) < 3:
        return None
    w_ref = float(np.median(widths[finite]))
    search = p.lot_search_m if w_ref >= p.lot_width_m else p.max_search_m
    if search > p.max_search_m + 1e-6:
        raw = [measure_at(s, x, y, search) for s, x, y in stations]
        widths = np.array(
            [((a[0] + a[1]) if a else np.nan) for a in raw],
            dtype=np.float64,
        )
        finite = np.isfinite(widths)
        if int(finite.sum()) < 3:
            return None
        w_ref = float(np.median(widths[finite]))

    n_retry = 0
    for i, (s, x, y) in enumerate(stations):
        w = widths[i]
        if np.isfinite(w) and _is_inlier(w, w_ref, p):
            continue
        hit = None
        for ds in p.retry_m:
            for sign in (1.0, -1.0):
                ss = float(np.clip(s + sign * ds, 0.0, length))
                pt = line.interpolate(ss)
                got = measure_at(ss, float(pt.x), float(pt.y), search)
                if got is None:
                    continue
                ww = got[0] + got[1]
                if _is_inlier(ww, w_ref, p):
                    hit = got
                    break
            if hit is not None:
                break
        if hit is not None:
            raw[i] = hit
            widths[i] = hit[0] + hit[1]
            n_retry += 1

    finite = np.isfinite(widths)
    if int(finite.sum()) < 3:
        return None
    w_ref = float(np.median(widths[finite]))
    ok = np.array(
        [bool(raw[i] is not None and _is_inlier(widths[i], w_ref, p)) for i in range(len(stations))],
        dtype=bool,
    )
    if int(ok.sum()) < p.min_inliers:
        # Keep the median width but still drop spikes; if almost nothing survived,
        # fall back to stations within 2x the looser gate.
        loose = np.array(
            [
                bool(raw[i] is not None and abs(widths[i] - w_ref) <= max(4.0, 0.9 * w_ref))
                for i in range(len(stations))
            ],
            dtype=bool,
        )
        if int(loose.sum()) >= 3:
            ok = loose
        else:
            return None

    in_w = widths[ok]
    w_hat = float(np.mean(in_w))
    if np.isfinite(widths).any():
        near = np.array(
            [bool(raw[i] is not None and _is_anchor(widths[i], w_hat, p)) for i in range(len(stations))],
            dtype=bool,
        )
        if int(near.sum()) >= 3:
            w_hat = float(np.mean(widths[near]))
    cap = search - 0.35
    width_anchor = np.zeros(len(stations), dtype=bool)
    left_s = np.full(len(stations), np.nan, dtype=np.float64)
    for i, got in enumerate(raw):
        if got is None or not _is_anchor(widths[i], w_hat, p):
            continue
        if got[0] >= cap or got[1] >= cap:
            continue
        tot = max(got[0] + got[1], 1e-3)
        width_anchor[i] = True
        left_s[i] = float(np.clip(got[0] * (w_hat / tot), p.min_half_m, w_hat - p.min_half_m))
    anchor = _split_consistent(left_s, width_anchor, w_hat, step, p)

    loc = _local_median_excl(left_s, anchor, max(2, int(round(p.split_radius_m / max(step, 0.4)))))
    gate = max(p.split_jump_m, p.split_jump_rel * max(w_hat, 1.0))
    n_split_retry = 0
    for i, (s, x, y) in enumerate(stations):
        if anchor[i] or not width_anchor[i]:
            continue
        hit = None
        for ds in p.retry_m:
            for sign in (1.0, -1.0):
                ss = float(np.clip(s + sign * ds, 0.0, length))
                pt = line.interpolate(ss)
                got = measure_at(ss, float(pt.x), float(pt.y), search)
                if got is None or got[0] >= cap or got[1] >= cap:
                    continue
                ww = got[0] + got[1]
                if not _is_anchor(ww, w_hat, p):
                    continue
                tot = max(ww, 1e-3)
                left_try = float(np.clip(got[0] * (w_hat / tot), p.min_half_m, w_hat - p.min_half_m))
                ref = loc[i] if np.isfinite(loc[i]) else (float(np.median(left_s[anchor])) if int(anchor.sum()) else 0.5 * w_hat)
                if abs(left_try - ref) <= gate:
                    hit = (got, left_try)
                    break
            if hit is not None:
                break
        if hit is not None:
            raw[i], left_s[i] = hit[0], hit[1]
            widths[i] = hit[0][0] + hit[0][1]
            anchor[i] = True
            n_split_retry += 1

    if int(anchor.sum()) < 1:
        anchor = np.zeros(len(stations), dtype=bool)
    split_a = np.full(len(stations), 0.5, dtype=np.float64)
    for i in np.flatnonzero(anchor):
        if not np.isfinite(left_s[i]) or w_hat <= 0:
            continue
        split_a[i] = float(np.clip(left_s[i] / w_hat, 0.08, 0.92))
    s_arr = np.array([st[0] for st in stations], dtype=np.float64)
    fill = float(np.median(split_a[anchor])) if int(anchor.sum()) else 0.5
    split = _interp_from_mask(s_arr, split_a, anchor, fill)
    win = int(round(p.smooth_m / max(step, 0.4)))
    if win % 2 == 0:
        win += 1
    split = np.clip(_moving_mean(split, win), 0.08, 0.92)
    left = np.clip(split * w_hat, p.min_half_m, max(w_hat - p.min_half_m, p.min_half_m))
    right = w_hat - left
    poly = _ribbon_polygon(stations, left, right, line, smooth_win=max(win, 5))
    if poly is None:
        return None
    kind = "lot" if w_hat >= p.lot_width_m else "road"
    n_samp = len(stations)
    skip_pct = int(round(100.0 * n_skip / max(n_samp, 1)))
    return {
        "geometry": poly,
        "width_m": round(w_hat, 2),
        "n_samp": n_samp,
        "n_in": int(ok.sum()),
        "n_anch": int(anchor.sum()),
        "n_retry": n_retry + n_split_retry,
        "n_skip": int(n_skip),
        "skip_pct": skip_pct,
        "kind": kind,
        "step_m": round(step, 3),
    }


RIBBON_COLS = [
    "src",
    "src_id",
    "name",
    "fclass",
    "length_m",
    "width_m",
    "w_naive",
    "w_skip",
    "w_clean",
    "n_samp",
    "n_in",
    "n_anch",
    "n_retry",
    "n_skip",
    "skip_pct",
    "kind",
    "step_m",
    "method",
    "geometry",
]


def ribbons_for_gdf(
    gdf: gpd.GeoDataFrame,
    rgb: np.ndarray,
    transform,
    gsd: float,
    p: RibbonParams | None = None,
    cleaner: CorridorCleaner | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> gpd.GeoDataFrame:
    p = p or RibbonParams()
    lab = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2LAB)
    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, LineString, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]] = []
    for i, rec in enumerate(gdf.itertuples(index=False)):
        geom = rec.geometry
        attrs = rec._asdict()
        attrs.pop("geometry", None)
        for part in _explode_lines(geom):
            naive = ribbon_for_line(part, rgb, lab, transform, gsd, p, skip_not_road=False)
            skipped = ribbon_for_line(part, rgb, lab, transform, gsd, p, skip_not_road=True)
            chosen = skipped or naive
            if chosen is None:
                continue
            w_naive = None if naive is None else float(naive["width_m"])
            w_skip = None if skipped is None else float(skipped["width_m"])
            row = {
                "src": attrs.get("src", ""),
                "src_id": str(attrs.get("src_id", "")),
                "name": str(attrs.get("name", "")),
                "fclass": str(attrs.get("fclass", "")),
                "length_m": round(float(part.length), 2),
                **chosen,
                "w_naive": None if w_naive is None else round(w_naive, 2),
                "w_skip": None if w_skip is None else round(w_skip, 2),
                "w_clean": None,
                "method": "skip" if skipped is not None else "naive",
            }
            if skipped is not None:
                row["n_skip"] = skipped.get("n_skip", 0)
                row["skip_pct"] = skipped.get("skip_pct", 0)
            elif naive is not None:
                row["n_skip"] = naive.get("n_skip", 0)
                row["skip_pct"] = naive.get("skip_pct", 0)
            if cleaner is not None and _needs_clean(skipped, float(part.length), p):
                pending.append((len(rows), part, row, naive, skipped))
            elif examples is not None and w_naive is not None and w_skip is not None and w_skip >= w_naive + p.min_widen_m:
                examples.append(
                    {
                        "kind": "idea1",
                        "line": part,
                        "name": row["name"],
                        "fclass": row["fclass"],
                        "length_m": row["length_m"],
                        "w_naive": w_naive,
                        "w_skip": w_skip,
                        "w_clean": None,
                        "naive_geom": naive["geometry"] if naive else None,
                        "skip_geom": skipped["geometry"] if skipped else None,
                        "clean_geom": None,
                        "crops": [],
                    }
                )
            rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  ribbon {i + 1}/{len(gdf)}", flush=True)
    if pending and cleaner is not None:
        work = rgb.copy()
        crop_map: dict[int, list[dict[str, Any]]] = {}
        print(f"  clean corridors {len(pending)}", flush=True)
        for k, (idx, part, row, naive, skipped) in enumerate(pending):
            half = 0.5 * float((skipped or naive or row).get("width_m") or 5.0)
            crop_map[idx] = paste_clean_corridor(work, rgb, part, transform, gsd, half, p, cleaner)
            if (k + 1) % 5 == 0 or k + 1 == len(pending):
                print(f"    cleaned {k + 1}/{len(pending)}", flush=True)
        work_lab = cv2.cvtColor(work[:, :, ::-1], cv2.COLOR_BGR2LAB)
        for idx, part, row, naive, skipped in pending:
            after = ribbon_for_line(part, work, work_lab, transform, gsd, p, skip_not_road=True)
            w_naive = row.get("w_naive")
            w_skip = row.get("w_skip")
            used_clean = False
            if after is not None and _paved_frac(work, part, transform, strict=True) >= 0.5:
                before = skipped or naive
                if before is None or _accept_clean(before, after, p):
                    keep_meta = {k: row[k] for k in ("src", "src_id", "name", "fclass", "length_m", "w_naive", "w_skip")}
                    row.clear()
                    row.update(keep_meta)
                    row.update(after)
                    row["w_clean"] = round(float(after["width_m"]), 2)
                    row["method"] = "clean"
                    used_clean = True
            if examples is None:
                continue
            if used_clean:
                crops = crop_map.get(idx, [])
                best = max(crops, key=lambda c: int(c.get("n_obj") or 0)) if crops else None
                if best is not None:
                    best = {
                        "rgb": best["rgb"],
                        "clean": best["clean"],
                        "box": best["box"],
                        "n_obj": best["n_obj"],
                    }
                examples.append(
                    {
                        "kind": "idea2",
                        "line": part,
                        "name": row["name"],
                        "fclass": row["fclass"],
                        "length_m": row["length_m"],
                        "w_naive": w_naive,
                        "w_skip": w_skip,
                        "w_clean": row.get("w_clean"),
                        "naive_geom": None if naive is None else naive.get("geometry"),
                        "skip_geom": None if skipped is None else skipped.get("geometry"),
                        "clean_geom": row.get("geometry"),
                        "crops": [] if best is None else [best],
                    }
                )
            elif w_naive is not None and w_skip is not None and float(w_skip) >= float(w_naive) + p.min_widen_m:
                examples.append(
                    {
                        "kind": "idea1",
                        "line": part,
                        "name": row["name"],
                        "fclass": row["fclass"],
                        "length_m": row["length_m"],
                        "w_naive": w_naive,
                        "w_skip": w_skip,
                        "w_clean": None,
                        "naive_geom": None if naive is None else naive.get("geometry"),
                        "skip_geom": None if skipped is None else skipped.get("geometry"),
                        "clean_geom": None,
                        "crops": [],
                    }
                )
    if not rows:
        return gpd.GeoDataFrame(columns=RIBBON_COLS, crs=gdf.crs)
    return gpd.GeoDataFrame(rows, crs=gdf.crs)


def rasterize_ribbons(
    ribbons: gpd.GeoDataFrame,
    shape: tuple[int, int],
    transform,
) -> np.ndarray:
    if ribbons.empty:
        return np.zeros(shape, dtype=np.uint8)
    shapes = ((geom, 1) for geom in ribbons.geometry if geom is not None and not geom.is_empty)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8)


def _polygon_parts(geom) -> list:
    if geom is None or geom.is_empty:
        return []
    geom = make_valid(geom)
    if geom.geom_type == "Polygon":
        return [geom] if geom.area > 0 else []
    if geom.geom_type == "MultiPolygon":
        return [g for g in geom.geoms if g is not None and not g.is_empty and g.area > 0]
    if geom.geom_type == "GeometryCollection":
        out: list = []
        for g in geom.geoms:
            out.extend(_polygon_parts(g))
        return out
    return []


def merge_touching_ribbons(gdf: gpd.GeoDataFrame, snap_m: float = 0.35) -> gpd.GeoDataFrame:
    """Union overlapping/touching ribbons into one polygon per connected piece."""
    empty_cols = ["id", "n_rib", "kind", "width_m", "area_m2", "geometry"]
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=empty_cols, crs=getattr(gdf, "crs", None))
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return gpd.GeoDataFrame(columns=empty_cols, crs=gdf.crs)
    if snap_m > 0:
        padded = [g.buffer(snap_m) for g in geoms]
        merged = unary_union(padded)
        if merged is not None and not merged.is_empty:
            merged = merged.buffer(-snap_m)
    else:
        merged = unary_union(geoms)
    parts = _polygon_parts(merged)
    src = gdf.copy()
    rows: list[dict[str, Any]] = []
    for i, poly in enumerate(parts, start=1):
        hit = src[src.intersects(poly)]
        n = 0
        kinds: list[str] = []
        widths: list[float] = []
        for rec in hit.itertuples(index=False):
            other = rec.geometry
            if other is None or other.is_empty:
                continue
            inter = other.intersection(poly)
            if inter.is_empty or inter.area < 0.5:
                continue
            n += 1
            kinds.append(str(getattr(rec, "kind", "road") or "road"))
            w = getattr(rec, "width_m", None)
            if w is not None and np.isfinite(float(w)):
                widths.append(float(w))
        lot_n = sum(1 for k in kinds if k == "lot")
        kind = "lot" if n and lot_n >= (n / 2.0) else "road"
        rows.append(
            {
                "id": i,
                "n_rib": n,
                "kind": kind,
                "width_m": None if not widths else round(float(np.median(widths)), 2),
                "area_m2": round(float(poly.area), 1),
                "geometry": poly,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(columns=empty_cols, crs=gdf.crs)
    return gpd.GeoDataFrame(rows, crs=gdf.crs)
