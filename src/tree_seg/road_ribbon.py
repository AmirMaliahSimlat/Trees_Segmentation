"""Grow an asymmetric road ribbon from edited polylines.

The line is a seed on pavement, not a centerline. Left and right extents are
measured independently; each feature keeps one total width W. Outlier stations
are retried nearby, then dropped — never blended into W.

With dyn_width on (off by default), each line still keeps one base width, but a
stretch that holds a different, stable width for at least dyn_min_run_m
overrides the base on that stretch only, per side, so a parking apron on the
right widens the right edge there while the neighbouring stretches stay at the
base width. Short stretches, and small differences from the base, stay noise:
the edge keeps running straight.

Stations whose seed pixel is clearly not pavement (vegetation, water, painted
objects) do not vote for W. Lines with many of those hits are sampled denser.
Remaining thin / heavily covered lines can be remeasured on a temporary
object-cleaned corridor (OEM + LaMa).

At T / + / Y junctions, each inner wedge between two adjacent arms gets one
quarter-astroid. The origin is halfway between that pair's mid-width center
(seed crossing plus the two ribbon mid-width offsets) and the inner outline
crossing of those two ribbons, so each quarter has its own center. The two
cusps sit on the inner outlines where pavement becomes grass (found by
walking the curb, then slid a little farther along it). The sides follow
those inner outlines from the outline crossing to the cusps; the outer lip
is the fitted arc (k can be below 2 when the lip is round, or above 2 when
it pinches in). A pair of lines can make more than one hub (split-merge
island). A ~180° through-side is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, substring, unary_union
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
    junction_arm_m: float = 4.0
    junction_max_m: float = 50.0
    junction_snap_m: float = 1.5
    junction_min_arms: int = 3
    # Opt-in. With this off every line keeps one width end to end. When on, the
    # line keeps one base width and a stretch that holds a different width for
    # at least dyn_min_run_m overrides it there, its neighbours staying at base.
    dyn_width: bool = False
    # Only a long stretch counts, and only a clearly different width: small
    # differences would put cosmetic bumps on an otherwise straight edge.
    dyn_min_run_m: float = 18.0
    dyn_min_delta_m: float = 2.0
    dyn_rel_delta: float = 0.30
    dyn_max_grow: float = 2.5
    # Off by default: a narrow measurement is usually canopy or a parked car,
    # not pavement that really ends, so the base width stays a floor.
    dyn_shrink: bool = False
    dyn_median_m: float = 5.0
    dyn_taper_m: float = 4.0
    dyn_gap_m: float = 2.0
    # Junctions are handled by the astroid corner fills, so the width stays at
    # base within this reach of a T / + node instead of overlapping them.
    dyn_junction_m: float = 12.0


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

    Yellow/white paint and tan/gray/blue-teal asphalt are kept. Green canopy,
    water, and strongly colored objects (cars, red roofs) are skipped.
    """
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    eg = 2.0 * g - r - b
    if g >= r + 6.0 and g >= b + 6.0 and eg >= 12.0:
        return True
    # Water is blue against green, not just against red: dark fresh asphalt is
    # blue-teal (g and b both far above r, but within ~14 of each other), so a
    # narrow b-vs-g margin here threw whole roads away as water.
    if b >= g + 16.0 and b >= r + 30.0 and b >= 48.0:
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


def _odd_win(span_m: float, step_m: float) -> int:
    win = int(round(span_m / max(step_m, 0.4)))
    if win % 2 == 0:
        win += 1
    return max(1, win)


def _moving_median(v: np.ndarray, win: int) -> np.ndarray:
    n = len(v)
    if n < 3 or win < 3:
        return v.astype(np.float64, copy=True)
    k = win if win % 2 else win + 1
    k = min(k, n if n % 2 else n - 1)
    if k < 3:
        return v.astype(np.float64, copy=True)
    pad = k // 2
    ext = np.pad(v.astype(np.float64), pad, mode="edge")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(np.median(ext[i : i + k]))
    return out


def _noise_sigma(half: np.ndarray, usable: np.ndarray) -> float:
    """Robust station-to-station scatter of one side, in metres."""
    x = half[usable]
    if len(x) < 4:
        return 0.5
    d = np.abs(np.diff(x))
    if len(d) == 0:
        return 0.5
    return float(min(max(0.15, 1.4826 * float(np.median(d)) / np.sqrt(2.0)), 3.0))


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(mask.astype(np.int8)))
    edges = np.concatenate([[0], idx + 1, [len(mask)]])
    return [(int(a), int(b)) for a, b in zip(edges, edges[1:]) if mask[a]]


def _exception_runs(
    v: np.ndarray, base: float, gate: float, n_min: int, n_gap: int, allow_shrink: bool
) -> list[tuple[int, int, float]]:
    """Stretches of at least n_min samples holding a different, stable width."""
    dev = v - base
    out: list[tuple[int, int, float]] = []
    signs = (1.0, -1.0) if allow_shrink else (1.0,)
    for sgn in signs:
        strong = (dev * sgn) > gate
        weak = (dev * sgn) > 0.5 * gate
        runs = _mask_runs(strong)
        merged: list[tuple[int, int]] = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= n_gap and bool(weak[merged[-1][1] : a].all()):
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for a, b in merged:
            if b - a < n_min:
                continue
            seg = v[a:b]
            val = float(np.median(seg))
            if abs(val - base) <= gate:
                continue
            # A ramp or a noisy patch is not a stable width.
            spread = 1.4826 * float(np.median(np.abs(seg - val)))
            if spread > max(1.0, 0.6 * abs(val - base)):
                continue
            out.append((a, b, val))
    out.sort(key=lambda t: t[0])
    return out


def _hub_blocked(
    stations: list[tuple[float, float, float]], hubs: list[Point] | None, reach_m: float
) -> np.ndarray | None:
    """Stations within reach_m of a T / + node, where the astroids take over."""
    n = len(stations)
    if not hubs or n == 0 or reach_m <= 0:
        return None
    xs = np.array([st[1] for st in stations], dtype=np.float64)
    ys = np.array([st[2] for st in stations], dtype=np.float64)
    lo_x, hi_x = float(xs.min()) - reach_m, float(xs.max()) + reach_m
    lo_y, hi_y = float(ys.min()) - reach_m, float(ys.max()) + reach_m
    out = np.zeros(n, dtype=bool)
    r2 = reach_m * reach_m
    for hub in hubs:
        hx, hy = float(hub.x), float(hub.y)
        if hx < lo_x or hx > hi_x or hy < lo_y or hy > hi_y:
            continue
        out |= ((xs - hx) ** 2 + (ys - hy) ** 2) <= r2
    return out if out.any() else None


def _trim_runs(
    segs: list[tuple[int, int, float]], v: np.ndarray, blocked: np.ndarray, n_min: int
) -> list[tuple[int, int, float]]:
    """Cut runs back out of blocked stations, dropping what is then too short."""
    if not segs or not blocked.any():
        return segs
    out: list[tuple[int, int, float]] = []
    for i, j, _val in segs:
        for a, b in _mask_runs(~blocked[i:j]):
            if b - a < n_min:
                continue
            out.append((i + a, i + b, float(np.median(v[i + a : i + b]))))
    return out


def _side_segments(
    half: np.ndarray,
    usable: np.ndarray,
    step_m: float,
    p: RibbonParams,
    base: float | None = None,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float]], float, float]:
    """One base offset for the whole side, overridden only on exception runs."""
    n = len(half)
    if base is None:
        base = float(np.median(half[usable])) if int(usable.sum()) else p.min_half_m
    s_idx = np.arange(n, dtype=np.float64)
    v = _interp_from_mask(s_idx, half, usable, base)
    v = _moving_median(v, _odd_win(p.dyn_median_m, step_m))
    # A different width has to beat the local scatter, else every shadow counts.
    gate = max(p.dyn_min_delta_m, p.dyn_rel_delta * base, 2.0 * _noise_sigma(half, usable))
    n_min = max(2, int(round(p.dyn_min_run_m / max(step_m, 0.4))))
    n_gap = max(1, int(round(p.dyn_gap_m / max(step_m, 0.4))))
    segs = _exception_runs(v, base, gate, n_min, n_gap, p.dyn_shrink)
    grow_cap = p.dyn_max_grow * base
    segs = [(i, j, val) for i, j, val in segs if val <= grow_cap]
    if blocked is not None:
        segs = _trim_runs(segs, v, blocked, n_min)
        segs = [(i, j, val) for i, j, val in segs if abs(val - base) > gate and val <= grow_cap]
    prof = np.full(n, max(base, p.min_half_m), dtype=np.float64)
    for i, j, val in segs:
        prof[i:j] = max(val, p.min_half_m)
    return prof, segs, base, gate


def _band_fracs(
    line: LineString,
    stations: list[tuple[float, float, float]],
    i: int,
    j: int,
    lo: float,
    hi: float,
    sign: float,
    rgb: np.ndarray,
    transform,
) -> tuple[float, float]:
    """Paved share and clearly-not-road share of the band between lo and hi."""
    if hi - lo < 0.5:
        return 0.0, 0.0
    length = float(line.length)
    rows = np.unique(np.linspace(i, j - 1, min(6, j - i)).astype(int))
    offs = np.linspace(lo + 0.35, hi - 0.2, 4)
    paved = 0
    off_road = 0
    tot = 0
    for k in rows:
        s, x, y = stations[int(k)]
        _tx, _ty, nx, ny = _tangent_at(line, s, length)
        for t in offs:
            c, r = _world_to_colrow(x + sign * nx * float(t), y + sign * ny * float(t), transform)
            px = _sample_nn(rgb, c, r)
            if px is None:
                continue
            tot += 1
            if _looks_like_pavement(px):
                paved += 1
            if _not_road_rgb(px):
                off_road += 1
    if tot == 0:
        return 0.0, 0.0
    return paved / float(tot), off_road / float(tot)


def _gate_segments(
    prof: np.ndarray,
    segs: list[tuple[int, int, float]],
    base: float,
    gate: float,
    sign: float,
    line: LineString,
    stations: list[tuple[float, float, float]],
    rgb: np.ndarray,
    transform,
) -> np.ndarray:
    """Widen only over real pavement; narrow only where the band is clearly not road.

    Shadowed asphalt reads as neither, so a stalled measurement keeps the base
    width instead of pinching the ribbon.
    """
    out = prof.copy()
    for i, j, val in segs:
        if val > base + gate:
            paved, _ = _band_fracs(line, stations, i, j, base, val, sign, rgb, transform)
            if paved < 0.6:
                out[i:j] = base
        elif val < base - gate:
            _, off_road = _band_fracs(line, stations, i, j, val, base, sign, rgb, transform)
            if off_road < 0.6:
                out[i:j] = base
    return out


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
    hubs: list[Point] | None = None,
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
    if p.dyn_width:
        lh = np.array([(a[0] if a is not None else np.nan) for a in raw], dtype=np.float64)
        rh = np.array([(a[1] if a is not None else np.nan) for a in raw], dtype=np.float64)
        usable = np.array(
            [bool(raw[i] is not None and lh[i] < cap and rh[i] < cap) for i in range(len(stations))],
            dtype=bool,
        )
        if int(usable.sum()) >= 3:
            # The base keeps the line's single width W: only the split between
            # the two sides comes from the stations whose total is close to W.
            anch = usable & np.array(
                [bool(raw[i] is not None and _is_anchor(widths[i], w_hat, p)) for i in range(len(stations))],
                dtype=bool,
            )
            base_l = base_r = None
            if int(anch.sum()) >= 3:
                bl = float(np.median(lh[anch]))
                br = float(np.median(rh[anch]))
                if bl + br > 1e-3:
                    scale = w_hat / (bl + br)
                    base_l = max(bl * scale, p.min_half_m)
                    base_r = max(br * scale, p.min_half_m)
            blocked = _hub_blocked(stations, hubs, max(p.dyn_junction_m, 0.9 * w_hat))
            left_prof, left_segs, base_l, delta_l = _side_segments(
                lh, usable, step, p, base_l, blocked
            )
            right_prof, right_segs, base_r, delta_r = _side_segments(
                rh, usable, step, p, base_r, blocked
            )
            left_prof = _gate_segments(
                left_prof, left_segs, base_l, delta_l, -1.0, line, stations, rgb, transform
            )
            right_prof = _gate_segments(
                right_prof, right_segs, base_r, delta_r, 1.0, line, stations, rgb, transform
            )
            taper = _odd_win(p.dyn_taper_m, step)
            left = np.clip(_moving_mean(left_prof, taper), p.min_half_m, None)
            right = np.clip(_moving_mean(right_prof, taper), p.min_half_m, None)
            poly = _ribbon_polygon(stations, left, right, line, smooth_win=3)
            if poly is not None:
                total = left_prof + right_prof
                w_med = float(np.median(left + right))
                steps = np.abs(np.diff(np.round(total, 2))) > 0.05
                n_samp = len(stations)
                return {
                    "geometry": poly,
                    "width_m": round(w_med, 2),
                    "w_min": round(float(total.min()), 2),
                    "w_max": round(float(total.max()), 2),
                    "n_seg": 1 + int(np.count_nonzero(steps)),
                    "n_samp": n_samp,
                    "n_in": int(ok.sum()),
                    "n_anch": int(usable.sum()),
                    "n_retry": n_retry,
                    "n_skip": int(n_skip),
                    "skip_pct": int(round(100.0 * n_skip / max(n_samp, 1))),
                    "kind": "lot" if w_med >= p.lot_width_m else "road",
                    "step_m": round(step, 3),
                }
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
        "w_min": round(w_hat, 2),
        "w_max": round(w_hat, 2),
        "n_seg": 1,
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
    "w_min",
    "w_max",
    "n_seg",
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


def _first_point(geom) -> Point | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Point":
        return geom
    if geom.geom_type == "MultiPoint" and len(geom.geoms):
        return geom.geoms[0]
    if geom.geom_type in ("LineString", "MultiLineString") and geom.length <= 2.5:
        return geom.centroid
    if geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            pt = _first_point(g)
            if pt is not None:
                return pt
    return Point(geom.representative_point())


def _cluster_points(points: list[Point], snap: float) -> list[Point]:
    clusters: list[list[Point]] = []
    for p in points:
        placed = False
        for cluster in clusters:
            cx = sum(q.x for q in cluster) / len(cluster)
            cy = sum(q.y for q in cluster) / len(cluster)
            if p.distance(Point(cx, cy)) <= snap:
                cluster.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    out: list[Point] = []
    for cluster in clusters:
        out.append(Point(sum(q.x for q in cluster) / len(cluster), sum(q.y for q in cluster) / len(cluster)))
    return out


def _endpoints_on_line(a: LineString, b: LineString, snap: float) -> list[Point]:
    """Every endpoint of a or b that sits on/near the other line."""
    out: list[Point] = []
    for line, other in ((a, b), (b, a)):
        for xy in (line.coords[0], line.coords[-1]):
            ep = Point(xy)
            if other.distance(ep) <= snap:
                out.append(nearest_points(ep, other)[1])
    return out


def _contact_points(lines: list[LineString], snap: float) -> list[Point]:
    if len(lines) < 2:
        return []
    pts: list[Point] = []
    tree = STRtree(lines)
    for i, a in enumerate(lines):
        for j in tree.query(a.buffer(snap)):
            j = int(j)
            if j <= i:
                continue
            b = lines[j]
            if a.intersects(b):
                inter = a.intersection(b)
                if inter.geom_type in ("LineString", "MultiLineString") and inter.length > 2.0:
                    continue
                pts.extend(p for p in _geom_points(inter) if p is not None and not p.is_empty)
            # A pair can meet twice (split-merge island). Keep near-miss
            # endpoints even when the lines already intersect elsewhere.
            pts.extend(_endpoints_on_line(a, b, snap))
    return _cluster_points(pts, snap)


def _as_linestring(geom) -> LineString | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString" and geom.length >= 0.5:
        return geom
    if geom.geom_type == "MultiLineString":
        parts = [g for g in geom.geoms if g is not None and not g.is_empty]
        if not parts:
            return None
        return max(parts, key=lambda g: g.length)
    return None


def _arms_at_point(lines: list[LineString], pt: Point, arm_m: float, snap: float) -> list[tuple[int, LineString]]:
    arms: list[tuple[int, LineString]] = []
    for i, line in enumerate(lines):
        if line.distance(pt) > snap:
            continue
        length = float(line.length)
        s = float(np.clip(line.project(pt), 0.0, length))
        if length - s >= 0.5:
            arm = _as_linestring(substring(line, s, s + min(arm_m, length - s)))
            if arm is not None:
                arms.append((i, arm))
        if s >= 0.5:
            arm = _as_linestring(substring(line, s - min(arm_m, s), s))
            if arm is not None:
                arms.append((i, arm))
    return arms


def junction_hubs(lines: list[LineString], p: RibbonParams) -> list[Point]:
    """Contact points where at least junction_min_arms distinct headings meet."""
    hubs: list[Point] = []
    for pt in _contact_points(lines, p.junction_snap_m):
        arms = _arms_at_point(lines, pt, max(p.junction_arm_m, 2.0), p.junction_snap_m)
        headings = [_arm_heading(arm, pt) for _, arm in arms]
        if _n_distinct_headings(headings) >= p.junction_min_arms:
            hubs.append(pt)
    return hubs


def _arm_heading(arm: LineString, pt: Point) -> float:
    coords = list(arm.coords)
    start, end = coords[0], coords[-1]
    if Point(start).distance(pt) <= Point(end).distance(pt):
        dx, dy = end[0] - start[0], end[1] - start[1]
    else:
        dx, dy = start[0] - end[0], start[1] - end[1]
    return float(np.arctan2(dy, dx))


def _kept_headings(headings: list[float], min_deg: float = 40.0) -> list[float]:
    kept: list[float] = []
    gate = np.deg2rad(min_deg)
    for h in headings:
        if all(min(abs(h - k), 2 * np.pi - abs(h - k)) >= gate for k in kept):
            kept.append(h)
    return kept


def _n_distinct_headings(headings: list[float], min_deg: float = 40.0) -> int:
    return len(_kept_headings(headings, min_deg))


def _ccw_delta(a: float, b: float) -> float:
    return float((b - a) % (2.0 * np.pi))


def _geom_points(geom) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)
    if geom.geom_type in ("LineString", "LinearRing"):
        return [Point(c) for c in geom.coords]
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        out: list[Point] = []
        for g in geom.geoms:
            out.extend(_geom_points(g))
        return out
    return []


def _ray_hit(poly, x: float, y: float, dx: float, dy: float, min_m: float, max_m: float) -> Point | None:
    u = _unit_vec(dx, dy)
    if u is None or poly is None or poly.is_empty:
        return None
    ux, uy = u
    ray = LineString([(x + min_m * ux, y + min_m * uy), (x + max_m * ux, y + max_m * uy)])
    try:
        hit = ray.intersection(poly.boundary)
    except Exception:
        return None
    pts = _geom_points(hit)
    if not pts:
        return None
    origin = Point(x, y)
    return min(pts, key=lambda q: origin.distance(q))


def _like_proto(
    rgb: np.ndarray,
    lab: np.ndarray,
    x: float,
    y: float,
    transform,
    proto: np.ndarray,
    p: RibbonParams,
) -> bool:
    col, row = _world_to_colrow(x, y, transform)
    lp = _sample_nn(lab, col, row)
    gp = _sample_nn(rgb, col, row)
    if lp is None or gp is None or _not_road_rgb(gp):
        return False
    return float(np.linalg.norm(lp.astype(np.float32) - proto)) < p.lab_cut


def _proto_from_pts(
    rgb: np.ndarray,
    lab: np.ndarray,
    transform,
    pts: list[tuple[float, float]],
) -> np.ndarray | None:
    samples: list[np.ndarray] = []
    for x, y in pts:
        col, row = _world_to_colrow(x, y, transform)
        lp = _sample_nn(lab, col, row)
        gp = _sample_nn(rgb, col, row)
        if lp is None or gp is None or _not_road_rgb(gp):
            continue
        samples.append(lp.astype(np.float32))
    if not samples:
        return None
    return np.mean(np.stack(samples, axis=0), axis=0)


def _walk_dir(
    rgb: np.ndarray,
    lab: np.ndarray,
    x: float,
    y: float,
    dx: float,
    dy: float,
    transform,
    gsd: float,
    proto: np.ndarray,
    max_m: float,
    p: RibbonParams,
) -> float:
    ln = float((dx * dx + dy * dy) ** 0.5)
    if ln < 1e-8:
        return p.min_half_m
    c, r, dc, dr = _px_offset(x, y, dx / ln, dy / ln, transform)
    return _walk_half(rgb, lab, c, r, dc, dr, proto, gsd, max_m, p)


def _wedge_dirs(
    ux: float, uy: float, vx: float, vy: float, fracs: tuple[float, ...] = (0.25, 0.5, 0.75)
) -> list[tuple[float, float]]:
    """Unit directions at fractions of the CCW wedge from u to v."""
    a0 = float(np.atan2(uy, ux))
    span = _ccw_delta(a0, float(np.atan2(vy, vx)))
    out: list[tuple[float, float]] = []
    for f in fracs:
        ang = a0 + float(f) * span
        out.append((float(np.cos(ang)), float(np.sin(ang))))
    return out


def _astroid_ray_reach(
    a: float, b: float, k: float, dx: float, dy: float, ux: float, uy: float, vx: float, vy: float
) -> float:
    """Distance from the origin to the quarter-curve along unit direction d."""
    det = ux * vy - uy * vx
    if abs(det) < 1e-10 or a < 1e-6 or b < 1e-6 or k < 1e-6:
        return 0.0
    a0 = (dx * vy - dy * vx) / det
    b0 = (ux * dy - uy * dx) / det
    if a0 <= 1e-9 and b0 <= 1e-9:
        return 0.0
    if a0 <= 1e-9:
        return b
    if b0 <= 1e-9:
        return a
    inv_k = 2.0 / k
    term = (a0 / a) ** inv_k + (b0 / b) ** inv_k
    if term <= 1e-12:
        return 0.0
    return float(term ** (-0.5 * k))


def _robust_gore_samples(
    samples: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Drop a walk that died early or ran away vs its neighbours."""
    if len(samples) < 2:
        return samples
    rs = np.array([s[2] for s in samples], dtype=np.float64)
    keep: list[tuple[float, float, float]] = []
    for i, s in enumerate(samples):
        others = np.delete(rs, i)
        ref = float(np.median(others))
        if ref >= 1.5 and (s[2] < 0.4 * ref or s[2] > 2.5 * ref):
            continue
        keep.append(s)
    return keep if keep else samples


def _fit_pinch_multi(
    a: float,
    b: float,
    samples: list[tuple[float, float, float]],
    ux: float,
    uy: float,
    vx: float,
    vy: float,
) -> float:
    """Least-squares k so the quarter-curve matches several wedge walks."""
    if not samples:
        return 2.8

    def sse(k: float) -> float:
        err = 0.0
        for dx, dy, r in samples:
            pred = _astroid_ray_reach(a, b, k, dx, dy, ux, uy, vx, vy)
            d = pred - r
            err += d * d
        return err

    best_k = 2.8
    best_e = sse(best_k)
    for k in np.linspace(1.0, 4.0, 31):
        e = sse(float(k))
        if e < best_e:
            best_k, best_e = float(k), e
    lo = max(1.0, best_k - 0.12)
    hi = min(4.0, best_k + 0.12)
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        left = 0.5 * (lo + mid)
        right = 0.5 * (mid + hi)
        if sse(left) < sse(right):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _pinch_until_inside(
    a: float,
    b: float,
    k: float,
    samples: list[tuple[float, float, float]],
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    slack: float = 0.15,
) -> float:
    """Raise k if the curve still sticks out past a walk."""
    k = float(np.clip(k, 1.0, 4.0))
    for _ in range(16):
        over = False
        for dx, dy, r in samples:
            if _astroid_ray_reach(a, b, k, dx, dy, ux, uy, vx, vy) > r + slack:
                over = True
                break
        if not over or k >= 4.0 - 1e-6:
            break
        k = min(4.0, k + 0.12)
    return k


def _astroid_corner_poly(
    cx: float,
    cy: float,
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    a: float,
    b: float,
    k: float = 3.0,
    n: int = 28,
) -> Polygon | None:
    if a < 0.5 or b < 0.5:
        return None
    ts = np.linspace(0.0, 0.5 * np.pi, n)
    pts = [(cx, cy)]
    for t in ts:
        c, s = float(np.cos(t)), float(np.sin(t))
        aa = a * (max(c, 0.0) ** k)
        bb = b * (max(s, 0.0) ** k)
        pts.append((cx + aa * ux + bb * vx, cy + aa * uy + bb * vy))
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon" or poly.is_empty or poly.area < 0.05:
        return None
    return poly


def _project_on_ring(poly, pt: Point) -> Point | None:
    geom = make_valid(poly) if poly is not None else None
    if geom is None or geom.is_empty or geom.geom_type != "Polygon":
        return None
    ring = geom.exterior
    if ring is None or ring.length < 0.5:
        return None
    return ring.interpolate(float(ring.project(pt)))


def _dedup_ring(pts: list[tuple[float, float]], tol: float = 0.06) -> list[tuple[float, float]]:
    if not pts:
        return pts
    out = [pts[0]]
    tol2 = tol * tol
    for p in pts[1:]:
        dx, dy = p[0] - out[-1][0], p[1] - out[-1][1]
        if dx * dx + dy * dy >= tol2:
            out.append(p)
    if len(out) >= 2:
        dx, dy = out[0][0] - out[-1][0], out[0][1] - out[-1][1]
        if dx * dx + dy * dy < tol2:
            out.pop()
    return out


def _astroid_arc_pts(
    cx: float, cy: float, ux: float, uy: float, vx: float, vy: float, a: float, b: float, k: float, n: int = 28
) -> list[tuple[float, float]]:
    ts = np.linspace(0.0, 0.5 * np.pi, n)
    pts: list[tuple[float, float]] = []
    for t in ts:
        c, s = float(np.cos(t)), float(np.sin(t))
        aa = a * (max(c, 0.0) ** k)
        bb = b * (max(s, 0.0) ** k)
        pts.append((cx + aa * ux + bb * vx, cy + aa * uy + bb * vy))
    return pts


def _astroid_hug_poly(
    cx: float,
    cy: float,
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    a: float,
    b: float,
    k: float,
    poly1,
    sp1: Point,
    n1x: float,
    n1y: float,
    poly2,
    sp2: Point,
    n2x: float,
    n2y: float,
    root: Point | None,
) -> Polygon | None:
    """Quarter fill: inner curbs to the cusps, then the fitted arc. Origin is not a vertex."""
    origin = Point(cx, cy)
    s1 = _project_on_ring(poly1, root) if root is not None else _project_on_ring(poly1, origin)
    s2 = _project_on_ring(poly2, root) if root is not None else _project_on_ring(poly2, origin)
    if s1 is None:
        s1 = sp1
    if s2 is None:
        s2 = sp2
    chain1 = _inner_outline_chain(poly1, s1, sp1, n1x, n1y)
    chain2 = _inner_outline_chain(poly2, sp2, s2, n2x, n2y)
    arc = _astroid_arc_pts(cx, cy, ux, uy, vx, vy, a, b, k)
    pts: list[tuple[float, float]] = []
    if root is not None:
        pts.append((root.x, root.y))
    pts.extend(chain1)
    if len(arc) > 2:
        pts.extend(arc[1:-1])
    pts.extend(chain2)
    pts = _dedup_ring(pts)
    if len(pts) < 3:
        return _astroid_corner_poly(cx, cy, ux, uy, vx, vy, a, b, k)
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon" or poly.is_empty or poly.area < 0.05:
        return _astroid_corner_poly(cx, cy, ux, uy, vx, vy, a, b, k)
    return poly


def _unit_vec(dx: float, dy: float) -> tuple[float, float] | None:
    n = float((dx * dx + dy * dy) ** 0.5)
    if n < 1e-8:
        return None
    return dx / n, dy / n


def _ribbon_poly(row: dict[str, Any] | None):
    if not row:
        return None
    geom = row.get("geometry")
    if geom is None or geom.is_empty:
        return None
    geom = make_valid(geom)
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon" and len(geom.geoms):
        return max(geom.geoms, key=lambda g: g.area)
    return None


def _mid_width_offset(
    poly,
    pt: Point,
    heading: float,
    max_half_m: float = 20.0,
) -> tuple[float, float]:
    """Vector from the seed hub to this ribbon's mid-width at that station."""
    hx, hy = float(np.cos(heading)), float(np.sin(heading))
    nx, ny = -hy, hx
    left = _ray_hit(poly, pt.x, pt.y, nx, ny, 0.05, max_half_m)
    right = _ray_hit(poly, pt.x, pt.y, -nx, -ny, 0.05, max_half_m)
    if left is None and right is None:
        return (0.0, 0.0)
    lx, ly = (left.x, left.y) if left is not None else (pt.x, pt.y)
    rx, ry = (right.x, right.y) if right is not None else (pt.x, pt.y)
    return (0.5 * (lx + rx) - pt.x, 0.5 * (ly + ry) - pt.y)


def _seed_offset_origin(
    poly1,
    h1: float,
    poly2,
    h2: float,
    pt: Point,
    max_off_m: float = 16.0,
) -> tuple[float, float] | None:
    """Seed crossing plus the two mid-width offsets."""
    o1x, o1y = _mid_width_offset(poly1, pt, h1)
    o2x, o2y = _mid_width_offset(poly2, pt, h2)
    hit = (pt.x + o1x + o2x, pt.y + o1y + o2y)
    if Point(hit[0], hit[1]).distance(pt) > max_off_m:
        return None
    return hit


def _in_wedge(x: float, y: float, ox: float, oy: float, h1: float, h2: float) -> bool:
    dx, dy = x - ox, y - oy
    if dx * dx + dy * dy < 0.04:
        return True
    span = _ccw_delta(h1, h2)
    t = _ccw_delta(h1, float(np.arctan2(dy, dx)))
    pad = min(0.08, 0.25 * span)
    return pad <= t <= span - pad


def _outline_crossings(poly1, poly2) -> list[Point]:
    if poly1 is None or poly2 is None or poly1.is_empty or poly2.is_empty:
        return []
    try:
        hit = make_valid(poly1).boundary.intersection(make_valid(poly2).boundary)
    except Exception:
        return []
    if hit is None or hit.is_empty:
        return []
    pts = [q for q in _geom_points(hit) if q is not None and not q.is_empty]
    parts = list(hit.geoms) if hit.geom_type in ("GeometryCollection", "MultiLineString", "MultiPoint") else [hit]
    for g in parts:
        if g.geom_type == "LineString" and g.length > 0.05:
            pts.append(g.interpolate(0.5, normalized=True))
    return pts


def _outline_connection(
    poly1,
    poly2,
    pt: Point,
    h1: float,
    h2: float,
    max_off_m: float = 16.0,
) -> Point | None:
    """Inner outline crossing of the two ribbons for this wedge."""
    cands = [q for q in _outline_crossings(poly1, poly2) if pt.distance(q) <= max_off_m]
    in_w = [q for q in cands if _in_wedge(q.x, q.y, pt.x, pt.y, h1, h2)]
    if in_w:
        return min(in_w, key=lambda q: pt.distance(q))
    try:
        ov = make_valid(poly1).intersection(make_valid(poly2))
    except Exception:
        ov = None
    if ov is not None and not ov.is_empty and getattr(ov, "area", 0.0) > 0.05:
        extra: list[Point] = []
        parts = ov.geoms if ov.geom_type in ("MultiPolygon", "GeometryCollection") else [ov]
        for g in parts:
            if g.geom_type != "Polygon":
                continue
            extra.extend(Point(x, y) for x, y in g.exterior.coords)
        extra = [q for q in extra if pt.distance(q) <= max_off_m and _in_wedge(q.x, q.y, pt.x, pt.y, h1, h2)]
        if extra:
            return min(extra, key=lambda q: pt.distance(q))
    try:
        a, b = nearest_points(make_valid(poly1).boundary, make_valid(poly2).boundary)
    except Exception:
        return None
    if a is None or b is None:
        return None
    mid = Point(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
    if pt.distance(mid) > max_off_m or not _in_wedge(mid.x, mid.y, pt.x, pt.y, h1, h2):
        return None
    return mid


def _wedge_origin(
    poly1,
    h1: float,
    poly2,
    h2: float,
    pt: Point,
    max_off_m: float = 16.0,
) -> tuple[float, float] | None:
    """Midpoint of the mid-width center and this wedge's outline connection."""
    center = _seed_offset_origin(poly1, h1, poly2, h2, pt, max_off_m)
    if center is None:
        return None
    conn = _outline_connection(poly1, poly2, pt, h1, h2, max_off_m)
    if conn is None:
        return center
    hit = (0.5 * (center[0] + conn.x), 0.5 * (center[1] + conn.y))
    if Point(hit[0], hit[1]).distance(pt) > max_off_m:
        return center
    return hit


def _first_outline_hit(
    cx: float, cy: float, dx: float, dy: float, poly1, poly2, max_m: float
) -> Point | None:
    hits: list[Point] = []
    for poly in (poly1, poly2):
        if poly is None or poly.is_empty:
            continue
        hit = _ray_hit(poly, cx, cy, dx, dy, 0.25, max_m)
        if hit is not None:
            hits.append(hit)
    if not hits:
        return None
    origin = Point(cx, cy)
    return min(hits, key=lambda q: origin.distance(q))


def _on_inner_curb(poly, q: Point, nx: float, ny: float) -> bool:
    """True if stepping from q back toward the strip lands inside the ribbon."""
    if poly is None or poly.is_empty or q is None:
        return False
    inside = Point(q.x - 0.45 * nx, q.y - 0.45 * ny)
    return bool(poly.contains(inside) or poly.distance(inside) < 0.12)


def _inward_outline_spike(
    poly, cx: float, cy: float, hx: float, hy: float, nx: float, ny: float, max_m: float
):
    """Hit the ribbon outline from a few stations along the arm."""
    for along in (8.0, 12.0, 4.0, 16.0):
        hit = _ray_hit(poly, cx + along * hx, cy + along * hy, nx, ny, 0.05, 16.0)
        if hit is not None and _on_inner_curb(poly, hit, nx, ny):
            return hit
    hit = _ray_hit(poly, cx, cy, hx, hy, 0.25, max_m + 6.0)
    if hit is not None and _on_inner_curb(poly, hit, nx, ny):
        return hit
    return None


def _slide_on_outline(
    poly, pt: Point, origin: Point, along_m: float, max_m: float, hx: float, hy: float, nx: float, ny: float
) -> Point:
    """Move a cusp along the inner outline, not onto the far curb."""
    geom = make_valid(poly) if poly is not None else None
    if geom is None or geom.is_empty or geom.geom_type != "Polygon":
        return pt
    ring = geom.exterior
    if ring is None or ring.length < along_m:
        return pt
    s = float(np.clip(ring.project(pt), 0.0, ring.length))
    length = float(ring.length)
    cands: list[Point] = []
    for ds in (along_m, -along_m):
        q = ring.interpolate((s + ds) % length)
        if origin.distance(q) > max_m + 1e-6:
            continue
        if not _on_inner_curb(geom, q, nx, ny):
            continue
        cands.append(q)
    if not cands:
        return pt
    return max(cands, key=lambda q: (q.x - origin.x) * hx + (q.y - origin.y) * hy)


def _inner_outline_chain(poly, p0: Point, p1: Point, nx: float, ny: float) -> list[tuple[float, float]]:
    """Vertices along the inner curb from p0 to p1 (the short on-curb way)."""
    geom = make_valid(poly) if poly is not None else None
    if geom is None or geom.is_empty or geom.geom_type != "Polygon":
        return [(p0.x, p0.y), (p1.x, p1.y)]
    ring = geom.exterior
    if ring is None or ring.length < 0.5:
        return [(p0.x, p0.y), (p1.x, p1.y)]
    length = float(ring.length)
    s0 = float(ring.project(p0))
    s1 = float(ring.project(p1))
    d_fwd = (s1 - s0) % length
    d_back = (s0 - s1) % length

    def walk(dist: float, sign: float) -> tuple[list[tuple[float, float]], int]:
        n = max(6, int(dist / 0.55) + 1)
        pts: list[tuple[float, float]] = []
        ok = 0
        for i in range(n + 1):
            q = ring.interpolate((s0 + sign * dist * i / n) % length)
            pts.append((q.x, q.y))
            if _on_inner_curb(geom, q, nx, ny):
                ok += 1
        return pts, ok

    pf, okf = walk(d_fwd, 1.0)
    pb, okb = walk(d_back, -1.0)
    use_fwd = (okf / max(len(pf), 1), -d_fwd) >= (okb / max(len(pb), 1), -d_back)
    dist = d_fwd if use_fwd else d_back
    if dist > min(length * 0.4, 80.0):
        return [(p0.x, p0.y), (p1.x, p1.y)]
    return pf if use_fwd else pb


def _inner_lip_start(poly, cx: float, cy: float, hx: float, hy: float, nx: float, ny: float, max_m: float):
    """First wedge-facing curb point near the origin."""
    for along in (1.0, 2.0, 0.6, 3.0, 4.0):
        qx, qy = cx + along * hx, cy + along * hy
        q = Point(qx, qy)
        if poly is not None and (not poly.contains(q)) and poly.distance(q) > 0.4:
            continue
        hit = _ray_hit(poly, qx, qy, nx, ny, 0.05, 12.0)
        if hit is not None and q.distance(hit) <= 12.0 and _on_inner_curb(poly, hit, nx, ny):
            return hit
    return _inward_outline_spike(poly, cx, cy, hx, hy, nx, ny, max_m)


def _inner_lip_point(
    rgb: np.ndarray,
    lab: np.ndarray,
    transform,
    gsd: float,
    proto: np.ndarray,
    p: RibbonParams,
    poly,
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    nx: float,
    ny: float,
    max_m: float,
):
    """Last inner-outline point whose gore side is still pavement (a or b)."""
    geom = make_valid(poly) if poly is not None else None
    start = _inner_lip_start(geom, cx, cy, hx, hy, nx, ny, max_m)
    if start is None:
        return None
    if geom is None or geom.geom_type != "Polygon":
        return start
    ring = geom.exterior
    if ring is None or ring.length < 1.0:
        return start
    origin = Point(cx, cy)
    length = float(ring.length)
    s0 = float(ring.project(start))
    step = max(0.4, float(gsd))
    probe = min(2.0, 3.0 * step)
    q_pos = ring.interpolate((s0 + probe) % length)
    q_neg = ring.interpolate((s0 - probe) % length)
    sign = 1.0 if (q_pos.x - cx) * hx + (q_pos.y - cy) * hy >= (q_neg.x - cx) * hx + (q_neg.y - cy) * hy else -1.0
    last = start
    along = 0.0
    max_along = min(length * 0.4, max_m + 8.0)
    while along <= max_along + 1e-6:
        q = ring.interpolate((s0 + sign * along) % length)
        if origin.distance(q) > max_m:
            break
        if not _on_inner_curb(geom, q, nx, ny):
            break
        ox, oy = q.x + 0.7 * nx, q.y + 0.7 * ny
        if _like_proto(rgb, lab, ox, oy, transform, proto, p):
            last = q
        elif last is not None and along > 0.5:
            return last
        along += step
    return last


def _junction_quarter_poly(
    rgb: np.ndarray,
    lab: np.ndarray,
    transform,
    gsd: float,
    p: RibbonParams,
    cx: float,
    cy: float,
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    a: float,
    b: float,
    poly1,
    poly2,
    max_m: float,
    sp1: Point,
    n1x: float,
    n1y: float,
    sp2: Point,
    n2x: float,
    n2y: float,
    root: Point | None,
):
    bis = _unit_vec(ux + vx, uy + vy)
    if bis is None:
        return None
    bx, by = bis
    proto = _proto_from_pts(
        rgb,
        lab,
        transform,
        [
            (cx, cy),
            (cx + 1.0 * ux, cy + 1.0 * uy),
            (cx + 1.0 * vx, cy + 1.0 * vy),
            (cx - 0.8 * bx, cy - 0.8 * by),
        ],
    )
    if proto is None:
        return None
    if ux * vy - uy * vx < 0:
        ux, uy, vx, vy, a, b = vx, vy, ux, uy, b, a
        poly1, poly2 = poly2, poly1
        sp1, sp2 = sp2, sp1
        n1x, n1y, n2x, n2y = n2x, n2y, n1x, n1y
    a = float(np.clip(a, 0.8, max_m))
    b = float(np.clip(b, 0.8, max_m))
    walks: list[tuple[float, float, float]] = []
    fracs = tuple(i / 8.0 for i in range(1, 8))
    for dx, dy in _wedge_dirs(ux, uy, vx, vy, fracs):
        r = _walk_dir(rgb, lab, cx, cy, dx, dy, transform, gsd, proto, max_m, p)
        walks.append((dx, dy, float(np.clip(r, 0.4, max_m))))
    r45 = walks[3][2] if len(walks) >= 4 else (walks[len(walks) // 2][2] if walks else p.min_half_m)
    samples = _robust_gore_samples(walks)
    k = _fit_pinch_multi(a, b, samples, ux, uy, vx, vy)
    k = _pinch_until_inside(a, b, k, samples, ux, uy, vx, vy)
    poly = _astroid_hug_poly(
        cx, cy, ux, uy, vx, vy, a, b, k, poly1, sp1, n1x, n1y, poly2, sp2, n2x, n2y, root
    )
    if poly is None:
        return None
    return poly, a, b, k, r45


def _junction_astroid_rows(
    parts: list[LineString],
    ribbons: list[dict[str, Any]],
    rgb: np.ndarray,
    lab: np.ndarray,
    transform,
    gsd: float,
    p: RibbonParams,
) -> list[dict[str, Any]]:
    extra: list[dict[str, Any]] = []
    if not parts or not ribbons:
        return extra
    n_junc = 0
    max_m = float(p.junction_max_m)
    arm_m = max(p.junction_arm_m, 2.0)
    snap = p.junction_snap_m
    min_w = np.deg2rad(14.0)
    max_w = np.deg2rad(165.0)
    for pt in _contact_points(parts, snap):
        arms = _arms_at_point(parts, pt, arm_m, snap)
        tagged: list[tuple[float, int, Any]] = []
        for idx, arm in arms:
            if idx < 0 or idx >= len(ribbons):
                continue
            poly = _ribbon_poly(ribbons[idx])
            if poly is None:
                continue
            tagged.append((_arm_heading(arm, pt), idx, poly))
        if len(tagged) < 2:
            continue
        tagged.sort(key=lambda t: t[0])
        n_here = 0
        n_tag = len(tagged)
        for i in range(n_tag):
            h1, i1, poly1 = tagged[i]
            h2, i2, poly2 = tagged[(i + 1) % n_tag]
            wedge = _ccw_delta(h1, h2)
            if wedge < min_w or wedge > max_w:
                continue
            if i1 == i2:
                continue
            origin = _wedge_origin(poly1, h1, poly2, h2, pt)
            if origin is None:
                continue
            root = _outline_connection(poly1, poly2, pt, h1, h2)
            cx, cy = origin
            hx1, hy1 = float(np.cos(h1)), float(np.sin(h1))
            hx2, hy2 = float(np.cos(h2)), float(np.sin(h2))
            n1x, n1y = -hy1, hx1
            n2x, n2y = hy2, -hx2
            proto = _proto_from_pts(
                rgb,
                lab,
                transform,
                [
                    (cx, cy),
                    (cx + 1.2 * hx1, cy + 1.2 * hy1),
                    (cx + 1.2 * hx2, cy + 1.2 * hy2),
                ],
            )
            if proto is None:
                continue
            sp1 = _inner_lip_point(
                rgb, lab, transform, gsd, proto, p, poly1, cx, cy, hx1, hy1, n1x, n1y, max_m
            )
            sp2 = _inner_lip_point(
                rgb, lab, transform, gsd, proto, p, poly2, cx, cy, hx2, hy2, n2x, n2y, max_m
            )
            if sp1 is None:
                sp1 = _first_outline_hit(cx, cy, hx1, hy1, poly1, poly2, max_m + 6.0)
            if sp2 is None:
                sp2 = _first_outline_hit(cx, cy, hx2, hy2, poly1, poly2, max_m + 6.0)
            if sp1 is None or sp2 is None:
                continue
            origin_pt = Point(cx, cy)
            sp1 = _slide_on_outline(poly1, sp1, origin_pt, 1.25, max_m, hx1, hy1, n1x, n1y)
            sp2 = _slide_on_outline(poly2, sp2, origin_pt, 1.25, max_m, hx2, hy2, n2x, n2y)
            a0 = max(0.8, float(origin_pt.distance(sp1)))
            b0 = max(0.8, float(origin_pt.distance(sp2)))
            u = _unit_vec(sp1.x - cx, sp1.y - cy)
            v = _unit_vec(sp2.x - cx, sp2.y - cy)
            if u is None or v is None:
                continue
            got = _junction_quarter_poly(
                rgb,
                lab,
                transform,
                gsd,
                p,
                cx,
                cy,
                u[0],
                u[1],
                v[0],
                v[1],
                a0,
                b0,
                poly1,
                poly2,
                max_m,
                sp1,
                n1x,
                n1y,
                sp2,
                n2x,
                n2y,
                root,
            )
            if got is None:
                continue
            poly, a, b, k, r45 = got
            extra.append(
                {
                    "src": "",
                    "src_id": "",
                    "name": "junction",
                    "fclass": "",
                    "length_m": round(0.5 * (a + b), 2),
                    "geometry": poly,
                    "width_m": round(0.5 * (a + b), 2),
                    "n_samp": 0,
                    "n_in": 0,
                    "n_anch": 0,
                    "n_retry": 0,
                    "n_skip": 0,
                    "skip_pct": 0,
                    "kind": "road",
                    "step_m": round(k, 2),
                    "w_naive": round(a, 2),
                    "w_skip": round(b, 2),
                    "w_clean": round(r45, 2),
                    "method": "junction",
                }
            )
            n_here += 1
        if n_here:
            n_junc += 1
    if n_junc:
        print(f"  junction astroids {n_junc} nodes / {len(extra)} corners", flush=True)
    return extra


def ribbons_for_gdf(
    gdf: gpd.GeoDataFrame,
    rgb: np.ndarray,
    transform,
    gsd: float,
    p: RibbonParams | None = None,
    cleaner: CorridorCleaner | None = None,
) -> gpd.GeoDataFrame:
    p = p or RibbonParams()
    lab = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2LAB)
    rows: list[dict[str, Any]] = []
    src_parts: list[LineString] = []
    pending: list[tuple[int, LineString, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]] = []
    all_parts: list[LineString] = []
    for rec in gdf.itertuples(index=False):
        all_parts.extend(_explode_lines(rec.geometry))
    hubs = junction_hubs(all_parts, p) if p.dyn_width else []
    for i, rec in enumerate(gdf.itertuples(index=False)):
        geom = rec.geometry
        attrs = rec._asdict()
        attrs.pop("geometry", None)
        for part in _explode_lines(geom):
            naive = ribbon_for_line(part, rgb, lab, transform, gsd, p, skip_not_road=False, hubs=hubs)
            skipped = ribbon_for_line(part, rgb, lab, transform, gsd, p, skip_not_road=True, hubs=hubs)
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
            rows.append(row)
            src_parts.append(part)
        if (i + 1) % 50 == 0:
            print(f"  ribbon {i + 1}/{len(gdf)}", flush=True)
    if pending and cleaner is not None:
        work = rgb.copy()
        print(f"  clean corridors {len(pending)}", flush=True)
        for k, (idx, part, row, naive, skipped) in enumerate(pending):
            half = 0.5 * float((skipped or naive or row).get("width_m") or 5.0)
            paste_clean_corridor(work, rgb, part, transform, gsd, half, p, cleaner)
            if (k + 1) % 5 == 0 or k + 1 == len(pending):
                print(f"    cleaned {k + 1}/{len(pending)}", flush=True)
        work_lab = cv2.cvtColor(work[:, :, ::-1], cv2.COLOR_BGR2LAB)
        for idx, part, row, naive, skipped in pending:
            after = ribbon_for_line(part, work, work_lab, transform, gsd, p, skip_not_road=True, hubs=hubs)
            if after is not None and _paved_frac(work, part, transform, strict=True) >= 0.5:
                before = skipped or naive
                if before is None or _accept_clean(before, after, p):
                    keep_meta = {k: row[k] for k in ("src", "src_id", "name", "fclass", "length_m", "w_naive", "w_skip")}
                    row.clear()
                    row.update(keep_meta)
                    row.update(after)
                    row["w_clean"] = round(float(after["width_m"]), 2)
                    row["method"] = "clean"
    if src_parts:
        rows.extend(_junction_astroid_rows(src_parts, rows, rgb, lab, transform, gsd, p))
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


def merge_touching_ribbons(
    gdf: gpd.GeoDataFrame,
    snap_m: float = 0.35,
) -> gpd.GeoDataFrame:
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
