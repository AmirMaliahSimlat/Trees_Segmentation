"""In-memory water mask editor: load the ortho once, pan/zoom without re-reading GeoTIFF.

OEM water is the starting mask. Click a blob to delete it, or paint/erase with brush or lasso.
Does not overwrite frozen OEM previews (_preview_water_*.jpg) or other task checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[2]
TILE = 256
RGB_CACHE = 768
MASK_CACHE = 256
UNDO_MAX = 40
OVERVIEW_CELL = 80
BLOCK_ROLES = ("NW", "N", "NE", "W", "center", "E", "SW", "S", "SE")
STEM_RC = re.compile(r"^(?P<pre>.*)_r(?P<r>\d+)_c(?P<c>\d+)$")
_TILE_WATER_SHP = re.compile(r"^(?P<stem>.+_r\d+_c\d+)_water(?P<edit>_edit)?$")
PROTECTED_PREVIEWS = {"_preview_water_FortRiley_r51200_c40960.jpg"}


def _log(path: Path, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _max_level(width: int, height: int) -> int:
    return max(0, math.ceil(math.log2(max(width, height))))


def _downsample_plane(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize a tile/pyramid plane. Binary masks use float AREA then keep any water."""
    if out_w < 1 or out_h < 1:
        return arr
    if arr.shape[1] == out_w and arr.shape[0] == out_h:
        return arr
    if arr.ndim == 2:
        shrinking = out_w < arr.shape[1] or out_h < arr.shape[0]
        if shrinking:
            plane = cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
            return np.where(plane > 0, np.uint8(255), np.uint8(0))
        return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    interp = cv2.INTER_AREA if (out_w < arr.shape[1] or out_h < arr.shape[0]) else cv2.INTER_LINEAR
    return cv2.resize(arr, (out_w, out_h), interpolation=interp)


def _bbox_from_mask(region: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(region)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _parse_seeds(items: list) -> list[tuple[int, int]]:
    seeds: list[tuple[int, int]] = []
    for p in items:
        try:
            seeds.append((int(p["x"]), int(p["y"])))
        except (KeyError, TypeError, ValueError):
            continue
    return seeds


def _index_record(
    x: int, y: int, w: int, h: int, cx: float, cy: float, area: int, kept: bool, gsd2: float
) -> dict:
    return {
        "id": 0,
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "cx": float(cx),
        "cy": float(cy),
        "area_px": int(area),
        "area_m2": float(int(area) * gsd2),
        "kept": bool(kept),
        "i": 0,
    }


def _sort_index(records: list[dict]) -> list[dict]:
    records.sort(key=lambda b: (b["area_px"], b["y"], b["x"]))
    for j, b in enumerate(records):
        b["i"] = j
        b["id"] = j + 1
    return records


def _label_touches_border(labels: np.ndarray, lab: int) -> bool:
    return bool(
        np.any(labels[0] == lab)
        or np.any(labels[-1] == lab)
        or np.any(labels[:, 0] == lab)
        or np.any(labels[:, -1] == lab)
    )


def _encode_jpeg(rgb: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _encode_png_rgba(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


@dataclass(frozen=True)
class GridCell:
    stem: str
    r: int
    c: int
    ri: int
    ci: int
    height: int
    width: int


@dataclass
class BlockSlice:
    stem: str
    y0: int
    x0: int
    height: int
    width: int
    transform: Affine
    raw_path: Path | None


def _join_water_mask(mask: np.ndarray) -> np.ndarray:
    """Close 1 px gaps so lasso/brush water joins the adjacent OEM blob."""
    m = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def _collect_tile_water_shps(out_dir: Path, combined: Path) -> list[Path]:
    combined_res = combined.resolve()
    by_stem: dict[str, Path] = {}
    for path in out_dir.glob("*.shp"):
        if path.resolve() == combined_res:
            continue
        hit = _TILE_WATER_SHP.match(path.stem)
        if not hit:
            continue
        stem = hit.group("stem")
        prev = by_stem.get(stem)
        if prev is None:
            by_stem[stem] = path
            continue
        newer = path.stat().st_mtime > prev.stat().st_mtime + 0.5
        older = path.stat().st_mtime < prev.stat().st_mtime - 0.5
        if newer or (not older and not hit.group("edit")):
            by_stem[stem] = path
    return [by_stem[k] for k in sorted(by_stem)]


def _write_map_water_shapefile(out_dir: Path, map_name: str) -> Path | None:
    import geopandas as gpd
    import pandas as pd
    from shapely.ops import unary_union

    combined = out_dir / f"{map_name}_water.shp"
    files = _collect_tile_water_shps(out_dir, combined)
    if not files:
        return None
    frames: list = []
    for shp in files:
        try:
            gdf = gpd.read_file(shp)
        except Exception:
            continue
        if gdf.empty:
            continue
        frames.append(gdf)
    if not frames:
        return None
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    geoms_in = [g for g in merged.geometry.values if g is not None and not g.is_empty]
    if not geoms_in:
        return None
    unioned = unary_union(geoms_in)
    if unioned is None or unioned.is_empty:
        return None
    if unioned.geom_type == "Polygon":
        geoms = [unioned]
    elif unioned.geom_type == "MultiPolygon":
        geoms = list(unioned.geoms)
    elif unioned.geom_type == "GeometryCollection":
        geoms = []
        for part in unioned.geoms:
            if part.geom_type == "Polygon":
                geoms.append(part)
            elif part.geom_type == "MultiPolygon":
                geoms.extend(part.geoms)
    else:
        geoms = [unioned]
    out = gpd.GeoDataFrame(
        {
            "id": list(range(1, len(geoms) + 1)),
            "area_m2": [float(g.area) for g in geoms],
            "geometry": geoms,
        },
        crs=merged.crs,
    )
    try:
        out.to_file(combined, driver="ESRI Shapefile")
        return combined
    except PermissionError:
        alt = out_dir / f"{map_name}_water_map.shp"
        out.to_file(alt, driver="ESRI Shapefile")
        return alt


def _parse_stem_rc(stem: str) -> tuple[int, int] | None:
    m = STEM_RC.search(stem)
    if not m:
        return None
    return int(m.group("r")), int(m.group("c"))


def _read_rgb_small(path: Path, height: int, width: int) -> np.ndarray:
    with rasterio.open(path) as ds:
        indexes = list(range(1, min(3, ds.count) + 1))
        while len(indexes) < 3:
            indexes.append(indexes[-1])
        arr = ds.read(indexes, out_shape=(len(indexes), height, width), resampling=Resampling.average)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    vmax = float(arr.max()) if arr.size else 1.0
    if vmax <= 1.5:
        vmax = 1.0
    elif vmax <= 255.0:
        vmax = 255.0
    else:
        vmax = 65535.0
    return np.clip(arr * (255.0 / vmax), 0, 255).astype(np.uint8)


def _read_mask_small(path: Path, height: int, width: int) -> np.ndarray:
    with rasterio.open(path) as ds:
        m = ds.read(1, out_shape=(height, width), resampling=Resampling.nearest)
    return (m > 0).astype(np.uint8)


def list_image_grid(image_dir: Path) -> list[GridCell]:
    raw: list[tuple[str, int, int]] = []
    for tif in image_dir.glob("*.tif"):
        rc = _parse_stem_rc(tif.stem)
        if rc is None:
            continue
        raw.append((tif.stem, rc[0], rc[1]))
    if not raw:
        return []
    rows = sorted({r for _, r, _ in raw})
    cols = sorted({c for _, _, c in raw})
    row_i = {r: i for i, r in enumerate(rows)}
    col_i = {c: i for i, c in enumerate(cols)}
    cells: list[GridCell] = []
    for stem, r, c in raw:
        cells.append(GridCell(stem=stem, r=r, c=c, ri=row_i[r], ci=col_i[c], height=10240, width=10240))
    cells.sort(key=lambda t: (t.ri, t.ci))
    return cells


def _grid_index(cells: list[GridCell]) -> tuple[dict[tuple[int, int], GridCell], int, int]:
    by_rc = {(t.ri, t.ci): t for t in cells}
    n_rows = max((t.ri for t in cells), default=-1) + 1
    n_cols = max((t.ci for t in cells), default=-1) + 1
    return by_rc, n_rows, n_cols


def _block_cells(by_rc: dict[tuple[int, int], GridCell], ri: int, ci: int) -> list[GridCell] | None:
    out: list[GridCell] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            cell = by_rc.get((ri + dr, ci + dc))
            if cell is None:
                return None
            out.append(cell)
    return out


def _is_selectable(by_rc: dict[tuple[int, int], GridCell], n_rows: int, n_cols: int, ri: int, ci: int) -> bool:
    if ri <= 0 or ci <= 0 or ri >= n_rows - 1 or ci >= n_cols - 1:
        return False
    return _block_cells(by_rc, ri, ci) is not None


class LruBytes:
    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self._data: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)
            return val

    def put(self, key: tuple, val: bytes) -> None:
        with self._lock:
            self._data[key] = val
            self._data.move_to_end(key)
            while len(self._data) > self.maxlen:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


@dataclass
class UndoSlice:
    r0: int
    r1: int
    c0: int
    c1: int
    pixels: np.ndarray


@dataclass
class WaterEditSession:
    image_dir: Path
    mask_dir: Path
    out_dir: Path
    log_path: Path
    only_stems: list[str] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    stem: str = ""
    rgb: np.ndarray | None = None
    mask: np.ndarray | None = None
    transform: Affine | None = None
    crs: object | None = None
    gsd: float = 0.3
    gen: int = 0
    dirty: bool = False
    water_px: int | None = None
    raw_path: Path | None = None
    block_slices: list[BlockSlice] = field(default_factory=list)
    overview_bytes: bytes | None = None
    overview_meta: dict | None = None
    overview_lock: threading.Lock = field(default_factory=threading.Lock)
    rgb_cache: LruBytes = field(default_factory=lambda: LruBytes(RGB_CACHE))
    mask_cache: LruBytes = field(default_factory=lambda: LruBytes(MASK_CACHE))
    rgb_pyr: dict[int, np.ndarray] = field(default_factory=dict)
    mask_pyr: dict[int, np.ndarray] = field(default_factory=dict)
    undo: list[list[UndoSlice]] = field(default_factory=list)
    blob_index: list[dict] | None = None
    blob_index_gen: int = -1
    hole_index: list[dict] | None = None
    hole_index_gen: int = -1
    kept: list[tuple[int, int]] = field(default_factory=list)
    kept_holes: list[tuple[int, int]] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        if self.rgb is None:
            return (0, 0)
        h, w = self.rgb.shape[:2]
        return w, h

    def _clear_tile_caches(self) -> None:
        self.rgb_cache.clear()
        self.mask_cache.clear()
        self.rgb_pyr.clear()
        self.mask_pyr.clear()
        self.blob_index = None
        self.hole_index = None

    def _kept_path(self) -> Path:
        return self.out_dir / f"{self.stem}_kept.json"

    def _load_kept_unlocked(self) -> None:
        self.kept = []
        self.kept_holes = []
        path = self._kept_path()
        if not self.stem or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.kept = _parse_seeds(data.get("seeds") or [])
        self.kept_holes = _parse_seeds(data.get("holes") or [])

    def _save_kept(self) -> None:
        if not self.stem:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "stem": self.stem,
            "seeds": [{"x": x, "y": y} for x, y in self.kept],
            "holes": [{"x": x, "y": y} for x, y in self.kept_holes],
        }
        self._kept_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _kept_labels(self, labels: np.ndarray, seeds: list[tuple[int, int]]) -> tuple[set[int], list[tuple[int, int]]]:
        h, w = labels.shape
        protected: set[int] = set()
        alive: list[tuple[int, int]] = []
        for x, y in seeds:
            if 0 <= x < w and 0 <= y < h:
                lab = int(labels[y, x])
                if lab > 0:
                    protected.add(lab)
                    alive.append((x, y))
        return protected, alive

    def _mask_paths(self, stem: str) -> tuple[Path | None, Path | None]:
        working = self.mask_dir / f"{stem}_water.tif"
        raw = self.mask_dir / f"{stem}_water_raw.tif"
        return (working if working.is_file() else None, raw if raw.is_file() else None)

    def _load_mask_array(self, stem: str, height: int, width: int) -> tuple[np.ndarray, Path | None]:
        working, raw = self._mask_paths(stem)
        path = working or raw
        if path is None:
            return np.zeros((height, width), dtype=np.uint8), raw
        with rasterio.open(path) as ds:
            mask = (ds.read(1) > 0).astype(np.uint8) * 255
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8) * 255
        return mask, raw or path

    def discover(self) -> list[dict[str, str | int | bool]]:
        rows = []
        if not self.image_dir.is_dir():
            return rows
        cells = list_image_grid(self.image_dir)
        by_rc, n_rows, n_cols = _grid_index(cells)
        for cell in cells:
            if self.only_stems is not None and cell.stem not in self.only_stems:
                continue
            working, raw = self._mask_paths(cell.stem)
            rows.append(
                {
                    "stem": cell.stem,
                    "has_working": working is not None,
                    "has_raw": raw is not None,
                    "selectable": _is_selectable(by_rc, n_rows, n_cols, cell.ri, cell.ci),
                    "ri": cell.ri,
                    "ci": cell.ci,
                }
            )
        return rows

    def overview_payload(self) -> tuple[bytes, dict]:
        with self.overview_lock:
            if self.overview_bytes is not None and self.overview_meta is not None:
                meta = dict(self.overview_meta)
                meta["current"] = self.stem or None
                meta["block"] = [s.stem for s in self.block_slices]
                return self.overview_bytes, meta
            jpg, meta = self._build_overview()
            self.overview_bytes = jpg
            self.overview_meta = meta
            meta = dict(meta)
            meta["current"] = self.stem or None
            meta["block"] = [s.stem for s in self.block_slices]
            return jpg, meta

    def _overview_cache_paths(self) -> tuple[Path, Path]:
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        return self.mask_dir / "_map_overview.jpg", self.mask_dir / "_map_overview.json"

    def _build_overview(self) -> tuple[bytes, dict]:
        jpg_path, json_path = self._overview_cache_paths()
        cells = list_image_grid(self.image_dir)
        by_rc, n_rows, n_cols = _grid_index(cells)
        stamp = {
            "n_tiles": len(cells),
            "cell": OVERVIEW_CELL,
            "n_rows": n_rows,
            "n_cols": n_cols,
        }
        if jpg_path.is_file() and json_path.is_file():
            try:
                cached = json.loads(json_path.read_text(encoding="utf-8"))
                if cached.get("stamp") == stamp and cached.get("width"):
                    _log(self.log_path, f"overview cache {jpg_path.name}")
                    return jpg_path.read_bytes(), cached
            except Exception:
                pass
        _log(self.log_path, f"building map overview {n_rows}x{n_cols} cells={len(cells)}")
        cell_px = OVERVIEW_CELL
        ow, oh = n_cols * cell_px, n_rows * cell_px
        canvas = np.zeros((oh, ow, 3), dtype=np.uint8)
        out_cells = []
        for cell in cells:
            x0, y0 = cell.ci * cell_px, cell.ri * cell_px
            tif = self.image_dir / f"{cell.stem}.tif"
            patch = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)
            if tif.is_file():
                try:
                    patch = _read_rgb_small(tif, cell_px, cell_px)
                except Exception as exc:
                    _log(self.log_path, f"overview skip {cell.stem}: {exc}")
            working, raw = self._mask_paths(cell.stem)
            mpath = working or raw
            if mpath is not None:
                try:
                    water = _read_mask_small(mpath, cell_px, cell_px)
                    on = water > 0
                    patch = patch.copy()
                    patch[on, 0] = (patch[on, 0].astype(np.uint16) * 2 // 5).astype(np.uint8)
                    patch[on, 1] = np.clip(patch[on, 1].astype(np.uint16) + 70, 0, 255).astype(np.uint8)
                    patch[on, 2] = np.clip(patch[on, 2].astype(np.uint16) + 110, 0, 255).astype(np.uint8)
                except Exception:
                    pass
            canvas[y0 : y0 + cell_px, x0 : x0 + cell_px] = patch
            selectable = _is_selectable(by_rc, n_rows, n_cols, cell.ri, cell.ci)
            out_cells.append(
                {
                    "stem": cell.stem,
                    "ri": cell.ri,
                    "ci": cell.ci,
                    "r": cell.r,
                    "c": cell.c,
                    "x": x0,
                    "y": y0,
                    "w": cell_px,
                    "h": cell_px,
                    "selectable": selectable,
                    "has_mask": mpath is not None,
                }
            )
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not ok:
            raise RuntimeError("overview jpeg failed")
        jpg = bytes(buf)
        meta = {
            "stamp": stamp,
            "width": ow,
            "height": oh,
            "cell": cell_px,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "cells": out_cells,
        }
        jpg_path.write_bytes(jpg)
        json_path.write_text(json.dumps(meta), encoding="utf-8")
        _log(self.log_path, f"wrote overview {jpg_path}")
        return jpg, meta

    def open(self, stem: str) -> None:
        tif = self.image_dir / f"{stem}.tif"
        if not tif.is_file():
            raise FileNotFoundError(tif)
        working, raw = self._mask_paths(stem)
        mask_path = working or raw
        if mask_path is None:
            raise FileNotFoundError(f"No OEM water mask for {stem}")
        _log(self.log_path, f"open {stem} image={tif.name} mask={mask_path.name}")
        from tree_seg.io_geotiff import read_rgb

        with rasterio.open(tif) as ds:
            rgb = read_rgb(ds)
            transform = ds.transform
            crs = ds.crs
            gsd = float(max(abs(ds.transform.a), abs(ds.transform.e)))
            height, width = int(ds.height), int(ds.width)
        mask, raw_used = self._load_mask_array(stem, height, width)
        slice_ = BlockSlice(
            stem=stem, y0=0, x0=0, height=height, width=width, transform=transform, raw_path=raw_used
        )
        with self.lock:
            self.stem = stem
            self.rgb = rgb
            self.mask = mask
            self.transform = transform
            self.crs = crs
            self.gsd = gsd
            self.raw_path = raw_used
            self.block_slices = [slice_]
            self.gen += 1
            self.dirty = False
            self.undo.clear()
            self.water_px = int((mask > 0).sum())
            self._clear_tile_caches()
            self._load_kept_unlocked()

    def open_block(self, center_stem: str) -> None:
        for ev in self.open_block_iter(center_stem):
            if ev.get("event") == "error":
                raise ValueError(ev.get("message") or "open-block failed")

    def open_block_iter(self, center_stem: str, neighborhood: int = 9) -> Iterator[dict]:
        cells = list_image_grid(self.image_dir)
        by_rc, n_rows, n_cols = _grid_index(cells)
        by_stem = {t.stem: t for t in cells}
        center = by_stem.get(center_stem)
        if center is None:
            raise FileNotFoundError(center_stem)
        neighborhood = 1 if int(neighborhood) == 1 else 9
        if neighborhood == 1:
            block = [center]
            roles = ["tile"]
            origin_ri, origin_ci = center.ri, center.ci
            n_side = 1
        else:
            if not _is_selectable(by_rc, n_rows, n_cols, center.ri, center.ci):
                raise ValueError("edge tile: pick a tile that has 8 neighbors")
            block = _block_cells(by_rc, center.ri, center.ci)
            if block is None:
                raise ValueError("3x3 neighborhood is incomplete")
            roles = list(BLOCK_ROLES)
            origin_ri, origin_ci = center.ri - 1, center.ci - 1
            n_side = 3
        from tree_seg.io_geotiff import read_rgb

        total = len(block)
        yield {
            "event": "start",
            "center": center_stem,
            "total": total,
            "neighborhood": neighborhood,
            "tiles": [{"stem": cell.stem, "role": roles[i] if i < len(roles) else ""} for i, cell in enumerate(block)],
        }

        with self.lock:
            self.rgb = None
            self.mask = None
            self.undo.clear()
            self._clear_tile_caches()

        tile_px = 10240
        H, W = n_side * tile_px, n_side * tile_px
        mosaic_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mosaic_mask = np.zeros((H, W), dtype=np.uint8)
        slices: list[BlockSlice] = []
        nw_transform = None
        crs = None
        gsd = 0.3
        for i, cell in enumerate(block):
            role = roles[i] if i < len(roles) else ""
            t0 = time.perf_counter()
            yield {
                "event": "tile",
                "i": i + 1,
                "total": total,
                "stem": cell.stem,
                "role": role,
                "phase": "ortho",
                "message": f"{i + 1}/{total}  {role}  {cell.stem}  reading ortho…",
            }
            tif = self.image_dir / f"{cell.stem}.tif"
            _log(self.log_path, f"open-block {cell.stem}")
            with rasterio.open(tif) as ds:
                rgb = read_rgb(ds)
                transform = ds.transform
                tile_crs = ds.crs
                tile_gsd = float(max(abs(ds.transform.a), abs(ds.transform.e)))
            h, w = rgb.shape[0], rgb.shape[1]
            working, raw = self._mask_paths(cell.stem)
            has_mask = working is not None or raw is not None
            yield {
                "event": "tile",
                "i": i + 1,
                "total": total,
                "stem": cell.stem,
                "role": role,
                "phase": "mask",
                "has_mask": has_mask,
                "message": f"{i + 1}/{total}  {role}  {cell.stem}  "
                + ("reading water mask…" if has_mask else "no mask yet (empty)"),
            }
            mask, raw_used = self._load_mask_array(cell.stem, h, w)
            y0 = (cell.ri - origin_ri) * tile_px
            x0 = (cell.ci - origin_ci) * tile_px
            y0 = max(0, min(H - 1, y0))
            x0 = max(0, min(W - 1, x0))
            rh, rw = min(h, H - y0), min(w, W - x0)
            mosaic_rgb[y0 : y0 + rh, x0 : x0 + rw] = rgb[:rh, :rw]
            mosaic_mask[y0 : y0 + rh, x0 : x0 + rw] = mask[:rh, :rw]
            slices.append(
                BlockSlice(
                    stem=cell.stem,
                    y0=y0,
                    x0=x0,
                    height=rh,
                    width=rw,
                    transform=transform,
                    raw_path=raw_used,
                )
            )
            if nw_transform is None:
                nw_transform = transform
                crs = tile_crs
                gsd = tile_gsd
            elapsed = time.perf_counter() - t0
            yield {
                "event": "tile",
                "i": i + 1,
                "total": total,
                "stem": cell.stem,
                "role": role,
                "phase": "done",
                "has_mask": has_mask,
                "elapsed_s": round(elapsed, 2),
                "message": f"{i + 1}/{total}  {role}  {cell.stem}  done ({elapsed:.1f}s)",
            }
            del rgb, mask
        if neighborhood == 1 and slices:
            sl = slices[0]
            mosaic_rgb = mosaic_rgb[sl.y0 : sl.y0 + sl.height, sl.x0 : sl.x0 + sl.width]
            mosaic_mask = mosaic_mask[sl.y0 : sl.y0 + sl.height, sl.x0 : sl.x0 + sl.width]
            H, W = mosaic_rgb.shape[0], mosaic_rgb.shape[1]
            slices[0] = BlockSlice(
                stem=sl.stem, y0=0, x0=0, height=H, width=W, transform=sl.transform, raw_path=sl.raw_path
            )
        yield {"event": "mosaic", "width": W, "height": H, "message": f"Assembled mosaic {W}×{H}"}
        _log(self.log_path, f"open-block center={center_stem} n={neighborhood} mosaic={W}x{H}")
        with self.lock:
            self.stem = center_stem
            self.rgb = mosaic_rgb
            self.mask = mosaic_mask
            self.transform = nw_transform
            self.crs = crs
            self.gsd = gsd
            self.raw_path = next((s.raw_path for s in slices if s.stem == center_stem), slices[0].raw_path if slices else None)
            self.block_slices = slices
            self.gen += 1
            self.dirty = False
            self.undo.clear()
            self.water_px = int((mosaic_mask > 0).sum())
            self._clear_tile_caches()
            self._load_kept_unlocked()
        ready = "1 tile ready" if neighborhood == 1 else "3×3 ready"
        yield {"event": "ready", "stats": self.stats(), "message": ready}

    def stats(self) -> dict:
        with self.lock:
            if self.mask is None or self.rgb is None:
                return {"open": False, "stem": self.stem or None, "block": []}
            water = self.water_px
            if water is None:
                water = int((self.mask > 0).sum())
                self.water_px = water
            w, h = self.size
            return {
                "open": True,
                "stem": self.stem,
                "block": [s.stem for s in self.block_slices],
                "n_tiles": len(self.block_slices),
                "width": w,
                "height": h,
                "max_level": _max_level(w, h),
                "tile": TILE,
                "gsd_m": self.gsd,
                "gen": self.gen,
                "dirty": self.dirty,
                "n_blobs": 0,
                "n_holes": 0,
                "water_px": water,
                "water_m2": float(water * self.gsd * self.gsd),
                "undo": len(self.undo),
                "n_kept": len(self.kept),
                "n_kept_holes": len(self.kept_holes),
            }

    def _blob_records_unlocked(self) -> list[dict]:
        if self.mask is None:
            return []
        if self.blob_index is not None and self.blob_index_gen == self.gen:
            return self.blob_index
        _n, labels, cc, centroids = cv2.connectedComponentsWithStats(
            (self.mask > 0).astype(np.uint8), connectivity=8
        )
        protected, self.kept = self._kept_labels(labels, self.kept)
        gsd2 = self.gsd * self.gsd
        blobs: list[dict] = []
        for i in range(1, cc.shape[0]):
            area = int(cc[i, cv2.CC_STAT_AREA])
            if area < 1:
                continue
            blobs.append(
                _index_record(
                    int(cc[i, cv2.CC_STAT_LEFT]),
                    int(cc[i, cv2.CC_STAT_TOP]),
                    int(cc[i, cv2.CC_STAT_WIDTH]),
                    int(cc[i, cv2.CC_STAT_HEIGHT]),
                    float(centroids[i][0]),
                    float(centroids[i][1]),
                    area,
                    i in protected,
                    gsd2,
                )
            )
        self.blob_index = _sort_index(blobs)
        self.blob_index_gen = self.gen
        return self.blob_index

    def _hole_records_unlocked(self) -> list[dict]:
        """Background components fully enclosed by water (boats, docks, etc.)."""
        if self.mask is None:
            return []
        if self.hole_index is not None and self.hole_index_gen == self.gen:
            return self.hole_index
        h, w = self.mask.shape
        inv = (self.mask == 0).astype(np.uint8)
        _n, labels, cc, centroids = cv2.connectedComponentsWithStats(inv, connectivity=4)
        protected, self.kept_holes = self._kept_labels(labels, self.kept_holes)
        gsd2 = self.gsd * self.gsd
        holes: list[dict] = []
        for i in range(1, cc.shape[0]):
            area = int(cc[i, cv2.CC_STAT_AREA])
            x = int(cc[i, cv2.CC_STAT_LEFT])
            y = int(cc[i, cv2.CC_STAT_TOP])
            bw = int(cc[i, cv2.CC_STAT_WIDTH])
            bh = int(cc[i, cv2.CC_STAT_HEIGHT])
            if area < 1:
                continue
            if x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h:
                continue
            holes.append(
                _index_record(
                    x,
                    y,
                    bw,
                    bh,
                    float(centroids[i][0]),
                    float(centroids[i][1]),
                    area,
                    i in protected,
                    gsd2,
                )
            )
        self.hole_index = _sort_index(holes)
        self.hole_index_gen = self.gen
        return self.hole_index

    def _centroid_xy(self, rec: dict) -> tuple[int, int]:
        return int(round(rec["cx"])), int(round(rec["cy"]))

    def _seed_kept(self, seeds: list[tuple[int, int]], pred) -> bool:
        for sx, sy in seeds:
            if pred(sx, sy):
                return True
        return False

    def _update_blob_index_region(self, r0: int, r1: int, c0: int, c1: int) -> None:
        if self.mask is None or self.blob_index is None:
            return
        h, w = self.mask.shape
        pr0, pr1 = max(0, r0 - 1), min(h, r1 + 1)
        pc0, pc1 = max(0, c0 - 1), min(w, c1 + 1)
        if pr1 <= pr0 or pc1 <= pc0:
            return
        crop = (self.mask[pr0:pr1, pc0:pc1] > 0).astype(np.uint8)
        _n, labels, stats, centroids = cv2.connectedComponentsWithStats(crop, connectivity=8)
        gsd2 = self.gsd * self.gsd
        fill = 254
        handled: set[int] = set()
        drop: set[int] = set()
        new_parts: list[dict] = []

        def claim_old_if(pred) -> dict | None:
            found = None
            for k, rec in enumerate(self.blob_index):
                if k in drop:
                    continue
                ox, oy = self._centroid_xy(rec)
                if 0 <= oy < h and 0 <= ox < w and pred(ox, oy):
                    drop.add(k)
                    found = rec
            return found

        for i in range(1, stats.shape[0]):
            if i in handled:
                continue
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 1:
                continue
            if not _label_touches_border(labels, i):
                x = int(stats[i, cv2.CC_STAT_LEFT]) + pc0
                y = int(stats[i, cv2.CC_STAT_TOP]) + pr0
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx = float(centroids[i][0]) + pc0
                cy = float(centroids[i][1]) + pr0
                kept = self._seed_kept(
                    self.kept,
                    lambda sx, sy, lab=i: pr0 <= sy < pr1 and pc0 <= sx < pc1 and int(labels[sy - pr0, sx - pc0]) == lab,
                )
                claim_old_if(
                    lambda ox, oy, lab=i: pr0 <= oy < pr1 and pc0 <= ox < pc1 and int(labels[oy - pr0, ox - pc0]) == lab
                )
                new_parts.append(_index_record(x, y, bw, bh, cx, cy, area, kept, gsd2))
                handled.add(i)
                continue
            ys, xs = np.where(labels == i)
            sx = int(xs[0] + pc0)
            sy = int(ys[0] + pr0)
            if self.mask[sy, sx] == 0:
                handled.add(i)
                continue
            ff_area, _, rect, _ = cv2.floodFill(self.mask, None, (sx, sy), fill, 0, 0, 8)
            rx, ry, rw, rh = (int(v) for v in rect)
            kept = self._seed_kept(
                self.kept, lambda x, y: 0 <= y < h and 0 <= x < w and self.mask[y, x] == fill
            )
            claimed = claim_old_if(lambda ox, oy: 0 <= oy < h and 0 <= ox < w and self.mask[oy, ox] == fill)
            cx, cy = (float(claimed["cx"]), float(claimed["cy"])) if claimed else (float(sx), float(sy))
            ring = self.mask[pr0:pr1, pc0:pc1]
            extra = labels[(ring == fill) & (labels > 0)]
            if extra.size:
                for lab in np.unique(extra):
                    handled.add(int(lab))
            sl = self.mask[ry : ry + rh, rx : rx + rw]
            sl[sl == fill] = 255
            if ff_area > 0:
                new_parts.append(_index_record(rx, ry, rw, rh, cx, cy, int(ff_area), kept, gsd2))
            handled.add(i)

        for k, rec in enumerate(self.blob_index):
            if k in drop:
                continue
            ox, oy = self._centroid_xy(rec)
            if pr0 <= oy < pr1 and pc0 <= ox < pc1 and self.mask[oy, ox] == 0:
                drop.add(k)

        kept_old = [rec for k, rec in enumerate(self.blob_index) if k not in drop]
        self.blob_index = _sort_index(kept_old + new_parts)
        self.blob_index_gen = self.gen

    def _update_hole_index_region(self, r0: int, r1: int, c0: int, c1: int) -> None:
        if self.mask is None or self.hole_index is None:
            return
        h, w = self.mask.shape
        pr0, pr1 = max(0, r0 - 1), min(h, r1 + 1)
        pc0, pc1 = max(0, c0 - 1), min(w, c1 + 1)
        if pr1 <= pr0 or pc1 <= pc0:
            return
        inv = (self.mask[pr0:pr1, pc0:pc1] == 0).astype(np.uint8)
        _n, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=4)
        gsd2 = self.gsd * self.gsd
        drop: set[int] = set()
        new_parts: list[dict] = []

        for i in range(1, stats.shape[0]):
            if _label_touches_border(labels, i):
                continue
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 1:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT]) + pc0
            y = int(stats[i, cv2.CC_STAT_TOP]) + pr0
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[i][0]) + pc0
            cy = float(centroids[i][1]) + pr0
            kept = self._seed_kept(
                self.kept_holes,
                lambda sx, sy, lab=i: pr0 <= sy < pr1 and pc0 <= sx < pc1 and int(labels[sy - pr0, sx - pc0]) == lab,
            )
            for k, rec in enumerate(self.hole_index):
                if k in drop:
                    continue
                ox, oy = self._centroid_xy(rec)
                if pr0 <= oy < pr1 and pc0 <= ox < pc1 and int(labels[oy - pr0, ox - pc0]) == i:
                    drop.add(k)
            new_parts.append(_index_record(x, y, bw, bh, cx, cy, area, kept, gsd2))

        for k, rec in enumerate(self.hole_index):
            if k in drop:
                continue
            bx, by, bw, bh = int(rec["x"]), int(rec["y"]), int(rec["w"]), int(rec["h"])
            if by + bh <= pr0 or by >= pr1 or bx + bw <= pc0 or bx >= pc1:
                continue
            ur0 = min(pr0, max(0, by - 1))
            ur1 = max(pr1, min(h, by + bh + 1))
            uc0 = min(pc0, max(0, bx - 1))
            uc1 = max(pc1, min(w, bx + bw + 1))
            ox, oy = self._centroid_xy(rec)
            work = self.mask[ur0:ur1, uc0:uc1].copy()
            seed_ok = ur0 <= oy < ur1 and uc0 <= ox < uc1 and work[oy - ur0, ox - uc0] == 0
            if not seed_ok:
                sub = self.mask[by : by + bh, bx : bx + bw]
                ys, xs = np.where(sub == 0)
                seed_ok = False
                if xs.size:
                    ox = int(xs[0] + bx)
                    oy = int(ys[0] + by)
                    seed_ok = ur0 <= oy < ur1 and uc0 <= ox < uc1 and work[oy - ur0, ox - uc0] == 0
            if not seed_ok:
                drop.add(k)
                continue
            cv2.floodFill(work, None, (ox - uc0, oy - ur0), 1, 0, 0, 4)
            filled = work == 1
            if filled[0].any() or filled[-1].any() or filled[:, 0].any() or filled[:, -1].any():
                drop.add(k)
                continue
            ys, xs = np.where(filled)
            area = int(xs.size)
            x = int(xs.min() + uc0)
            y = int(ys.min() + ur0)
            nw = int(xs.max() - xs.min() + 1)
            nh = int(ys.max() - ys.min() + 1)
            cx = float(xs.mean() + uc0)
            cy = float(ys.mean() + ur0)
            kept = self._seed_kept(
                self.kept_holes,
                lambda sx, sy, ff=filled, u0=uc0, v0=ur0: (
                    0 <= sy - v0 < ff.shape[0] and 0 <= sx - u0 < ff.shape[1] and ff[sy - v0, sx - u0]
                ),
            )
            drop.add(k)
            new_parts.append(_index_record(x, y, nw, nh, cx, cy, area, kept, gsd2))

        kept_old = [rec for k, rec in enumerate(self.hole_index) if k not in drop]
        self.hole_index = _sort_index(kept_old + new_parts)
        self.hole_index_gen = self.gen

    def _refresh_indexes_region(self, r0: int, r1: int, c0: int, c1: int) -> None:
        try:
            self._update_blob_index_region(r0, r1, c0, c1)
            self._update_hole_index_region(r0, r1, c0, c1)
        except Exception as exc:
            _log(self.log_path, f"index update failed: {exc}")
            self.blob_index = None
            self.hole_index = None

    def list_blobs(self) -> dict:
        with self.lock:
            if self.mask is None:
                return {"blobs": [], "holes": [], "gen": self.gen}
            return {
                "blobs": self._blob_records_unlocked(),
                "holes": self._hole_records_unlocked(),
                "gen": self.gen,
                "n_kept": len(self.kept),
                "n_kept_holes": len(self.kept_holes),
            }

    def fill_hole_at(self, x: int, y: int) -> dict:
        with self.lock:
            if self.mask is None:
                raise RuntimeError("no mask")
            h, w = self.mask.shape
            x, y = int(x), int(y)
            if not (0 <= x < w and 0 <= y < h) or int(self.mask[y, x]) != 0:
                return {"ok": False, "reason": "not a hole"}
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(self.mask, ff, (x, y), 0, flags=cv2.FLOODFILL_MASK_ONLY | 4)
            region = ff[1 : h + 1, 1 : w + 1] > 0
            if not region.any():
                return {"ok": False, "reason": "empty"}
            if region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any():
                return {"ok": False, "reason": "not enclosed"}
            r0, r1, c0, c1 = _bbox_from_mask(region)
            before = self.mask[r0:r1, c0:c1].copy()
            self.mask[region] = 255
            if not self._commit_region(r0, r1, c0, c1, before):
                return {"ok": False, "reason": "empty"}
            filled = int(region.sum())
            self.kept_holes = [
                (sx, sy) for sx, sy in self.kept_holes if not (0 <= sy < h and 0 <= sx < w and region[sy, sx])
            ]
            self._save_kept()
            _log(self.log_path, f"{self.stem} fill hole at {x},{y} px={filled}")
            return {"ok": True, "filled_px": filled, "bbox": [c0, r0, c1, r1]}

    def _commit_region(self, r0: int, r1: int, c0: int, c1: int, before: np.ndarray) -> bool:
        if self.mask is None:
            return False
        after = self.mask[r0:r1, c0:c1]
        if before.shape != after.shape or not np.any(before != after):
            return False
        self.undo.append([UndoSlice(r0, r1, c0, c1, before)])
        if len(self.undo) > UNDO_MAX:
            self.undo.pop(0)
        if self.water_px is not None:
            self.water_px += int((after > 0).sum() - (before > 0).sum())
        self.gen += 1
        self.dirty = True
        self.mask_cache.clear()
        self.mask_pyr.clear()
        self._refresh_indexes_region(r0, r1, c0, c1)
        return True

    def _push_undo(self, before: np.ndarray, after: np.ndarray) -> None:
        diff = before != after
        if not diff.any():
            return
        ys, xs = np.where(diff)
        r0, r1 = int(ys.min()), int(ys.max()) + 1
        c0, c1 = int(xs.min()), int(xs.max()) + 1
        self.undo.append([UndoSlice(r0, r1, c0, c1, before[r0:r1, c0:c1].copy())])
        if len(self.undo) > UNDO_MAX:
            self.undo.pop(0)

    def _touch(self) -> None:
        self.gen += 1
        self.dirty = True
        self.mask_cache.clear()
        self.mask_pyr.clear()
        self.blob_index = None
        self.hole_index = None
        self.water_px = None

    def delete_at(self, x: int, y: int) -> dict:
        with self.lock:
            if self.mask is None:
                raise RuntimeError("no mask")
            h, w = self.mask.shape
            x, y = int(x), int(y)
            if not (0 <= x < w and 0 <= y < h):
                return {"ok": False, "reason": "out of bounds"}
            if self.mask[y, x] == 0:
                return {"ok": False, "reason": "not water"}
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(self.mask, ff, (x, y), 0, flags=cv2.FLOODFILL_MASK_ONLY | 8)
            region = ff[1 : h + 1, 1 : w + 1] > 0
            if not region.any():
                return {"ok": False, "reason": "not water"}
            r0, r1, c0, c1 = _bbox_from_mask(region)
            before = self.mask[r0:r1, c0:c1].copy()
            self.mask[region] = 0
            if not self._commit_region(r0, r1, c0, c1, before):
                return {"ok": False, "reason": "not water"}
            removed = int(region.sum())
            self.kept = [(sx, sy) for sx, sy in self.kept if not (0 <= sy < h and 0 <= sx < w and region[sy, sx])]
            self._save_kept()
            _log(self.log_path, f"{self.stem} delete blob at {x},{y} px={removed}")
            return {"ok": True, "removed_px": removed, "bbox": [c0, r0, c1, r1]}

    def toggle_kept(self, x: int, y: int, kind: str = "blob") -> dict:
        kind = "hole" if str(kind).lower() == "hole" else "blob"
        with self.lock:
            if self.mask is None:
                raise RuntimeError("no mask")
            h, w = self.mask.shape
            x, y = int(x), int(y)
            if not (0 <= x < w and 0 <= y < h):
                return {"ok": False, "reason": "out of bounds", "kept": False, "kind": kind}
            if kind == "blob":
                if self.mask[y, x] == 0:
                    return {"ok": False, "reason": "not water", "kept": False, "kind": kind, "n_kept": len(self.kept)}
                flags = cv2.FLOODFILL_MASK_ONLY | 8
                seeds = self.kept
            else:
                if self.mask[y, x] != 0:
                    return {"ok": False, "reason": "not a hole", "kept": False, "kind": kind, "n_kept": len(self.kept_holes)}
                flags = cv2.FLOODFILL_MASK_ONLY | 4
                seeds = self.kept_holes
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(self.mask, ff, (x, y), 0, flags=flags)
            region = ff[1 : h + 1, 1 : w + 1] > 0
            if not region.any():
                return {"ok": False, "reason": "empty", "kept": False, "kind": kind}
            if kind == "hole" and (
                region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any()
            ):
                return {"ok": False, "reason": "not enclosed", "kept": False, "kind": kind, "n_kept": len(self.kept_holes)}
            inside = [(sx, sy) for sx, sy in seeds if 0 <= sy < h and 0 <= sx < w and region[sy, sx]]
            if inside:
                drop = set(inside)
                seeds = [s for s in seeds if s not in drop]
                kept = False
            else:
                seeds = list(seeds)
                seeds.append((x, y))
                kept = True
            if kind == "blob":
                self.kept = seeds
            else:
                self.kept_holes = seeds
            self._save_kept()
            n_kept = len(self.kept if kind == "blob" else self.kept_holes)
            _log(self.log_path, f"{self.stem} {kind} at {x},{y} kept={kept} n={n_kept}")
            return {"ok": True, "kept": kept, "kind": kind, "n_kept": n_kept}

    def _zero_component_at(self, b: dict, *, holes: bool, min_px: int = 1) -> tuple[UndoSlice | None, int, bool]:
        """Erase/fill a small blob or hole around its recorded bbox. Skips components >= min_px."""
        if self.mask is None:
            return None, 0, False
        h, w = self.mask.shape
        min_px = max(1, int(min_px))
        bx, by = int(b["x"]), int(b["y"])
        bw, bh = max(1, int(b["w"])), max(1, int(b["h"]))
        cy = int(round(float(b["cy"])))
        cx = int(round(float(b["cx"])))
        rad = max(bw, bh, int(math.ceil(math.sqrt(min_px)) * 2) + 2, 4)
        r0 = max(0, min(by, cy - rad))
        c0 = max(0, min(bx, cx - rad))
        r1 = min(h, max(by + bh, cy + rad + 1))
        c1 = min(w, max(bx + bw, cx + rad + 1))
        if r1 <= r0 or c1 <= c0:
            return None, 0, False
        crop = self.mask[r0:r1, c0:c1]
        if holes:
            binary = (crop == 0).astype(np.uint8)
            seeds = self.kept_holes
            connectivity = 4
            fill_val = 255
        else:
            binary = (crop > 0).astype(np.uint8)
            seeds = self.kept
            connectivity = 8
            fill_val = 0
        _n, lab, stats, _cent = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
        protected_any = False
        keep_labs: set[int] = set()
        for sx, sy in seeds:
            if r0 <= sy < r1 and c0 <= sx < c1:
                lid = int(lab[sy - r0, sx - c0])
                if lid > 0:
                    keep_labs.add(lid)
                    protected_any = True
        before = None
        delta = 0
        n_hit = 0
        for i in range(1, stats.shape[0]):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 1 or area >= min_px:
                continue
            if i in keep_labs:
                continue
            if _label_touches_border(lab, i):
                at_edge = r0 == 0 or c0 == 0 or r1 == h or c1 == w
                if not at_edge:
                    continue
            if before is None:
                before = crop.copy()
            crop[lab == i] = fill_val
            n_hit += 1
        if before is None:
            return None, 0, protected_any
        delta = int((crop > 0).sum() - (before > 0).sum())
        if delta == 0:
            return None, 0, protected_any
        return UndoSlice(r0, r1, c0, c1, before), delta, False

    def prune_small_blobs(self, min_px: int) -> dict:
        min_px = max(1, int(min_px))
        with self.lock:
            if self.mask is None:
                raise RuntimeError("no mask")
            blobs = self._blob_records_unlocked()
            pack: list[UndoSlice] = []
            n_removed = 0
            n_protected = 0
            water_delta = 0
            for b in blobs:
                if int(b["area_px"]) >= min_px:
                    continue
                sl, delta, protected = self._zero_component_at(b, holes=False, min_px=min_px)
                if protected:
                    n_protected += 1
                    continue
                if sl is None:
                    continue
                pack.append(sl)
                n_removed += 1
                water_delta += delta
            if not pack:
                return {"ok": True, "removed": 0, "protected": n_protected, "min_px": min_px}
            self.undo.append(pack)
            if len(self.undo) > UNDO_MAX:
                self.undo.pop(0)
            if self.water_px is not None:
                self.water_px += water_delta
            self.gen += 1
            self.dirty = True
            self.mask_cache.clear()
            self.mask_pyr.clear()
            self.blob_index = None
            self.hole_index = None
            _log(self.log_path, f"{self.stem} prune blobs < {min_px} px removed={n_removed} kept={n_protected}")
            return {"ok": True, "removed": n_removed, "protected": n_protected, "min_px": min_px}

    def prune_small_holes(self, min_px: int) -> dict:
        min_px = max(1, int(min_px))
        with self.lock:
            if self.mask is None:
                raise RuntimeError("no mask")
            holes = self._hole_records_unlocked()
            pack: list[UndoSlice] = []
            n_filled = 0
            n_protected = 0
            water_delta = 0
            for b in holes:
                if int(b["area_px"]) >= min_px:
                    continue
                sl, delta, protected = self._zero_component_at(b, holes=True, min_px=min_px)
                if protected:
                    n_protected += 1
                    continue
                if sl is None:
                    continue
                pack.append(sl)
                n_filled += 1
                water_delta += delta
            if not pack:
                return {"ok": True, "filled": 0, "protected": n_protected, "min_px": min_px}
            self.undo.append(pack)
            if len(self.undo) > UNDO_MAX:
                self.undo.pop(0)
            if self.water_px is not None:
                self.water_px += water_delta
            self.gen += 1
            self.dirty = True
            self.mask_cache.clear()
            self.mask_pyr.clear()
            self.blob_index = None
            self.hole_index = None
            _log(self.log_path, f"{self.stem} fill holes < {min_px} px filled={n_filled} kept={n_protected}")
            return {"ok": True, "filled": n_filled, "protected": n_protected, "min_px": min_px}

    def brush(self, points: list[tuple[int, int]], radius: int, mode: str) -> dict:
        if radius < 1:
            radius = 1
        value = 255 if mode == "fill" else 0
        with self.lock:
            if self.mask is None:
                raise RuntimeError("no mask")
            h, w = self.mask.shape
            pts = [(int(x), int(y)) for x, y in points if 0 <= int(x) < w and 0 <= int(y) < h]
            if not pts:
                return {"ok": False, "reason": "no points"}
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            r0 = max(0, min(ys) - radius)
            r1 = min(h, max(ys) + radius + 1)
            c0 = max(0, min(xs) - radius)
            c1 = min(w, max(xs) + radius + 1)
            before = self.mask[r0:r1, c0:c1].copy()
            for i, (x, y) in enumerate(pts):
                cv2.circle(self.mask, (x, y), radius, value, -1, lineType=cv2.LINE_8)
                if i:
                    cv2.line(self.mask, pts[i - 1], (x, y), value, thickness=max(1, radius * 2), lineType=cv2.LINE_8)
            if not self._commit_region(r0, r1, c0, c1, before):
                return {"ok": False, "reason": "no change"}
            return {"ok": True, "bbox": [c0, r0, c1, r1]}

    def lasso(self, points: list[tuple[int, int]], mode: str) -> dict:
        value = 255 if mode == "fill" else 0
        with self.lock:
            if self.mask is None:
                raise RuntimeError("no mask")
            h, w = self.mask.shape
            arr = np.array([(int(x), int(y)) for x, y in points], dtype=np.int32)
            if arr.shape[0] > 20000:
                step = int(math.ceil(arr.shape[0] / 20000))
                arr = np.concatenate([arr[::step], arr[-1:]], axis=0)
            if arr.shape[0] < 3:
                return {"ok": False, "reason": "need 3 points"}
            xs, ys = arr[:, 0], arr[:, 1]
            r0 = max(0, int(ys.min()))
            r1 = min(h, int(ys.max()) + 1)
            c0 = max(0, int(xs.min()))
            c1 = min(w, int(xs.max()) + 1)
            if r1 <= r0 or c1 <= c0:
                return {"ok": False, "reason": "outside image"}
            before = self.mask[r0:r1, c0:c1].copy()
            cv2.fillPoly(self.mask, [arr], int(value), lineType=cv2.LINE_8)
            if not self._commit_region(r0, r1, c0, c1, before):
                return {"ok": False, "reason": "no change"}
            return {"ok": True, "bbox": [c0, r0, c1, r1]}

    def undo_last(self) -> dict:
        with self.lock:
            if not self.undo or self.mask is None:
                return {"ok": False, "reason": "nothing to undo"}
            pack = self.undo.pop()
            if isinstance(pack, UndoSlice):
                pack = [pack]
            for sl in pack:
                current = self.mask[sl.r0 : sl.r1, sl.c0 : sl.c1]
                if self.water_px is not None:
                    self.water_px += int((sl.pixels > 0).sum() - (current > 0).sum())
                self.mask[sl.r0 : sl.r1, sl.c0 : sl.c1] = sl.pixels
            r0 = min(sl.r0 for sl in pack)
            r1 = max(sl.r1 for sl in pack)
            c0 = min(sl.c0 for sl in pack)
            c1 = max(sl.c1 for sl in pack)
            self.gen += 1
            self.mask_cache.clear()
            self.mask_pyr.clear()
            self.dirty = bool(self.undo)
            self._refresh_indexes_region(r0, r1, c0, c1)
            return {"ok": True, "undo": len(self.undo)}

    def reset_raw(self) -> dict:
        with self.lock:
            slices = list(self.block_slices)
            if self.mask is None or self.rgb is None:
                raise RuntimeError("no mask")
            shape = self.mask.shape
        if not slices:
            return {"ok": False, "reason": "no OEM raw mask"}
        raw = np.zeros(shape, dtype=np.uint8)
        n = 0
        for sl in slices:
            if sl.raw_path is None or not sl.raw_path.is_file():
                continue
            with rasterio.open(sl.raw_path) as ds:
                m = (ds.read(1) > 0).astype(np.uint8) * 255
            if m.shape[:2] != (sl.height, sl.width):
                m = cv2.resize(m, (sl.width, sl.height), interpolation=cv2.INTER_NEAREST)
                m = (m > 0).astype(np.uint8) * 255
            raw[sl.y0 : sl.y0 + sl.height, sl.x0 : sl.x0 + sl.width] = m
            n += 1
        if n == 0:
            return {"ok": False, "reason": "no OEM raw mask"}
        with self.lock:
            self.mask = raw
            self.undo.clear()
            self._touch()
            self.dirty = False
            _log(self.log_path, f"{self.stem} reset OEM tiles={n}")
            return {"ok": True, "tiles": n}

    def save(self) -> dict:
        from tree_seg.io_geotiff import write_geotiff
        from tree_seg.postprocess import mask_to_polygons

        with self.lock:
            if self.mask is None or self.rgb is None or self.transform is None:
                raise RuntimeError("nothing to save")
            mask = self.mask.copy()
            rgb = self.rgb
            transform = self.transform
            crs = self.crs
            gsd = self.gsd
            stem = self.stem
            slices = list(self.block_slices) or [
                BlockSlice(stem, 0, 0, mask.shape[0], mask.shape[1], transform, self.raw_path)
            ]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        n_water = 0
        area = 0.0
        last_shp = None
        for sl in slices:
            crop = _join_water_mask(mask[sl.y0 : sl.y0 + sl.height, sl.x0 : sl.x0 + sl.width])
            mask_path = self.out_dir / f"{sl.stem}_water.tif"
            write_geotiff(mask_path, (crop > 0).astype(np.uint8), sl.transform, crs, nodata=0, dtype="uint8")
            gdf = mask_to_polygons(crop, sl.transform, min_area_m2=25.0, pixel_size_m=gsd, connectivity=8)
            if crs is not None and not gdf.empty:
                gdf = gdf.set_crs(crs)
            shp = self.out_dir / f"{sl.stem}_water.shp"
            if not gdf.empty:
                gdf["tile_id"] = sl.stem
                gdf["source"] = "oem_water_edit"
                try:
                    gdf.to_file(shp, driver="ESRI Shapefile")
                except PermissionError:
                    shp = self.out_dir / f"{sl.stem}_water_edit.shp"
                    gdf.to_file(shp, driver="ESRI Shapefile")
                last_shp = shp
            n_water += int(len(gdf))
            area += float(gdf["area_m2"].sum()) if len(gdf) else 0.0
            saved.append({"tile": sl.stem, "mask": str(mask_path), "n_water": int(len(gdf))})
            _log(self.log_path, f"saved {sl.stem} n={len(gdf)} {mask_path.name}")
        preview = self.out_dir / f"_preview_water_edited_{stem}.jpg"
        if preview.name in PROTECTED_PREVIEWS:
            preview = self.out_dir / f"_preview_water_edited_{stem}_block.jpg"
        gdf_all = mask_to_polygons(_join_water_mask(mask), transform, min_area_m2=25.0, pixel_size_m=gsd, connectivity=8)
        if crs is not None and not gdf_all.empty:
            gdf_all = gdf_all.set_crs(crs)
        _write_preview(rgb, transform, gdf_all, preview)
        map_name = self.image_dir.parent.name
        map_shp = None
        try:
            map_shp = _write_map_water_shapefile(self.out_dir, map_name)
            if map_shp is not None:
                _log(self.log_path, f"map shapefile {map_shp.name} tiles={len(_collect_tile_water_shps(self.out_dir, map_shp))}")
        except Exception as exc:
            _log(self.log_path, f"map shapefile failed: {exc}")
        with self.lock:
            self.dirty = False
        meta = {
            "tile": stem,
            "tiles": saved,
            "n_water": n_water,
            "water_area_m2": area,
            "mask": saved[0]["mask"] if saved else None,
            "shp": str(last_shp) if last_shp is not None and last_shp.is_file() else None,
            "map_shp": str(map_shp) if map_shp is not None else None,
            "preview": str(preview),
        }
        (self.out_dir / f"{stem}_edit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with self.lock:
            self._save_kept()
        return meta

    def _render_tile(self, arr: np.ndarray, width: int, height: int, level: int, x: int, y: int, pyr: dict[int, np.ndarray]) -> np.ndarray | None:
        max_level = _max_level(width, height)
        if level < 0 or level > max_level or x < 0 or y < 0:
            return None
        factor = 1 << (max_level - level)
        dim_x = width / factor
        dim_y = height / factor
        px = x * TILE
        py = y * TILE
        if px >= dim_x or py >= dim_y:
            return None
        out_w = max(1, int(round(min(float(TILE), dim_x - px))))
        out_h = max(1, int(round(min(float(TILE), dim_y - py))))
        if factor == 1:
            sx0, sy0 = int(px), int(py)
            return arr[sy0 : sy0 + out_h, sx0 : sx0 + out_w]
        plane = pyr.get(factor)
        if plane is None:
            nw = max(1, int(round(dim_x)))
            nh = max(1, int(round(dim_y)))
            nbytes = nw * nh * (arr.shape[2] if arr.ndim == 3 else 1)
            if nbytes <= 200_000_000:
                plane = _downsample_plane(arr, nw, nh)
                pyr[factor] = plane
        if plane is not None:
            ph, pw = plane.shape[:2]
            x0 = min(pw, max(0, int(math.floor(px + 1e-9))))
            y0 = min(ph, max(0, int(math.floor(py + 1e-9))))
            crop = plane[y0 : y0 + out_h, x0 : x0 + out_w]
            if crop.size == 0:
                return None
            if crop.shape[0] != out_h or crop.shape[1] != out_w:
                crop = _downsample_plane(crop, out_w, out_h)
            return crop
        src_x0 = x * TILE * factor
        src_y0 = y * TILE * factor
        src_x1 = min(width, src_x0 + TILE * factor)
        src_y1 = min(height, src_y0 + TILE * factor)
        crop = arr[src_y0:src_y1, src_x0:src_x1]
        if crop.size == 0:
            return None
        return _downsample_plane(crop, out_w, out_h)

    def rgb_tile(self, level: int, x: int, y: int) -> bytes:
        key = (level, x, y)
        hit = self.rgb_cache.get(key)
        if hit is not None:
            return hit
        with self.lock:
            rgb = self.rgb
            w, h = self.size
            if rgb is None:
                raise RuntimeError("no image")
            tile = self._render_tile(rgb, w, h, level, x, y, self.rgb_pyr)
        if tile is None:
            tile = np.zeros((TILE, TILE, 3), dtype=np.uint8)
        data = _encode_jpeg(tile)
        self.rgb_cache.put(key, data)
        return data

    def mask_tile(self, level: int, x: int, y: int) -> bytes:
        with self.lock:
            gen = self.gen
            mask = self.mask
            w, h = self.size
            if mask is None:
                raise RuntimeError("no mask")
            key = (gen, level, x, y)
            hit = self.mask_cache.get(key)
            if hit is not None:
                return hit
            crop = self._render_tile(mask, w, h, level, x, y, self.mask_pyr)
        if crop is None:
            rgba = np.zeros((TILE, TILE, 4), dtype=np.uint8)
        else:
            rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
            on = crop > 0
            rgba[on, 0] = 255
            rgba[on, 1] = 255
            rgba[on, 2] = 255
            rgba[on, 3] = 255
        data = _encode_png_rgba(rgba)
        self.mask_cache.put(key, data)
        return data


def _to_pixels(gdf, transform: Affine, out_w: int, out_h: int) -> list[list[tuple[int, int]]]:
    rings: list[list[tuple[int, int]]] = []
    if gdf is None or getattr(gdf, "empty", True):
        return rings
    inv = ~transform

    def add_ring(coords) -> None:
        pts = []
        for x, y in coords:
            c, r = inv * (x, y)
            pts.append((int(round(c)), int(round(r))))
        if len(pts) >= 3:
            rings.append(pts)

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            add_ring(geom.exterior.coords)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                add_ring(part.exterior.coords)
    return [pts for pts in rings if any(0 <= x < out_w and 0 <= y < out_h for x, y in pts)]


def _write_preview(rgb: np.ndarray, transform: Affine, water, path: Path, size: int = 2048) -> None:
    if path.name in PROTECTED_PREVIEWS:
        raise RuntimeError(f"refusing to overwrite OEM preview {path.name}")
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    base = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    ds_t = Affine(transform.a * w / nw, transform.b, transform.c, transform.d, transform.e * h / nh, transform.f)
    im = Image.fromarray(base.copy())
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ring in _to_pixels(water, ds_t, nw, nh):
        draw.polygon(ring, fill=(0, 160, 255, 90), outline=(0, 200, 255, 230))
    mid = np.asarray(Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB"))
    gap = np.full((nh, 8, 3), 30, dtype=np.uint8)
    sheet = np.concatenate([base, gap, mid], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=88)


def build_app(session: WaterEditSession) -> FastAPI:
    app = FastAPI(title="OEM water editor")
    html_path = Path(__file__).with_name("water_edit.html")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/tiles")
    def list_tiles() -> dict:
        return {"tiles": session.discover(), "current": session.stem, "block": [s.stem for s in session.block_slices]}

    @app.get("/api/overview")
    def overview() -> dict:
        _jpg, meta = session.overview_payload()
        return meta

    @app.get("/overview.jpg")
    def overview_jpg() -> Response:
        jpg, _meta = session.overview_payload()
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/meta")
    def meta() -> dict:
        return session.stats()

    @app.get("/api/blobs")
    def blobs() -> dict:
        return session.list_blobs()

    @app.post("/api/open")
    def open_tile(body: dict) -> dict:
        stem = str(body.get("stem") or "").strip()
        if not stem:
            raise HTTPException(400, "stem required")
        try:
            session.open(stem)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return session.stats()

    @app.post("/api/open-block")
    def open_block(body: dict) -> StreamingResponse:
        stem = str(body.get("stem") or "").strip()
        if not stem:
            raise HTTPException(400, "stem required")
        neighborhood = int(body.get("neighborhood") or 9)

        def gen():
            pending: queue.Queue = queue.Queue()

            def run() -> None:
                try:
                    for ev in session.open_block_iter(stem, neighborhood=neighborhood):
                        pending.put(ev)
                except (FileNotFoundError, ValueError) as exc:
                    pending.put({"event": "error", "message": str(exc)})
                except Exception as exc:
                    pending.put({"event": "error", "message": str(exc)})
                finally:
                    pending.put(None)

            threading.Thread(target=run, daemon=True).start()
            while True:
                ev = pending.get()
                if ev is None:
                    break
                yield json.dumps(ev, ensure_ascii=False) + "\n"

        return StreamingResponse(
            gen(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/prune-blobs")
    def prune_blobs(body: dict) -> dict:
        min_px = int(body.get("min_px") or 1)
        return {**session.prune_small_blobs(min_px), **session.stats()}

    @app.post("/api/prune-holes")
    def prune_holes(body: dict) -> dict:
        min_px = int(body.get("min_px") or 1)
        return {**session.prune_small_holes(min_px), **session.stats()}

    @app.post("/api/fill-hole")
    def fill_hole(body: dict) -> dict:
        x, y = int(body["x"]), int(body["y"])
        return {**session.fill_hole_at(x, y), **session.stats()}

    @app.post("/api/delete")
    def delete_blob(body: dict) -> dict:
        x, y = int(body["x"]), int(body["y"])
        return {**session.delete_at(x, y), **session.stats()}

    @app.post("/api/keep")
    def keep_blob(body: dict) -> dict:
        x, y = int(body["x"]), int(body["y"])
        kind = str(body.get("kind") or "blob")
        return {**session.toggle_kept(x, y, kind), **session.stats()}

    @app.post("/api/brush")
    def brush(body: dict) -> dict:
        pts = [(int(p["x"]), int(p["y"])) for p in body.get("points") or []]
        radius = int(body.get("radius") or 12)
        mode = str(body.get("mode") or "erase")
        if mode not in {"fill", "erase"}:
            raise HTTPException(400, "mode must be fill or erase")
        return {**session.brush(pts, radius, mode), "dirty": True}

    @app.post("/api/lasso")
    def lasso(body: dict) -> dict:
        pts = [(int(p["x"]), int(p["y"])) for p in body.get("points") or []]
        mode = str(body.get("mode") or "erase")
        if mode not in {"fill", "erase"}:
            raise HTTPException(400, "mode must be fill or erase")
        return {**session.lasso(pts, mode), "dirty": True}

    @app.post("/api/undo")
    def undo() -> dict:
        return {**session.undo_last(), **session.stats()}

    @app.post("/api/reset")
    def reset() -> dict:
        return {**session.reset_raw(), **session.stats()}

    @app.post("/api/save")
    def save() -> dict:
        return session.save()

    @app.get("/tiles/rgb/{level}/{x}/{y}.jpg")
    def rgb_tile(level: int, x: int, y: int) -> Response:
        if session.rgb is None:
            raise HTTPException(404, "no image")
        data = session.rgb_tile(level, x, y)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/tiles/mask/{level}/{x}/{y}.png")
    def mask_tile(level: int, x: int, y: int) -> Response:
        if session.mask is None:
            empty = np.zeros((TILE, TILE, 4), dtype=np.uint8)
            return Response(_encode_png_rgba(empty), media_type="image/png", headers={"Cache-Control": "no-store"})
        data = session.mask_tile(level, x, y)
        return Response(data, media_type="image/png", headers={"Cache-Control": "no-store"})

    return app


def launch(
    *,
    map_name: str = "Fort_Riley",
    tile: str = "",
    host: str = "127.0.0.1",
    port: int = 7861,
    only_stems: list[str] | None = None,
) -> None:
    import uvicorn

    image_dir = ROOT / "data" / map_name / "Imagery"
    mask_dir = ROOT / "outputs" / "footprints" / map_name / "water_oem"
    session = WaterEditSession(
        image_dir=image_dir,
        mask_dir=mask_dir,
        out_dir=mask_dir,
        log_path=ROOT / "outputs" / "logs" / "water_edit.log",
        only_stems=only_stems,
    )
    threading.Thread(target=lambda: session.overview_payload(), daemon=True).start()
    if tile:
        try:
            session.open_block(tile)
        except (FileNotFoundError, ValueError) as exc:
            print(f"could not open 3x3 around {tile}: {exc}", flush=True)
    app = build_app(session)
    print(f"Water editor  http://{host}:{port}  (map overview; click an interior tile)", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OEM water editor (load ortho once, pan/zoom in RAM)")
    parser.add_argument("--map", default="Fort_Riley")
    parser.add_argument("--tile", default="", help="Optional center stem; if set, load that 3x3 at startup")
    parser.add_argument("--tiles", nargs="*", default=None, help="Limit the tile dropdown to these stems")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)
    launch(map_name=args.map, tile=args.tile, host=args.host, port=args.port, only_stems=args.tiles)
    return 0
