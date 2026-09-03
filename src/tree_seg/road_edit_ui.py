"""In-RAM road centerline editor: 1 or 3×3 ortho mosaic, vector overlay, no CV snap.

Load OSM/TIGER as the starting map. Edits are line geometry only (instant redraw).
Click a line to branch a new spur from that point. Save writes shapefiles.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from rasterio.transform import Affine
from shapely.geometry import LineString, Point, box

from tree_seg.water_edit_ui import (
    BLOCK_ROLES,
    OVERVIEW_CELL,
    RGB_CACHE,
    TILE,
    BlockSlice,
    LruBytes,
    _block_cells,
    _downsample_plane,
    _encode_jpeg,
    _grid_index,
    _is_selectable,
    _max_level,
    _read_rgb_small,
    list_image_grid,
)

ROOT = Path(__file__).resolve().parents[2]
OSM_SKIP = {"footway", "path", "steps", "cycleway", "bridleway", "pedestrian"}
UNDO_MAX = 40
SOURCES = {
    "osm": {"file": "osm_roads_full.shp", "id": "osm_id", "name": "name", "cls": "fclass"},
    "tiger": {"file": "tiger_roads_clipped.shp", "id": "LINEARID", "name": "FULLNAME", "cls": "MTFCC"},
}


def _log(path: Path, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _length_px(coords: list[list[float]]) -> float:
    if len(coords) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i - 1][0]
        dy = coords[i][1] - coords[i - 1][1]
        s += (dx * dx + dy * dy) ** 0.5
    return s


def _insert_on_line(coords: list[list[float]], x: float, y: float) -> tuple[list[list[float]], list[float]]:
    if len(coords) < 2:
        return coords, [x, y]
    line = LineString([(p[0], p[1]) for p in coords])
    pt = Point(x, y)
    d = float(line.project(pt))
    snapped = line.interpolate(d)
    sx, sy = float(snapped.x), float(snapped.y)
    # Reuse an existing vertex if we landed on one.
    for p in coords:
        if (p[0] - sx) ** 2 + (p[1] - sy) ** 2 < 0.25:
            return [list(q) for q in coords], [p[0], p[1]]
    acc = 0.0
    out = [list(coords[0])]
    inserted = False
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        seg = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        if not inserted and acc - 1e-6 <= d <= acc + seg + 1e-6:
            out.append([sx, sy])
            inserted = True
        out.append(list(b))
        acc += seg
    if not inserted:
        out.append([sx, sy])
    return out, [sx, sy]


def _feat(fid: int, src: str, coords: list[list[float]], extra: dict[str, Any] | None = None) -> dict:
    rec = {
        "id": int(fid),
        "src": str(src)[:10],
        "src_id": "",
        "name": "",
        "fclass": "",
        "coords": [[float(x), float(y)] for x, y in coords],
    }
    if extra:
        rec["src_id"] = str(extra.get("src_id") or "")[:24]
        rec["name"] = str(extra.get("name") or "")[:80]
        rec["fclass"] = str(extra.get("fclass") or "")[:24]
    rec["length_px"] = round(_length_px(rec["coords"]), 1)
    return rec


@dataclass
class RoadEditSession:
    image_dir: Path
    roads_dir: Path
    out_dir: Path
    log_path: Path
    only_stems: list[str] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    stem: str = ""
    rgb: np.ndarray | None = None
    transform: Affine | None = None
    crs: object | None = None
    gsd: float = 0.3
    gen: int = 0
    dirty: bool = False
    block_slices: list[BlockSlice] = field(default_factory=list)
    overview_bytes: bytes | None = None
    overview_meta: dict | None = None
    overview_lock: threading.Lock = field(default_factory=threading.Lock)
    rgb_cache: LruBytes = field(default_factory=lambda: LruBytes(RGB_CACHE))
    rgb_pyr: dict[int, np.ndarray] = field(default_factory=dict)
    roads: list[dict] = field(default_factory=list)
    roads_orig: list[dict] = field(default_factory=list)
    undo: list[list[dict]] = field(default_factory=list)
    next_id: int = 1
    source: str = "osm"
    world_roads: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    open_gen: int = 0

    @property
    def size(self) -> tuple[int, int]:
        if self.rgb is None:
            return (0, 0)
        h, w = self.rgb.shape[:2]
        return w, h

    def _push_undo(self) -> None:
        self.undo.append(copy.deepcopy(self.roads))
        if len(self.undo) > UNDO_MAX:
            self.undo.pop(0)

    def _load_world_roads(self) -> None:
        if self.world_roads:
            return
        for key, spec in SOURCES.items():
            path = self.roads_dir / spec["file"]
            if not path.is_file() and key == "osm":
                path = self.roads_dir / "osm_roads_clipped.shp"
            if not path.is_file():
                continue
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                gdf = gdf.set_crs(4326)
            self.world_roads[key] = gdf

    def _clip_to_mosaic(self) -> list[dict]:
        if self.transform is None or self.rgb is None:
            return []
        w, h = self.size
        corners = [
            self.transform * (0, 0),
            self.transform * (w, 0),
            self.transform * (w, h),
            self.transform * (0, h),
        ]
        poly = box(
            min(p[0] for p in corners),
            min(p[1] for p in corners),
            max(p[0] for p in corners),
            max(p[1] for p in corners),
        )
        inv = ~self.transform
        keys = ["osm", "tiger"] if self.source == "both" else [self.source]
        feats: list[dict] = []
        fid = 1
        for key in keys:
            gdf = self.world_roads.get(key)
            if gdf is None or gdf.empty:
                continue
            spec = SOURCES[key]
            layer = gdf.to_crs(self.crs) if self.crs is not None else gdf
            if key == "osm" and spec["cls"] in layer.columns:
                skip = layer[spec["cls"]].astype(str).str.lower().isin(OSM_SKIP)
                layer = layer.loc[~skip]
            hit = layer[layer.intersects(poly)]
            if hit.empty:
                continue
            clipped = gpd.clip(hit, poly)
            for _, rec in clipped.iterrows():
                geom = rec.geometry
                if geom is None or geom.is_empty:
                    continue
                parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
                for part in parts:
                    if part is None or part.is_empty or part.geom_type != "LineString":
                        continue
                    coords = []
                    for x, y in part.coords:
                        c, r = inv * (x, y)
                        coords.append([float(c), float(r)])
                    if len(coords) < 2:
                        continue
                    extra = {
                        "src_id": rec[spec["id"]] if spec["id"] in rec.index else "",
                        "name": rec[spec["name"]] if spec["name"] in rec.index else "",
                        "fclass": rec[spec["cls"]] if spec["cls"] in rec.index else "",
                    }
                    feats.append(_feat(fid, key, coords, extra))
                    fid += 1
        self.next_id = fid
        return feats

    def _part_to_coords(self, part) -> list[list[float]] | None:
        inv = ~self.transform
        coords = []
        for x, y in part.coords:
            c, r = inv * (x, y)
            coords.append([float(c), float(r)])
        return coords if len(coords) >= 2 else None

    def _shp_to_feats(self, path: Path, start_id: int) -> list[dict]:
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            _log(self.log_path, f"could not read {path.name}: {exc}")
            return []
        if gdf.empty:
            return []
        if self.crs is not None and gdf.crs is not None:
            gdf = gdf.to_crs(self.crs)
        feats: list[dict] = []
        fid = start_id
        for _, rec in gdf.iterrows():
            geom = rec.geometry
            if geom is None or geom.is_empty:
                continue
            parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for part in parts:
                if part is None or part.is_empty or part.geom_type != "LineString":
                    continue
                coords = self._part_to_coords(part)
                if not coords:
                    continue
                extra = {
                    "src_id": rec["src_id"] if "src_id" in rec.index else "",
                    "name": rec["name"] if "name" in rec.index else "",
                    "fclass": rec["fclass"] if "fclass" in rec.index else "",
                }
                src = str(rec["src"]) if "src" in rec.index and rec["src"] else "edit"
                feats.append(_feat(fid, src, coords, extra))
                fid += 1
        return feats

    def _merge_saved(self, baseline: list[dict]) -> list[dict]:
        slices = self.block_slices
        if not slices:
            return baseline
        if len(slices) == 1:
            path = self.out_dir / f"{slices[0].stem}_edit.shp"
            if path.is_file():
                feats = self._shp_to_feats(path, 1)
                if feats:
                    self.next_id = max(f["id"] for f in feats) + 1
                    return feats
            return baseline
        combined = self.out_dir / f"{self.stem}_edit.shp"
        if combined.is_file():
            feats = self._shp_to_feats(combined, 1)
            if feats:
                self.next_id = max(f["id"] for f in feats) + 1
                return feats
        extra: list[dict] = []
        edited: list[BlockSlice] = []
        next_id = max((f["id"] for f in baseline), default=0) + 1
        for sl in slices:
            path = self.out_dir / f"{sl.stem}_edit.shp"
            if not path.is_file():
                continue
            feats = self._shp_to_feats(path, next_id)
            if not feats:
                continue
            next_id = max(f["id"] for f in feats) + 1
            extra.extend(feats)
            edited.append(sl)
        if not extra:
            return baseline

        def in_edited(feat: dict) -> bool:
            c = feat["coords"]
            mx = sum(p[0] for p in c) / len(c)
            my = sum(p[1] for p in c) / len(c)
            for sl in edited:
                if sl.x0 <= mx < sl.x0 + sl.width and sl.y0 <= my < sl.y0 + sl.height:
                    return True
            return False

        out: list[dict] = []
        fid = 1
        for f in [x for x in baseline if not in_edited(x)] + extra:
            rec = dict(f)
            rec["id"] = fid
            out.append(rec)
            fid += 1
        self.next_id = fid
        return out

    def discover(self) -> list[dict]:
        rows = []
        cells = list_image_grid(self.image_dir)
        by_rc, n_rows, n_cols = _grid_index(cells)
        for cell in cells:
            if self.only_stems is not None and cell.stem not in self.only_stems:
                continue
            edit = self.out_dir / f"{cell.stem}_edit.shp"
            rows.append(
                {
                    "stem": cell.stem,
                    "has_working": edit.is_file(),
                    "has_raw": True,
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
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir / "_map_overview.jpg", self.out_dir / "_map_overview.json"

    def _build_overview(self) -> tuple[bytes, dict]:
        jpg_path, json_path = self._overview_cache_paths()
        cells = list_image_grid(self.image_dir)
        by_rc, n_rows, n_cols = _grid_index(cells)
        stamp = {"n_tiles": len(cells), "cell": OVERVIEW_CELL, "n_rows": n_rows, "n_cols": n_cols, "kind": "roads"}
        if jpg_path.is_file() and json_path.is_file():
            try:
                cached = json.loads(json_path.read_text(encoding="utf-8"))
                if cached.get("stamp") == stamp and cached.get("width"):
                    return jpg_path.read_bytes(), cached
            except Exception:
                pass
        _log(self.log_path, f"building road-map overview {n_rows}x{n_cols}")
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
            has_edit = (self.out_dir / f"{cell.stem}_edit.shp").is_file()
            if has_edit:
                patch = patch.copy()
                patch[..., 2] = np.clip(patch[..., 2].astype(np.uint16) + 50, 0, 255).astype(np.uint8)
            canvas[y0 : y0 + cell_px, x0 : x0 + cell_px] = patch
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
                    "selectable": _is_selectable(by_rc, n_rows, n_cols, cell.ri, cell.ci),
                    "has_mask": has_edit,
                }
            )
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not ok:
            raise RuntimeError("overview jpeg failed")
        jpg = bytes(buf)
        meta = {"stamp": stamp, "width": ow, "height": oh, "cell": cell_px, "n_rows": n_rows, "n_cols": n_cols, "cells": out_cells}
        jpg_path.write_bytes(jpg)
        json_path.write_text(json.dumps(meta), encoding="utf-8")
        return jpg, meta

    def open_block_iter(self, center_stem: str, neighborhood: int = 9) -> Iterator[dict]:
        self._load_world_roads()
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
        specs: list[tuple] = []
        nw_transform = None
        crs = None
        gsd = 0.3
        for i, cell in enumerate(block):
            role = roles[i] if i < len(roles) else ""
            tif = self.image_dir / f"{cell.stem}.tif"
            with rasterio.open(tif) as ds:
                transform = ds.transform
                th, tw = int(ds.height), int(ds.width)
                tile_crs = ds.crs
                tile_gsd = float(max(abs(ds.transform.a), abs(ds.transform.e)))
            if nw_transform is None:
                nw_transform = transform
                crs = tile_crs
                gsd = tile_gsd
            col0, row0 = (~nw_transform) * (transform * (0.0, 0.0))
            x0 = max(0, int(round(col0)))
            y0 = max(0, int(round(row0)))
            specs.append((cell, role, tif, transform, x0, y0, th, tw))
        W = max(x0 + tw for _, _, _, _, x0, y0, th, tw in specs)
        H = max(y0 + th for _, _, _, _, x0, y0, th, tw in specs)

        with self.lock:
            self.open_gen += 1
            open_gen = self.open_gen
            old = self.rgb
            self.rgb = None
            self.roads = []
            self.undo.clear()
            self.rgb_cache.clear()
            self.rgb_pyr.clear()

        yield {
            "event": "alloc",
            "width": W,
            "height": H,
            "message": f"Preparing {W}×{H} mosaic…",
        }
        _log(
            self.log_path,
            f"open {center_stem} n={neighborhood} {W}x{H} reuse={old is not None and old.shape == (H, W, 3)} y0s={[s[5] for s in specs]}",
        )
        if old is not None and old.shape == (H, W, 3):
            mosaic_rgb = old
            mosaic_rgb.fill(0)
            old = None
        else:
            old = None
            gc.collect()
            mosaic_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        slices: list[BlockSlice] = []
        for i, (cell, role, tif, transform, x0, y0, th, tw) in enumerate(specs):
            if open_gen != self.open_gen:
                return
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
            with rasterio.open(tif) as ds:
                rgb = read_rgb(ds)
            h, w = rgb.shape[0], rgb.shape[1]
            rh, rw = min(h, H - y0), min(w, W - x0)
            mosaic_rgb[y0 : y0 + rh, x0 : x0 + rw] = rgb[:rh, :rw]
            slices.append(
                BlockSlice(stem=cell.stem, y0=y0, x0=x0, height=rh, width=rw, transform=transform, raw_path=None)
            )
            elapsed = time.perf_counter() - t0
            yield {
                "event": "tile",
                "i": i + 1,
                "total": total,
                "stem": cell.stem,
                "role": role,
                "phase": "done",
                "elapsed_s": round(elapsed, 2),
                "message": f"{i + 1}/{total}  {role}  {cell.stem}  done ({elapsed:.1f}s)",
            }
            del rgb
        if neighborhood == 1 and slices:
            sl = slices[0]
            mosaic_rgb = mosaic_rgb[sl.y0 : sl.y0 + sl.height, sl.x0 : sl.x0 + sl.width].copy()
            H, W = mosaic_rgb.shape[0], mosaic_rgb.shape[1]
            slices[0] = BlockSlice(stem=sl.stem, y0=0, x0=0, height=H, width=W, transform=sl.transform, raw_path=None)
            gc.collect()
        yield {"event": "mosaic", "width": W, "height": H, "message": f"Assembled mosaic {W}×{H}"}
        with self.lock:
            if open_gen != self.open_gen:
                return
            self.stem = center_stem
            self.rgb = mosaic_rgb
            self.transform = nw_transform
            self.crs = crs
            self.gsd = gsd
            self.block_slices = slices
            self.gen += 1
            self.dirty = False
            self.undo.clear()
            self.rgb_cache.clear()
            self.rgb_pyr.clear()
        yield {"event": "clip", "message": "Clipping road map…"}
        roads = self._merge_saved(self._clip_to_mosaic())
        orig = copy.deepcopy(roads)
        with self.lock:
            if open_gen != self.open_gen:
                return
            self.roads = roads
            self.roads_orig = orig
            self.next_id = max((f["id"] for f in roads), default=0) + 1
        _log(self.log_path, f"ready {center_stem} n_roads={len(roads)}")
        payload = self.roads_payload()
        yield {"event": "ready", "stats": {k: v for k, v in payload.items() if k != "roads"}, "roads": payload["roads"], "message": "ready"}

    def stats(self) -> dict:
        with self.lock:
            if self.rgb is None:
                return {"open": False, "stem": self.stem or None, "block": []}
            w, h = self.size
            n = len(self.roads)
            km = sum(f["length_px"] for f in self.roads) * self.gsd / 1000.0
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
                "undo": len(self.undo),
                "n_roads": n,
                "km": round(km, 2),
                "source": self.source,
            }

    def roads_payload(self) -> dict:
        with self.lock:
            return {"roads": self.roads, **self.stats()}

    def set_source(self, source: str) -> dict:
        if source not in {"osm", "tiger", "both"}:
            raise ValueError("source must be osm, tiger, or both")
        with self.lock:
            self.source = source
            if self.rgb is None:
                return self.stats()
            self._push_undo()
            self.roads = self._clip_to_mosaic()
            self.roads_orig = copy.deepcopy(self.roads)
            self.dirty = False
            self.gen += 1
        return self.roads_payload()

    def delete_id(self, fid: int) -> dict:
        with self.lock:
            self._push_undo()
            self.roads = [f for f in self.roads if f["id"] != fid]
            self.dirty = True
            self.gen += 1
        return self.roads_payload()

    def add_line(self, coords: list[list[float]], src: str = "edit") -> dict:
        if len(coords) < 2:
            return {"ok": False, "reason": "need 2 points"}
        with self.lock:
            self._push_undo()
            fid = self.next_id
            self.next_id += 1
            self.roads.append(_feat(fid, src, coords))
            self.dirty = True
            self.gen += 1
        return self.roads_payload()

    def stretch(self, fid: int, x: float, y: float, coords: list[list[float]]) -> dict:
        if len(coords) < 2:
            return {"ok": False, "reason": "need 2 points"}
        with self.lock:
            hit = next((f for f in self.roads if f["id"] == fid), None)
            if hit is None:
                return {"ok": False, "reason": "line not found"}
            self._push_undo()
            new_coords, at = _insert_on_line(hit["coords"], x, y)
            hit["coords"] = new_coords
            hit["length_px"] = round(_length_px(new_coords), 1)
            spur = [at] + [c for c in coords if (c[0] - at[0]) ** 2 + (c[1] - at[1]) ** 2 > 0.01]
            if len(spur) < 2:
                spur = [at, coords[-1]]
            nid = self.next_id
            self.next_id += 1
            self.roads.append(_feat(nid, "edit", spur, {"name": "spur"}))
            self.dirty = True
            self.gen += 1
        return self.roads_payload()

    def set_coords(self, fid: int, coords: list[list[float]]) -> dict:
        if len(coords) < 2:
            return {"ok": False, "reason": "need 2 points"}
        with self.lock:
            hit = next((f for f in self.roads if f["id"] == fid), None)
            if hit is None:
                return {"ok": False, "reason": "line not found"}
            self._push_undo()
            hit["coords"] = [[float(a), float(b)] for a, b in coords]
            hit["length_px"] = round(_length_px(hit["coords"]), 1)
            self.dirty = True
            self.gen += 1
        return self.roads_payload()

    def drop_short(self, min_m: float) -> dict:
        min_px = max(1.0, float(min_m) / max(self.gsd, 1e-6))
        with self.lock:
            self._push_undo()
            before = len(self.roads)
            self.roads = [f for f in self.roads if f["length_px"] >= min_px]
            dropped = before - len(self.roads)
            self.dirty = True
            self.gen += 1
        return {**self.roads_payload(), "dropped": dropped}

    def join_ends(self, max_m: float) -> dict:
        max_px = max(1.0, float(max_m) / max(self.gsd, 1e-6))
        with self.lock:
            self._push_undo()
            roads = self.roads
            n_join = 0
            used: set[int] = set()
            ends: list[tuple[int, int, float, float]] = []
            for i, f in enumerate(roads):
                c = f["coords"]
                if len(c) < 2:
                    continue
                ends.append((i, 0, c[0][0], c[0][1]))
                ends.append((i, -1, c[-1][0], c[-1][1]))
            for a in range(len(ends)):
                ia, sa, xa, ya = ends[a]
                if ia in used:
                    continue
                best = None
                best_d = max_px
                for b in range(a + 1, len(ends)):
                    ib, sb, xb, yb = ends[b]
                    if ib == ia or ib in used:
                        continue
                    d = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best = (ib, sb, xb, yb)
                if best is None:
                    continue
                ib, sb, xb, yb = best
                ca = list(roads[ia]["coords"])
                cb = list(roads[ib]["coords"])
                if sa == -1 and sb == 0:
                    merged = ca + cb[1:]
                elif sa == -1 and sb == -1:
                    merged = ca + list(reversed(cb))[1:]
                elif sa == 0 and sb == -1:
                    merged = cb + ca[1:]
                else:
                    merged = list(reversed(ca)) + cb[1:]
                roads[ia]["coords"] = merged
                roads[ia]["length_px"] = round(_length_px(merged), 1)
                used.add(ib)
                n_join += 1
            self.roads = [f for i, f in enumerate(roads) if i not in used]
            self.dirty = True
            self.gen += 1
        return {**self.roads_payload(), "joined": n_join}

    def undo_last(self) -> dict:
        with self.lock:
            if not self.undo:
                return self.roads_payload()
            self.roads = self.undo.pop()
            self.dirty = True
            self.gen += 1
        return self.roads_payload()

    def reset_orig(self) -> dict:
        with self.lock:
            self._push_undo()
            self.roads = copy.deepcopy(self.roads_orig)
            self.dirty = False
            self.gen += 1
        return self.roads_payload()

    def _px_to_world(self, c: float, r: float) -> tuple[float, float]:
        x, y = self.transform * (c, r)
        return float(x), float(y)

    def replace_roads(self, items: list[dict]) -> None:
        feats: list[dict] = []
        next_id = 1
        for raw in items:
            coords = raw.get("coords") or []
            if len(coords) < 2:
                continue
            fid = int(raw.get("id") or next_id)
            extra = {
                "src_id": raw.get("src_id") or "",
                "name": raw.get("name") or "",
                "fclass": raw.get("fclass") or "",
            }
            feats.append(_feat(fid, str(raw.get("src") or "edit"), coords, extra))
            next_id = max(next_id, fid + 1)
        with self.lock:
            self.roads = feats
            self.next_id = next_id
            self.dirty = True
            self.gen += 1

    def save(self, items: list[dict] | None = None) -> dict:
        from shapely.geometry import LineString as LS

        if items is not None:
            self.replace_roads(items)
        with self.lock:
            if self.rgb is None or self.transform is None:
                raise RuntimeError("nothing loaded")
            roads = copy.deepcopy(self.roads)
            slices = list(self.block_slices)
            crs = self.crs
            stem = self.stem
            transform = self.transform
            gsd = self.gsd
        self.out_dir.mkdir(parents=True, exist_ok=True)

        def px_to_world(c: float, r: float) -> tuple[float, float]:
            x, y = transform * (c, r)
            return float(x), float(y)

        rows = []
        for f in roads:
            world = [px_to_world(c, r) for c, r in f["coords"]]
            if len(world) < 2:
                continue
            rows.append(
                {
                    "src": f["src"],
                    "src_id": f["src_id"],
                    "name": f["name"],
                    "fclass": f["fclass"],
                    "length_m": round(f["length_px"] * gsd, 2),
                    "geometry": LS(world),
                }
            )
        gdf = gpd.GeoDataFrame(rows, crs=crs)
        combined = self.out_dir / f"{stem}_edit.shp"
        if gdf.empty:
            gpd.GeoDataFrame({"src": []}, geometry=[], crs=crs).to_file(combined, driver="ESRI Shapefile")
        else:
            gdf.to_file(combined, driver="ESRI Shapefile")
        saved = []
        for sl in slices:
            poly = box(*rasterio.transform.array_bounds(sl.height, sl.width, sl.transform))
            if gdf.empty:
                tile_g = gdf
            else:
                tile_g = gpd.clip(gdf, poly)
            path = self.out_dir / f"{sl.stem}_edit.shp"
            if tile_g.empty:
                gpd.GeoDataFrame({"src": []}, geometry=[], crs=crs).to_file(path, driver="ESRI Shapefile")
            else:
                tile_g.to_file(path, driver="ESRI Shapefile")
            saved.append({"stem": sl.stem, "n": int(len(tile_g)), "shp": str(path)})
            _log(self.log_path, f"saved {sl.stem} n={len(tile_g)}")
        with self.lock:
            self.dirty = False
        return {"shp": str(combined), "n_roads": int(len(gdf)), "tiles": saved}

    def _render_tile(self, arr: np.ndarray, width: int, height: int, level: int, x: int, y: int, pyr: dict) -> np.ndarray | None:
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


def build_app(session: RoadEditSession) -> FastAPI:
    app = FastAPI(title="Road map editor")
    html_path = Path(__file__).with_name("road_edit.html")

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

    @app.get("/api/roads")
    def roads() -> dict:
        return session.roads_payload()

    @app.post("/api/open-block")
    def open_block(body: dict) -> StreamingResponse:
        stem = str(body.get("stem") or "").strip()
        if not stem:
            raise HTTPException(400, "stem required")
        neighborhood = int(body.get("neighborhood") or 9)
        source = str(body.get("source") or session.source)
        if source in {"osm", "tiger", "both"}:
            session.source = source

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

    @app.post("/api/source")
    def source(body: dict) -> dict:
        return session.set_source(str(body.get("source") or "osm"))

    @app.post("/api/delete")
    def delete(body: dict) -> dict:
        return session.delete_id(int(body["id"]))

    @app.post("/api/add")
    def add(body: dict) -> dict:
        coords = [[float(p[0]), float(p[1])] for p in body.get("coords") or []]
        return session.add_line(coords, str(body.get("src") or "edit"))

    @app.post("/api/stretch")
    def stretch(body: dict) -> dict:
        coords = [[float(p[0]), float(p[1])] for p in body.get("coords") or []]
        return session.stretch(int(body["id"]), float(body["x"]), float(body["y"]), coords)

    @app.post("/api/set-coords")
    def set_coords(body: dict) -> dict:
        coords = [[float(p[0]), float(p[1])] for p in body.get("coords") or []]
        return session.set_coords(int(body["id"]), coords)

    @app.post("/api/drop-short")
    def drop_short(body: dict) -> dict:
        return session.drop_short(float(body.get("min_m") or 12))

    @app.post("/api/join-ends")
    def join_ends(body: dict) -> dict:
        return session.join_ends(float(body.get("max_m") or 8))

    @app.post("/api/undo")
    def undo() -> dict:
        return session.undo_last()

    @app.post("/api/reset")
    def reset() -> dict:
        return session.reset_orig()

    @app.post("/api/save")
    def save(body: dict) -> dict:
        return session.save(body.get("roads"))

    @app.get("/tiles/rgb/{level}/{x}/{y}.jpg")
    def rgb_tile(level: int, x: int, y: int) -> Response:
        if session.rgb is None:
            raise HTTPException(404, "no image")
        data = session.rgb_tile(level, x, y)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


def launch(*, map_name: str = "Fort_Riley", host: str = "127.0.0.1", port: int = 7862) -> None:
    import uvicorn

    image_dir = ROOT / "data" / map_name / "Imagery"
    roads_dir = ROOT / "data" / map_name / "Roads"
    out_dir = ROOT / "outputs" / "footprints" / map_name / "roads_edit"
    session = RoadEditSession(
        image_dir=image_dir,
        roads_dir=roads_dir,
        out_dir=out_dir,
        log_path=ROOT / "outputs" / "logs" / "road_edit.log",
    )

    def warmup() -> None:
        try:
            session._load_world_roads()
        except Exception as exc:
            print(f"road map preload: {exc}", flush=True)
        session.overview_payload()

    threading.Thread(target=warmup, daemon=True).start()
    app = build_app(session)
    print(f"Road editor  http://{host}:{port}  (map overview; click a tile)", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fort Riley road map editor (vectors on in-RAM ortho)")
    parser.add_argument("--map", default="Fort_Riley")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args(argv)
    launch(map_name=args.map, host=args.host, port=args.port)
    return 0
