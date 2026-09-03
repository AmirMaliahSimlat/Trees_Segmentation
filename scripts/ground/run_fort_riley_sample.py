#!/usr/bin/env python
"""Fort Riley sample (~10% of tiles): densest building areas, LaMa fill.

Does not run Nablus. Does not overwrite tree / orchard / roof checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "ground_fill_fr_sample.log"
IMG_DIR = ROOT / "data" / "Fort_Riley" / "Imagery"
OUT_DIR = ROOT / "outputs" / "ground_fill" / "Fort_Riley"
BLDG = ROOT / "data" / "Fort_Riley" / "Buildings" / "1.shp"
LIST_PATH = OUT_DIR / "sample_tiles.txt"
SAMPLE_FRAC = 0.10


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pick_sample(tifs: list[Path], k: int) -> list[Path]:
    bldg = gpd.read_file(BLDG)
    with rasterio.open(tifs[0]) as ds:
        bldg = bldg.to_crs(ds.crs)
    sindex = bldg.sindex
    scored: list[tuple[float, Path]] = []
    for tif in tifs:
        with rasterio.open(tif) as ds:
            geom = box(*ds.bounds)
        idx = list(sindex.intersection(geom.bounds))
        if not idx:
            area = 0.0
        else:
            hit = bldg.iloc[idx]
            hit = hit[hit.intersects(geom)]
            area = float(hit.geometry.intersection(geom).area.sum()) if len(hit) else 0.0
        scored.append((area, tif))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for area, p in scored[:k] if area > 0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None, help="Number of densest-building tiles")
    parser.add_argument("--tiles", nargs="+", default=None, help="Specific tile filenames")
    parser.add_argument("--reuse-mask", action="store_true", help="Keep existing *_objects.tif holes")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    try:
        from tree_seg.config import load_config
        from tree_seg.ground_fill import fill_geotiff, load_landcover, make_filler
        from tree_seg.hf_auth import ensure_hf_auth

        ensure_hf_auth(verbose=True)
        cfg = load_config(ROOT / "configs" / "ground_fill.yaml")
        m = cfg.get("model", {})
        inp = cfg.get("inpaint", {})

        tifs = sorted(IMG_DIR.glob("*.tif")) + sorted(IMG_DIR.glob("*.tiff"))
        if not tifs:
            raise FileNotFoundError(IMG_DIR)
        by_name = {p.name: p for p in tifs}
        if args.tiles:
            sample = []
            for name in args.tiles:
                if name not in by_name:
                    raise FileNotFoundError(name)
                sample.append(by_name[name])
        else:
            k = args.count if args.count is not None else max(1, int(round(SAMPLE_FRAC * len(tifs))))
            sample = pick_sample(tifs, k)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        LIST_PATH.write_text("\n".join(p.name for p in sample) + "\n", encoding="utf-8")
        log(f"=== FR sample k={len(sample)}/{len(tifs)} backend={inp.get('backend', 'lama')}")
        for p in sample:
            log(f"  {p.name}")

        processor, model, id2label, object_ids, device = load_landcover(m.get("hf_id"))
        filler = make_filler(inp, device)
        log(f"device={device} object_ids={sorted(object_ids)} fill={filler.backend}")

        for i, tif in enumerate(sample, start=1):
            log(f"[sample {i}/{len(sample)}] {tif.name}")
            try:
                result = fill_geotiff(
                    tif,
                    OUT_DIR,
                    processor=processor,
                    model=model,
                    object_ids=object_ids,
                    device=device,
                    filler=filler,
                    tile_size=int(m.get("tile_size", 512)),
                    overlap=float(m.get("overlap", 0.125)),
                    dilate_px=int(inp.get("dilate_px", 12)),
                    inpaint_tile=int(inp.get("tile_size", 1024)),
                    shadow_grow_px=int(inp.get("shadow_grow_px", 96)),
                    shadow_luma_ratio=float(inp.get("shadow_luma_ratio", 0.55)),
                    lift_context=bool(inp.get("lift_context", True)),
                    relight=False,
                    reuse_mask=bool(args.reuse_mask),
                    skip_existing=False,
                    per_object=bool(inp.get("per_object", True)),
                    object_pad_px=int(inp.get("object_pad_px", 128)),
                    split_erode_px=int(inp.get("split_erode_px", 12)),
                )
                slim = {
                    key: (str(v) if isinstance(v, Path) else v)
                    for key, v in result.items()
                    if key != "id2label"
                }
                log(json.dumps(slim, default=str))
            except Exception:
                log(f"FAILED {tif.name}:\n{traceback.format_exc()}")
                continue
        log("=== SAMPLE DONE")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
