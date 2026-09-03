#!/usr/bin/env python
"""Nablus + Fort Riley: OpenEarthMap object mask + LaMa fill.

Does not use building shapefiles.
Does not overwrite outputs/checkpoints/oam_tcd_30cm, orchard_*, or roof_types.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "ground_fill_maps.log"

MAPS = (
    ("Nablus", ROOT / "data" / "Nablus" / "Imagery", ROOT / "outputs" / "ground_fill" / "Nablus"),
    ("Fort_Riley", ROOT / "data" / "Fort_Riley" / "Imagery", ROOT / "outputs" / "ground_fill" / "Fort_Riley"),
)


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def export_map_tms(map_name: str, out_dir: Path, cfg: dict, *, skip_existing: bool) -> None:
    from tree_seg.tms import export_geotiffs_to_tms

    tms_cfg = cfg.get("tms") or {}
    if not bool(tms_cfg.get("enabled", True)):
        log(f"=== {map_name} TMS disabled")
        return
    bare_dir = out_dir / "bare"
    tifs = sorted(bare_dir.glob("*_bare.tif"))
    if not tifs:
        log(f"WARNING {map_name} no bare GeoTIFFs in {bare_dir}")
        return
    tms_dir = out_dir / "tms"
    log(f"=== {map_name} TMS n_tiles={len(tifs)} -> {tms_dir}")
    result = export_geotiffs_to_tms(
        tifs,
        tms_dir,
        title=map_name,
        tile_size=int(tms_cfg.get("tile_size", 256)),
        min_zoom=tms_cfg.get("min_zoom", 0),
        max_zoom=tms_cfg.get("max_zoom"),
        skip_existing=skip_existing,
        block_tiles=int(tms_cfg.get("block_tiles", 16)),
        workers=int(tms_cfg.get("workers", 4)),
    )
    log(json.dumps({k: v for k, v in result.items() if k != "sources"}, default=str))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-skip", action="store_true", help="Recompute even if bare TIFFs / TMS tiles already exist")
    parser.add_argument("--tms-only", action="store_true", help="Skip fill; build full-map TMS from existing bare TIFFs")
    parser.add_argument("--skip-tms", action="store_true", help="Skip TMS export after fill")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    try:
        from tree_seg.config import load_config
        from tree_seg.ground_fill import fill_geotiff, load_landcover, make_filler
        from tree_seg.hf_auth import ensure_hf_auth

        cfg = load_config(ROOT / "configs" / "ground_fill.yaml")
        m = cfg.get("model", {})
        inp = cfg.get("inpaint", {})

        processor = model = id2label = object_ids = device = filler = None
        if not args.tms_only:
            ensure_hf_auth(verbose=True)
            log("=== load land-cover + inpaint backend")
            processor, model, id2label, object_ids, device = load_landcover(m.get("hf_id"))
            filler = make_filler(inp, device)
            log(
                f"device={device} backend={filler.backend} object_ids={sorted(object_ids)} "
                f"labels={id2label} skip_existing={not args.no_skip}"
            )

        for map_name, src_dir, out_dir in MAPS:
            if not args.tms_only:
                tifs = sorted(src_dir.glob("*.tif")) + sorted(src_dir.glob("*.tiff"))
                log(f"=== {map_name} n_tiles={len(tifs)} -> {out_dir}")
                if not tifs:
                    log(f"WARNING no tiles in {src_dir}")
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, tif in enumerate(tifs, start=1):
                    log(f"[{map_name} {i}/{len(tifs)}] {tif.name}")
                    try:
                        result = fill_geotiff(
                            tif,
                            out_dir,
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
                            relight=bool(inp.get("relight", False)),
                            skip_existing=not args.no_skip,
                            per_object=bool(inp.get("per_object", True)),
                            object_pad_px=int(inp.get("object_pad_px", 128)),
                            split_erode_px=int(inp.get("split_erode_px", 12)),
                        )
                        slim = {
                            k: (str(v) if isinstance(v, Path) else v)
                            for k, v in result.items()
                            if k != "id2label"
                        }
                        log(json.dumps(slim, default=str))
                    except Exception:
                        log(f"FAILED {tif.name}:\n{traceback.format_exc()}")
                        continue
            if not args.skip_tms:
                try:
                    export_map_tms(map_name, out_dir, cfg, skip_existing=not args.no_skip)
                except Exception:
                    log(f"FAILED TMS {map_name}:\n{traceback.format_exc()}")
                    return 1
        log("=== ALL DONE")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
