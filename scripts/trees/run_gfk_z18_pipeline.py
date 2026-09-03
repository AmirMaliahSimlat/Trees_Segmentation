#!/usr/bin/env python
"""Process GFK TMS zoom-18 (~30cm) in blocks → one combined tree shapefile.

The full z18 extent is too large for a single mosaic, so we:
  1) mosaic PNG blocks (~10k px)
  2) warp each block to UTM 14N
  3) predict (no probability GeoTIFF)
  4) export per-block footprints
  5) merge into outputs/footprints/GFK/trees/GFK_tree_footprints.shp
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "gfk_z18_pipeline.log"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from tree_seg.hf_auth import ensure_hf_auth

        ensure_hf_auth(verbose=False)

        from tree_seg.config import load_config
        from tree_seg.model import bundle_from_config
        from tree_seg.postprocess import export_from_config, merge_footprint_shapefiles
        from tree_seg.predict import predict_from_config
        from tree_seg.tms import (
            iter_tms_blocks,
            list_tms_png_tiles,
            mosaic_tms_block_to_geotiff,
            warp_geotiff_to_utm,
        )

        gfk_root = ROOT / "data" / "GFK" / "Imagery"
        zoom = 18
        block_tiles = 40  # 40*256 = 10240 px
        checkpoint = ROOT / "outputs" / "checkpoints" / "oam_tcd_30cm" / "best"
        tmp_dir = ROOT / "outputs" / "tmp" / "gfk_z18"
        pred_dir = ROOT / "outputs" / "pred" / "GFK" / "z18"
        footprints_dir = ROOT / "outputs" / "footprints" / "GFK" / "trees"
        blocks_dir = footprints_dir / "blocks"
        state_path = ROOT / "outputs" / "logs" / "gfk_z18_state.json"
        combined = footprints_dir / "GFK_tree_footprints.shp"

        if not (gfk_root / "tms.xml").is_file():
            raise FileNotFoundError(gfk_root / "tms.xml")
        if not (gfk_root / str(zoom)).is_dir():
            raise FileNotFoundError(gfk_root / str(zoom))
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)

        tmp_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        blocks_dir.mkdir(parents=True, exist_ok=True)

        log(f"=== list TMS z{zoom} tiles")
        tiles = list_tms_png_tiles(gfk_root, zoom)
        blocks = iter_tms_blocks(tiles, block_tiles=block_tiles)
        log(f"tiles={len(tiles)} non-empty blocks={len(blocks)} block_tiles={block_tiles}")

        cfg = load_config(ROOT / "configs" / "default.yaml")
        # z18 ~20-30cm — matches 30cm fine-tune; no GSD resample
        cfg.setdefault("predict", {})["match_train_gsd"] = False
        cfg.setdefault("predict", {})["write_proba"] = False
        cfg.setdefault("postprocess", {})["write_clean_mask"] = False
        cfg.setdefault("postprocess", {})["per_tile_folder"] = True

        state = {"done": [], "failed": []}
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.setdefault("done", [])
            state.setdefault("failed", [])

        done = set(state["done"])
        log(f"=== load model {checkpoint} (resume done={len(done)})")
        bundle = bundle_from_config(cfg, checkpoint=str(checkpoint), device="cuda")

        for i, block in enumerate(blocks, start=1):
            name = block["name"]
            shp = blocks_dir / name / f"{name}_utm_tree_footprints.shp"
            if name in done and shp.is_file():
                continue

            log(
                f"[{i}/{len(blocks)}] block {name} "
                f"tiles={block['n_tiles']} x={block['x0']}-{block['x1']} y={block['y0']}-{block['y1']}"
            )
            rgb_ll = tmp_dir / f"{name}_4326.tif"
            rgb_utm = tmp_dir / f"{name}_utm.tif"
            block_pred = pred_dir / name
            block_pred.mkdir(parents=True, exist_ok=True)

            try:
                mosaic_tms_block_to_geotiff(
                    gfk_root,
                    rgb_ll,
                    zoom=zoom,
                    x0=block["x0"],
                    y0=block["y0"],
                    x1=block["x1"],
                    y1=block["y1"],
                    tms_xml=gfk_root / "tms.xml",
                )
                # Skip nearly empty mosaics (all black)
                import rasterio
                import numpy as np

                with rasterio.open(rgb_ll) as ds:
                    sample = ds.read(
                        out_shape=(3, max(1, ds.height // 32), max(1, ds.width // 32))
                    )
                if float((sample > 0).mean()) < 0.001:
                    log(f"  skip empty block {name}")
                    done.add(name)
                    state["done"] = sorted(done)
                    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                    rgb_ll.unlink(missing_ok=True)
                    continue

                warp_geotiff_to_utm(rgb_ll, rgb_utm, dst_crs="EPSG:32614")
                rgb_ll.unlink(missing_ok=True)

                paths = predict_from_config(rgb_utm, block_pred, cfg, bundle)
                mask_path = Path(paths["mask"])
                # Rename stem for clearer shapefile ids
                export_from_config(mask_path, blocks_dir, cfg)
                # export writes <stem>_tree_footprints under per-tile folder named after mask stem
                # Ensure stable folder name
                stem = mask_path.stem.replace("_tree_mask", "")
                produced = blocks_dir / stem / f"{stem}_tree_footprints.shp"
                target_dir = blocks_dir / name
                if produced.is_file():
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                        src = produced.with_suffix(ext)
                        if src.is_file():
                            dst = target_dir / f"{name}_utm_tree_footprints{ext}"
                            if dst.exists():
                                dst.unlink()
                            src.replace(dst)
                    # cleanup export folder named after mask stem if different
                    if stem != name:
                        shutil.rmtree(blocks_dir / stem, ignore_errors=True)

                rgb_utm.unlink(missing_ok=True)
                # drop mask to save disk (shapefile is the deliverable)
                for p in block_pred.glob("*"):
                    p.unlink(missing_ok=True)

                done.add(name)
                state["done"] = sorted(done)
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            except Exception:
                log(f"  FAILED block {name}:\n{traceback.format_exc()}")
                state["failed"] = sorted(set(state.get("failed", [])) | {name})
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                for p in (rgb_ll, rgb_utm):
                    p.unlink(missing_ok=True)
                continue

        log(f"=== merge shapefiles -> {combined}")
        # Clear previous combined deliverable only
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            p = combined.with_suffix(ext)
            p.unlink(missing_ok=True)
        merge_footprint_shapefiles(blocks_dir, combined)
        log(f"export done: {combined}")
        log("=== ALL DONE")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
