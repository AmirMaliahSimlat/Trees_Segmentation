#!/usr/bin/env python
"""Mosaic GFK TMS PNGs → predict → one tree-footprint shapefile."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "gfk_pipeline.log"


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
        from tree_seg.tms import mosaic_tms_pngs_to_geotiff, warp_geotiff_to_utm

        gfk_root = ROOT / "data" / "GFK" / "Imagery"
        mosaic_ll = ROOT / "data" / "GFK" / "Imagery" / "GFK_z19_4326.tif"
        mosaic_utm = ROOT / "data" / "GFK" / "Imagery" / "GFK_z19_utm14n.tif"
        pred_dir = ROOT / "outputs" / "pred" / "GFK" / "trees"
        footprints_dir = ROOT / "outputs" / "footprints" / "GFK" / "trees"
        checkpoint = ROOT / "outputs" / "checkpoints" / "oam_tcd_30cm" / "best"

        if not (gfk_root / "tms.xml").is_file():
            raise FileNotFoundError(gfk_root / "tms.xml")
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)

        log("=== STEP 1/4 mosaic TMS PNGs (EPSG:4326)")
        if mosaic_ll.is_file() and mosaic_ll.stat().st_size > 1_000_000:
            log(f"reuse existing mosaic {mosaic_ll}")
        else:
            summary = mosaic_tms_pngs_to_geotiff(
                gfk_root,
                mosaic_ll,
                zoom=19,
                tms_xml=gfk_root / "tms.xml",
            )
            log(
                f"mosaic done: {summary['width']}x{summary['height']} "
                f"tiles={summary['tiles_used']} gsd~{summary['approx_gsd_m']:.3f}m"
            )

        log("=== STEP 2/4 warp to UTM 14N (metric CRS for areas)")
        if mosaic_utm.is_file() and mosaic_utm.stat().st_size > 1_000_000:
            log(f"reuse existing UTM mosaic {mosaic_utm}")
        else:
            warp_geotiff_to_utm(mosaic_ll, mosaic_utm, dst_crs="EPSG:32614")
            log(f"warp done: {mosaic_utm}")

        log(f"=== STEP 3/4 predict checkpoint={checkpoint}")
        cfg = load_config(ROOT / "configs" / "default.yaml")
        # Checkpoint was fine-tuned at ~30 cm; GFK mosaic is ~12 cm → downsample for inference
        cfg.setdefault("predict", {})["match_train_gsd"] = True
        cfg.setdefault("predict", {})["train_gsd_m"] = 0.30
        cfg.setdefault("postprocess", {})["write_clean_mask"] = False
        cfg.setdefault("postprocess", {})["per_tile_folder"] = False

        mask_path = pred_dir / "GFK_z19_utm14n_tree_mask.tif"
        force_predict = False
        if (not force_predict) and mask_path.is_file() and mask_path.stat().st_size > 1_000_000:
            log(f"reuse existing mask {mask_path}")
        else:
            for stale in pred_dir.glob("GFK_z19_utm14n_tree_*"):
                stale.unlink(missing_ok=True)
            bundle = bundle_from_config(cfg, checkpoint=str(checkpoint), device="cuda")
            paths = predict_from_config(mosaic_utm, pred_dir, cfg, bundle)
            mask_path = Path(paths["mask"])
            log(f"predict done: {mask_path}")

        log(f"=== STEP 4/4 export shapefile -> {footprints_dir}")
        footprints_dir.mkdir(parents=True, exist_ok=True)
        # Clear previous footprint products
        for stale in footprints_dir.glob("*"):
            if stale.is_file():
                stale.unlink(missing_ok=True)
        export_from_config(mask_path, footprints_dir, cfg)
        combined = footprints_dir / "GFK_tree_footprints.shp"
        merge_footprint_shapefiles(footprints_dir, combined)
        log(f"export done: {combined}")
        log("=== ALL DONE")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
