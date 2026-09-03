#!/usr/bin/env python
"""Retile Fort Riley, full-res predict, export one map shapefile.

Designed to run as a detached process with logging (long-running).
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "fort_riley_pipeline.log"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from tree_seg.hf_auth import ensure_hf_auth

        ensure_hf_auth(verbose=False)

        from tree_seg.batch_predict import batch_predict_folder
        from tree_seg.config import load_config
        from tree_seg.model import bundle_from_config
        from tree_seg.postprocess import export_map_footprints
        from tree_seg.split_geotiff import split_geotiff

        src = ROOT / "data" / "Fort_Riley" / "Imagery" / "merged_rgb.tif"
        tiles_dir = ROOT / "data" / "Fort_Riley" / "Imagery"
        session_dir = ROOT / "data" / "Fort_Riley" / "Review" / "trees"
        masks_dir = session_dir / "geotiff_masks"
        footprints_dir = ROOT / "outputs" / "footprints" / "Fort_Riley" / "trees"
        checkpoint = ROOT / "outputs" / "checkpoints" / "oam_tcd_30cm" / "best"

        if not src.is_file():
            raise FileNotFoundError(src)
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)

        log(f"=== STEP 1/3 split {src} -> {tiles_dir}")
        summary = split_geotiff(
            src,
            tiles_dir,
            tile_size=10240,
            overlap=0,
            prefix="FortRiley",
            resume=True,
        )
        log(
            f"split done: written={summary['tiles_written']} "
            f"resumed={summary.get('tiles_resumed', 0)} "
            f"empty_skipped={summary['tiles_skipped_empty']}"
        )

        log(f"=== STEP 2/3 full-res predict checkpoint={checkpoint}")
        cfg = load_config(ROOT / "configs" / "default.yaml")
        bundle = bundle_from_config(cfg, checkpoint=str(checkpoint), device="cuda")
        session = batch_predict_folder(
            tiles_dir,
            session_dir,
            bundle,
            cfg,
            pred_dir=masks_dir,
            skip_existing=True,
            full_resolution=True,
            recursive=False,
        )
        log(f"predict done: items={len(session.items)} masks={masks_dir}")

        log(f"=== STEP 3/3 export map shapefile -> {footprints_dir}")
        result = export_map_footprints(
            masks_dir,
            footprints_dir,
            cfg,
            map_name="Fort_Riley",
        )
        log(f"export done: tiles={result['n_tiles']} shapefile={result['map_shapefile']}")
        log("=== ALL DONE")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
