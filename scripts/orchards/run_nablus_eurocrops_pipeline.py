#!/usr/bin/env python
"""Nablus tiles → EuroCrops orchard model → shapefile.

Does not overwrite the NAIP+CDL Nablus_orchards outputs or tree-canopy files.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "logs" / "nablus_eurocrops_pipeline.log"


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

        from tree_seg.batch_predict import batch_predict_folder
        from tree_seg.config import load_config
        from tree_seg.model import bundle_from_config
        from tree_seg.postprocess import export_map_footprints

        tiles_dir = ROOT / "data" / "Nablus" / "Imagery"
        session_dir = ROOT / "data" / "Nablus" / "Review" / "orchards"
        masks_dir = ROOT / "outputs" / "pred" / "Nablus" / "orchards"
        footprints_dir = ROOT / "outputs" / "footprints" / "Nablus" / "orchards"
        checkpoint = ROOT / "outputs" / "checkpoints" / "orchard_eurocrops" / "best"

        tifs = sorted(tiles_dir.glob("*.tif"))
        if not tifs:
            raise FileNotFoundError(f"No tiles in {tiles_dir}")
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)

        log(f"=== predict EuroCrops orchard n_tiles={len(tifs)} checkpoint={checkpoint}")
        cfg = load_config(ROOT / "configs" / "orchard_eurocrops.yaml")
        cfg.setdefault("predict", {})["write_proba"] = False
        cfg.setdefault("predict", {})["match_train_gsd"] = False
        cfg["predict"]["native_gsd_m"] = 0.24
        cfg.setdefault("postprocess", {})["write_clean_mask"] = False
        cfg.setdefault("postprocess", {})["per_tile_folder"] = True

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

        log(f"=== export shapefile -> {footprints_dir}")
        footprints_dir.mkdir(parents=True, exist_ok=True)
        result = export_map_footprints(
            masks_dir,
            footprints_dir,
            cfg,
            map_name="Nablus_orchards_eurocrops",
        )
        log(f"tiles={result['n_tiles']} shapefile={result['map_shapefile']}")
        log("=== ALL DONE")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
