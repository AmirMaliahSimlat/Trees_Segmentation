#!/usr/bin/env python
"""Fine-tune SegFormer on EuroCrops orchard blocks (isolated checkpoint)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tree_seg.hf_auth import ensure_hf_auth

ensure_hf_auth(verbose=True)

from tree_seg.config import load_config
from tree_seg.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/orchard_eurocrops.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/shared/datasets/orchard_eurocrops"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output or cfg["train"]["checkpoint_dir"])
    out.mkdir(parents=True, exist_ok=True)

    (out / "ORCHARD_EUROCROPS.txt").write_text(
        "Isolated orchard-block SegFormer trained on EuroCrops field polygons.\n"
        "Do not overwrite outputs/checkpoints/oam_tcd_30cm (tree canopy).\n"
        "Do not overwrite outputs/checkpoints/orchard_blocks (NAIP+CDL run).\n"
        f"Init weights: {cfg['model']['name']}\n"
        "Labels: EuroCrops ES_NA olives/fruit/vines rasterized on IGN PNOA ~25 cm RGB.\n",
        encoding="utf-8",
    )

    best = train_from_config(cfg, args.dataset, out)
    summary = {
        "best_checkpoint": str(best),
        "dataset": str(args.dataset),
        "config": str(args.config),
        "task": "orchard_eurocrops",
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved EuroCrops orchard checkpoint: {best}")


if __name__ == "__main__":
    main()
