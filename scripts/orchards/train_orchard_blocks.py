#!/usr/bin/env python
"""Fine-tune SegFormer for orchard blocks into an isolated checkpoint directory."""

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
    parser.add_argument("--config", type=Path, default=Path("configs/orchard_blocks.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/shared/datasets/orchard_blocks"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to config train.checkpoint_dir (orchard_blocks)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output or cfg["train"]["checkpoint_dir"])
    out.mkdir(parents=True, exist_ok=True)

    (out / "ORCHARD_BLOCKS.txt").write_text(
        "Isolated orchard-block SegFormer (planted parcels / rows of trees or vines).\n"
        "Do not overwrite outputs/checkpoints/oam_tcd_30cm (tree canopy).\n"
        f"Init weights: {cfg['model']['name']}\n"
        "Labels: USDA CDL orchard/vineyard/citrus/nut classes on NAIP RGB.\n",
        encoding="utf-8",
    )

    best = train_from_config(cfg, args.dataset, out)
    summary = {
        "best_checkpoint": str(best),
        "dataset": str(args.dataset),
        "config": str(args.config),
        "task": "orchard_blocks",
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved orchard-block checkpoint: {best}")


if __name__ == "__main__":
    main()
