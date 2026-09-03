#!/usr/bin/env python
"""Fine-tune on OAM-TCD@30cm into an isolated checkpoint directory."""

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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/oam_tcd_30cm.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/shared/datasets/oam_tcd_30cm"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to config train.checkpoint_dir (oam_tcd_30cm)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = args.output or Path(cfg["train"]["checkpoint_dir"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Freeze marker so UI / later runs know not to overwrite casually
    (out / "BASELINE_OAM_TCD_30CM.txt").write_text(
        "Isolated baseline fine-tuned on OAM-TCD downsampled to ~30cm/px.\n"
        "Do not overwrite. Further domain fine-tunes should go to "
        "outputs/checkpoints/gfk_finetune (or another new folder).\n"
        f"Init weights: {cfg['model']['name']}\n",
        encoding="utf-8",
    )

    best = train_from_config(cfg, args.dataset, out)
    summary = {"best_checkpoint": str(best), "dataset": str(args.dataset), "config": str(args.config)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved isolated baseline: {best}")


if __name__ == "__main__":
    main()
