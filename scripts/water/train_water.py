"""Train binary water SegFormer on LandCover.ai chips.

Writes only to outputs/checkpoints/water_landcoverai.
Does not touch oam_tcd_30cm, orchard_*, roof_types, building_footprints, or OEM water previews.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "shared" / "datasets" / "landcoverai_water",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "checkpoints" / "water_landcoverai",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "water.yaml")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from tree_seg.config import load_config
    from tree_seg.hf_auth import ensure_hf_auth
    from tree_seg.train import train_from_config

    ensure_hf_auth(verbose=True)
    cfg = load_config(args.config)
    print(f"train water -> {args.output}", flush=True)
    best = train_from_config(cfg, args.dataset, args.output)
    print(f"best checkpoint: {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
