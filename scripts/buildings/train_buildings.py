"""Train the building-footprint SegFormer on SpaceNet chips.

Writes only to outputs/checkpoints/building_footprints.
Does not touch oam_tcd_30cm, orchard_*, roof_types, or OEM baseline previews.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _wait_if_held(dataset: Path) -> None:
    hold = dataset / ".hold_train"
    if not hold.exists():
        return
    print("training held until Paris and Khartoum chips are ready", flush=True)
    while hold.exists():
        time.sleep(15)
    print("hold released; starting training", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "shared" / "datasets" / "spacenet2_buildings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "checkpoints" / "building_footprints",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "building_footprints.yaml")
    args = parser.parse_args()
    _wait_if_held(args.dataset)

    sys.path.insert(0, str(ROOT / "src"))
    from tree_seg.config import load_config
    from tree_seg.hf_auth import ensure_hf_auth
    from tree_seg.train import train_from_config

    ensure_hf_auth(verbose=True)
    cfg = load_config(args.config)
    print(f"train buildings -> {args.output}", flush=True)
    best = train_from_config(cfg, args.dataset, args.output)
    print(f"best checkpoint: {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
