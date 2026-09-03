#!/usr/bin/env python
"""Fill one GeoTIFF: segment buildings/trees, inpaint with surrounding ground.

Does not use building shapefiles. Does not overwrite tree/orchard/roof checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ground_fill.yaml")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--reuse-mask", action="store_true")
    args = parser.parse_args()

    from tree_seg.config import load_config
    from tree_seg.ground_fill import fill_from_config
    from tree_seg.hf_auth import ensure_hf_auth

    ensure_hf_auth(verbose=True)
    cfg = load_config(args.config)
    result = fill_from_config(
        args.input, args.output, cfg, skip_existing=not args.no_skip, reuse_mask=args.reuse_mask
    )
    printable = {k: (str(v) if isinstance(v, Path) else v) for k, v in result.items()}
    print(json.dumps(printable, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
