#!/usr/bin/env python
"""Convert a GeoTIFF or a folder of GeoTIFFs to OSGeo TMS (zoom folders + TMS.xml).

Does not overwrite tree / orchard / roof checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _inputs(path: Path) -> list[Path]:
    if path.is_dir():
        tifs = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
        if not tifs:
            raise FileNotFoundError(f"No GeoTIFFs in {path}")
        return tifs
    if not path.is_file():
        raise FileNotFoundError(path)
    return [path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True, help="GeoTIFF or folder of GeoTIFFs")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--min-zoom", type=int, default=None)
    parser.add_argument("--max-zoom", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from tree_seg.tms import export_geotiffs_to_tms

    paths = _inputs(args.input)
    result = export_geotiffs_to_tms(
        paths,
        args.output,
        title=args.title or (args.input.name if args.input.is_dir() else args.input.stem),
        min_zoom=args.min_zoom,
        max_zoom=args.max_zoom,
        skip_existing=not args.no_skip,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
