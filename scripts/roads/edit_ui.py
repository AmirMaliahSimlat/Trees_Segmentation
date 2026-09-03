"""Launch the road map editor (ortho loaded once into RAM, vector overlay)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tree_seg.road_edit_ui import main


if __name__ == "__main__":
    raise SystemExit(main())
