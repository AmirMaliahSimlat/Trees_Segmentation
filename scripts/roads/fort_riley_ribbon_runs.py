"""Fort Riley ribbon I/O: original imagery; corridor-clean only when a line needs it."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEM = "FortRiley_r61440_c81920"
EDIT = ROOT / "outputs" / "footprints" / "Fort_Riley" / "roads_edit" / f"{STEM}_edit.shp"

RUNS = [
    {
        "name": "original",
        "img": ROOT / "data" / "Fort_Riley" / "Imagery" / f"{STEM}.tif",
        "out": ROOT / "outputs" / "footprints" / "Fort_Riley" / "ribbons without parking",
        "log_cover": ROOT / "outputs" / "logs" / "roads_ribbon_original.log",
        "log_union": ROOT / "outputs" / "logs" / "roads_ribbon_original_union.log",
    },
]
