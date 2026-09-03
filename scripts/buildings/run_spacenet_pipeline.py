"""Download SpaceNet Vegas/Paris/Khartoum, train building SegFormer, compare to OEM.

Does not overwrite buildings_oem/_preview_*.jpg or tree/orchard/roof checkpoints.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    cmd = [str(PY), "-u", str(HERE / script), *args]
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    run("prepare_spacenet.py", "--aoi", "vegas", "paris", "khartoum")
    run("train_buildings.py")
    run("run_spacenet_compare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
