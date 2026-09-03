"""After Vegas chips exist, download Paris + Khartoum and release the train hold.

Safe to run while the current Vegas prepare is still downloading.
Does not overwrite OEM baseline previews or other checkpoints.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
HERE = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "shared" / "datasets" / "spacenet2_buildings"
HOLD = DATASET / ".hold_train"


def vegas_ready() -> bool:
    meta = DATASET / "dataset_meta.json"
    if not meta.exists():
        return False
    imgs = DATASET / "train" / "images"
    if not imgs.is_dir():
        return False
    return any(imgs.glob("vegas_*.png"))


def main() -> int:
    print("waiting for Vegas chips before Paris/Khartoum...", flush=True)
    while not vegas_ready():
        time.sleep(30)
    print("Vegas chips are ready; starting Paris + Khartoum", flush=True)
    cmd = [str(PY), "-u", str(HERE / "prepare_spacenet.py"), "--aoi", "paris", "khartoum"]
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    if HOLD.exists():
        HOLD.unlink()
        print("released .hold_train", flush=True)
    print("Paris + Khartoum queued and prepared", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
