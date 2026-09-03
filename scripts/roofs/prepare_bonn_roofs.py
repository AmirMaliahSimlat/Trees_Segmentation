#!/usr/bin/env python
"""Download Bonn roof-shape YOLO-seg data and crop one PNG chip per instance."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
FIGSHARE_ZIP = "https://ndownloader.figshare.com/files/53780282"

# Bonn README lists these eight types (figshare abstract says seven; keep all).
DEFAULT_NAMES = [
    "gabled",
    "flat",
    "skillion",
    "hipped",
    "gambrel",
    "half-hipped",
    "pyramidal",
    "mansard",
]


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 10_000_000:
        return dest
    print(f"downloading {url}", flush=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 tree-seg"})
    with urlopen(req, timeout=600) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def _find_yaml(root: Path) -> Path | None:
    cands = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    return cands[0] if cands else None


def _load_names(raw_root: Path) -> list[str]:
    yml = _find_yaml(raw_root)
    if yml is None:
        return DEFAULT_NAMES
    data = yaml.safe_load(yml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(n) for n in names]
    return DEFAULT_NAMES


def _split_dirs(raw_root: Path) -> dict[str, tuple[Path, Path]]:
    """Return split -> (images_dir, labels_dir)."""
    found: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val", "valid", "test"):
        img_a = raw_root / "images" / split
        lbl_a = raw_root / "labels" / split
        img_b = raw_root / split / "images"
        lbl_b = raw_root / split / "labels"
        if img_a.is_dir() and lbl_a.is_dir():
            found["val" if split == "valid" else split] = (img_a, lbl_a)
        elif img_b.is_dir() and lbl_b.is_dir():
            found["val" if split == "valid" else split] = (img_b, lbl_b)
    # nested extra folder
    if not found:
        for p in raw_root.rglob("images"):
            if p.is_dir() and (p / "train").is_dir():
                parent = p.parent
                return _split_dirs(parent)
    return found


def _parse_yolo_seg_line(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    cls = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    if len(coords) < 6 or len(coords) % 2:
        return None
    xs = np.array(coords[0::2]) * w
    ys = np.array(coords[1::2]) * h
    x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    return cls, x0, y0, x1, y1


def _crop(img: np.ndarray, x0, y0, x1, y1, pad_frac: float = 0.12) -> np.ndarray | None:
    h, w = img.shape[:2]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return img[y0:y1, x0:x1]


def prepare(raw_dir: Path, out_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    zip_path = raw_dir / "bonn_roof_shape.zip"
    if not zip_path.is_file() or zip_path.stat().st_size < 10_000_000:
        _download(FIGSHARE_ZIP, zip_path)
    extract_root = raw_dir / "raw"
    if not list(extract_root.rglob("*.jpg")) and not list(extract_root.rglob("*.png")):
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"extracting {zip_path}", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)

    names = _load_names(extract_root)
    splits = _split_dirs(extract_root)
    if not splits:
        # one more nesting
        kids = [p for p in extract_root.iterdir() if p.is_dir()]
        if len(kids) == 1:
            names = _load_names(kids[0])
            splits = _split_dirs(kids[0])
    if not splits:
        raise FileNotFoundError(f"Could not find YOLO images/labels under {extract_root}")

    print(f"classes={names}", flush=True)
    print(f"splits={ {k: str(v[0]) for k, v in splits.items()} }", flush=True)

    counts: dict[str, dict[str, int]] = {}
    for split, (img_dir, lbl_dir) in splits.items():
        out_split = "val" if split in {"val", "valid"} else split
        if out_split == "test":
            out_split = "val"
        counts.setdefault(out_split, {n: 0 for n in names})
        images = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        for img_path in tqdm(images, desc=f"crop {split}"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for i, line in enumerate(lbl_path.read_text(encoding="utf-8").splitlines()):
                parsed = _parse_yolo_seg_line(line, w, h)
                if parsed is None:
                    continue
                cls, x0, y0, x1, y1 = parsed
                if cls < 0 or cls >= len(names):
                    continue
                chip = _crop(img, x0, y0, x1, y1)
                if chip is None:
                    continue
                name = names[cls]
                dest_dir = out_dir / out_split / name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"{img_path.stem}_{i:03d}.jpg"
                cv2.imwrite(str(dest), chip)
                counts[out_split][name] = counts[out_split].get(name, 0) + 1

    meta = {"source": "Bonn Roof Geometry Dataset (figshare 28823390)", "classes": names, "counts": counts}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "shared" / "sources" / "bonn_roofs")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "data" / "shared" / "datasets" / "roof_types")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
