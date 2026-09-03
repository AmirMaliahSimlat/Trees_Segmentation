"""Download LandCover.ai and write binary water PNG chips (water vs not-water).

Roads stay background so the model can unlearn OEM's road-as-water mistake.
Does not touch tree / orchard / roof / building / SpaceNet checkpoints.
Does not overwrite water_oem OEM previews.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import ssl
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "shared" / "sources" / "landcover_ai"
DATASET = ROOT / "data" / "shared" / "datasets" / "landcoverai_water"
URL = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
MD5 = "3268c89070e8734b4e91d531c0617e03"
WATER_CLASS = 3
ROAD_CLASS = 4
CHIP = 512


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _urlopen(req: Request):
    try:
        return urlopen(req, timeout=600)
    except Exception as exc:
        msg = str(exc).lower()
        if "certificate" not in msg and "ssl" not in msg:
            raise
        print("  SSL verify failed; retrying without verify (md5 will be checked)", flush=True)
        ctx = ssl._create_unverified_context()
        return urlopen(req, timeout=600, context=ctx)


def _download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        digest = _md5(dest)
        if digest == MD5:
            print(f"have {dest.name} ({dest.stat().st_size / 1e9:.2f} GB) md5 ok", flush=True)
            return dest
        print(f"md5 mismatch {digest}; re-downloading", flush=True)
        dest.unlink()
    print(f"downloading {URL}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.is_file() else 0
    headers = {"User-Agent": "tree-seg-landcoverai"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"  resume from {existing / 1e9:.2f} GB", flush=True)
    req = Request(URL, headers=headers)
    with _urlopen(req) as resp:
        status = getattr(resp, "status", 200)
        if existing > 0 and status == 200:
            print("  server sent full file; restarting download", flush=True)
            existing = 0
            mode = "wb"
        else:
            mode = "ab" if existing > 0 else "wb"
        remaining = int(resp.headers.get("Content-Length") or 0)
        total = existing + remaining if remaining else 0
        got = existing
        with tmp.open(mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print(flush=True)
    tmp.replace(dest)
    digest = _md5(dest)
    if digest != MD5:
        raise RuntimeError(f"LandCover.ai md5 {digest} != {MD5}")
    return dest


def _zip_image_mask_members(zpath: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    imgs = [
        n
        for n in names
        if ("/images/" in n.replace("\\", "/") or n.replace("\\", "/").startswith("images/"))
        and n.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
    ]
    masks = {
        Path(n).name: n
        for n in names
        if "/masks/" in n.replace("\\", "/") or n.replace("\\", "/").startswith("masks/")
    }
    if not imgs:
        tifs = [n for n in names if n.lower().endswith(".tif")]
        by_stem: dict[str, list[str]] = {}
        for n in tifs:
            by_stem.setdefault(Path(n).stem, []).append(n)
        pairs = []
        for _stem, hits in sorted(by_stem.items()):
            if len(hits) >= 2:
                img = next((h for h in hits if "mask" not in h.lower()), hits[0])
                msk = next((h for h in hits if h != img), hits[1])
                pairs.append((img, msk))
        return pairs
    pairs = []
    for img in sorted(imgs):
        msk = masks.get(Path(img).name) or masks.get(f"{Path(img).stem}.tif")
        if msk:
            pairs.append((img, msk))
    return pairs


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.transpose(arr[:3], (1, 2, 0))
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        arr = np.repeat(arr[0] if arr.ndim == 3 else arr[..., None], 3, axis=2)
    if arr.dtype == np.uint8:
        return arr
    x = arr.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.nanpercentile(x[finite], 1))
    hi = float(np.nanpercentile(x[finite], 99))
    if hi <= lo:
        hi = lo + 1.0
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    y[~finite] = 0
    return (y * 255.0).astype(np.uint8)


def _keep_chip(water: np.ndarray, labels: np.ndarray, rng: random.Random) -> bool:
    if float(water.mean()) > 0:
        return True
    if float((labels == ROAD_CLASS).mean()) > 0:
        return True
    return rng.random() < 0.15


def _write_chips_for_scene(
    img_p: Path,
    msk_p: Path,
    chip: int,
    split: str,
    rng: random.Random,
) -> int:
    n = 0
    with rasterio.open(img_p) as im, rasterio.open(msk_p) as mk:
        h, w = im.height, im.width
        for r in range(0, h - chip + 1, chip):
            for c in range(0, w - chip + 1, chip):
                win = Window(c, r, chip, chip)
                rgb = _to_uint8_rgb(im.read(window=win))
                lab = mk.read(1, window=win)
                if rgb.mean() < 2:
                    continue
                water = (lab == WATER_CLASS).astype(np.uint8)
                if not _keep_chip(water, lab, rng):
                    continue
                stem = f"{img_p.stem}_{r:05d}_{c:05d}"
                img_out = DATASET / split / "images" / f"{stem}.png"
                msk_out = DATASET / split / "masks" / f"{stem}.png"
                img_out.parent.mkdir(parents=True, exist_ok=True)
                msk_out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(msk_out), water * 255)
                n += 1
    print(f"  {img_p.name} -> {split}: {n} chips", flush=True)
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chip", type=int, default=CHIP)
    parser.add_argument("--keep-extract", action="store_true")
    args = parser.parse_args()

    zpath = SOURCE / "landcover.ai.v1.zip"
    _download(zpath)

    if (DATASET / "train" / "images").exists() and any((DATASET / "train" / "images").glob("*.png")):
        n_train = len(list((DATASET / "train" / "images").glob("*.png")))
        n_val = len(list((DATASET / "val" / "images").glob("*.png"))) if (DATASET / "val" / "images").exists() else 0
        print(f"have chips train={n_train} val={n_val}", flush=True)
        return 0

    extract = SOURCE / "v1_scratch"
    if extract.exists():
        shutil.rmtree(extract, ignore_errors=True)
    extract.mkdir(parents=True, exist_ok=True)
    members = _zip_image_mask_members(zpath)
    if not members:
        raise FileNotFoundError(f"No image/mask pairs in {zpath}")
    rng = random.Random(args.seed)
    rng.shuffle(members)
    n_val_scenes = max(1, int(round(len(members) * args.val_fraction))) if len(members) > 5 else 1
    val_members = members[:n_val_scenes]
    train_members = members[n_val_scenes:] or members
    print(f"found {len(members)} LandCover.ai scenes (train={len(train_members)} val={len(val_members)})", flush=True)

    written = {"train": 0, "val": 0}
    with zipfile.ZipFile(zpath) as zf:
        for split, items in (("train", train_members), ("val", val_members)):
            for img_member, msk_member in tqdm(items, desc=split):
                img_p = Path(zf.extract(img_member, extract))
                msk_p = Path(zf.extract(msk_member, extract))
                try:
                    written[split] += _write_chips_for_scene(img_p, msk_p, args.chip, split, rng)
                finally:
                    img_p.unlink(missing_ok=True)
                    msk_p.unlink(missing_ok=True)

    meta = {
        "source": "LandCover.ai v1",
        "url": URL,
        "water_class": WATER_CLASS,
        "n_scenes": len(members),
        "n_train_scenes": len(train_members),
        "n_val_scenes": len(val_members),
        "written": written,
        "note": "binary water; roads and other classes are background; scene-level val split",
    }
    DATASET.mkdir(parents=True, exist_ok=True)
    (DATASET / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    if not args.keep_extract:
        print(f"remove extract {extract} to free disk", flush=True)
        shutil.rmtree(extract, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
