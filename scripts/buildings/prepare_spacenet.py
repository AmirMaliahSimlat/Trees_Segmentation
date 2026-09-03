"""Download SpaceNet 2 building chips and write train/val PNG pairs.

Uses the public AWS tarball (Vegas by default). Does not touch tree/orchard/roof
checkpoints. Does not overwrite OEM baseline_v1 previews.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "shared" / "sources" / "spacenet2"
DATASET = ROOT / "data" / "shared" / "datasets" / "spacenet2_buildings"

S3 = "https://spacenet-dataset.s3.amazonaws.com"
TARBALLS = {
    "sample": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_sample.tar.gz",
    "vegas": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz",
    "paris": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz",
    "shanghai": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_4_Shanghai.tar.gz",
    "khartoum": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_5_Khartoum.tar.gz",
}


def _download(key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3}/{key}"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"have {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
        return dest
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.is_file() else 0
    headers = {"User-Agent": "tree-seg-spacenet"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"  resume from {existing / 1e9:.2f} GB", flush=True)
    req = Request(url, headers=headers)
    with urlopen(req, timeout=600) as resp:
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
    return dest


def _to_uint8(arr: np.ndarray) -> np.ndarray:
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


def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    rgb_dirs = list(root.rglob("RGB-PanSharpen")) + list(root.rglob("RGB-PanSharpen-8bit"))
    if not rgb_dirs:
        rgb_dirs = [p for p in root.rglob("*") if p.is_dir() and "RGB" in p.name.upper()]
    for img_dir in rgb_dirs:
        label_dir = None
        for cand in (
            img_dir.parent / "geojson" / "buildings",
            img_dir.parent / "geojson",
            img_dir.parent / "labels",
        ):
            if cand.is_dir():
                label_dir = cand
                break
        if label_dir is None:
            continue
        for tif in sorted(img_dir.glob("*.tif")):
            stem = tif.stem.replace("RGB-PanSharpen_", "").replace("RGB-PanSharpen-", "")
            hits = (
                list(label_dir.glob(f"*{stem}*.geojson"))
                + list(label_dir.glob(f"*{stem}*.json"))
            )
            if not hits:
                continue
            pairs.append((tif, hits[0]))
    return pairs


def _write_chip(tif: Path, label: Path, img_out: Path, msk_out: Path) -> bool:
    with rasterio.open(tif) as ds:
        data = ds.read()
        if data.shape[0] >= 3:
            rgb = np.transpose(data[:3], (1, 2, 0))
        else:
            rgb = np.repeat(np.transpose(data[:1], (1, 2, 0)), 3, axis=2)
        rgb = _to_uint8(rgb)
        transform = ds.transform
        crs = ds.crs
        h, w = rgb.shape[:2]
    try:
        gdf = gpd.read_file(label)
    except Exception:
        return False
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    shapes = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom, 1))
    if shapes:
        mask = rasterize(
            shapes,
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
    if float(mask.mean()) < 1e-4:
        return False
    img_out.parent.mkdir(parents=True, exist_ok=True)
    msk_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(msk_out), (mask > 0).astype(np.uint8) * 255)
    return True


def _keep_tar_member(name: str) -> bool:
    n = name.replace("\\", "/").lower()
    return "rgb-pansharpen" in n or "/geojson/" in n or n.endswith(".geojson") or n.endswith(".json")


def _extract_needed(tarball: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"extract RGB+labels from {tarball.name}", flush=True)
    with tarfile.open(tarball, "r:gz") as tar:
        members = [m for m in tar.getmembers() if _keep_tar_member(m.name)]
        print(f"  {len(members)} members (RGB-PanSharpen + geojson only)", flush=True)
        tar.extractall(dest, members=members, filter="data")


def _chip_count(aoi: str) -> int:
    n = 0
    for split in ("train", "val"):
        folder = DATASET / split / "images"
        if folder.is_dir():
            n += len(list(folder.glob(f"{aoi}_*.png")))
    return n


def _prepare_aoi(aoi: str, val_fraction: float, seed: int, max_chips: int) -> dict:
    key = TARBALLS[aoi]
    tarball = SOURCE / Path(key).name
    _download(key, tarball)
    extract = SOURCE / aoi
    marker = extract / ".extracted"
    if not marker.exists():
        if extract.exists():
            print(f"remove incomplete extract {extract}", flush=True)
            shutil.rmtree(extract, ignore_errors=True)
        _extract_needed(tarball, extract)
        marker.write_text("ok", encoding="utf-8")

    chipped = extract / ".chipped"
    existing = _chip_count(aoi)
    if chipped.exists() or existing > 10:
        written = {"train": 0, "val": 0, "skipped_existing": existing}
        print(f"skip chip write for {aoi} ({existing} chips already on disk)", flush=True)
        return {
            "aoi": aoi,
            "tarball": tarball.name,
            "n_pairs_found": existing,
            "written": written,
        }

    pairs = _find_pairs(extract)
    if not pairs:
        raise FileNotFoundError(f"No RGB+geojson pairs under {extract}")
    random.Random(seed).shuffle(pairs)
    if max_chips > 0:
        pairs = pairs[:max_chips]
    n_val = max(1, int(round(len(pairs) * val_fraction))) if len(pairs) > 5 else max(1, len(pairs) // 10)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:] or pairs

    written = {"train": 0, "val": 0}
    for split, items in (("train", train_pairs), ("val", val_pairs)):
        for i, (tif, lab) in enumerate(tqdm(items, desc=f"{aoi}:{split}")):
            stem = f"{aoi}_{i:05d}"
            img_out = DATASET / split / "images" / f"{stem}.png"
            msk_out = DATASET / split / "masks" / f"{stem}.png"
            if img_out.is_file() and msk_out.is_file():
                written[split] += 1
                continue
            ok = _write_chip(tif, lab, img_out, msk_out)
            if ok:
                written[split] += 1

    chipped.write_text("ok", encoding="utf-8")
    print(f"remove extract {extract} to free disk", flush=True)
    shutil.rmtree(extract, ignore_errors=True)
    return {
        "aoi": aoi,
        "tarball": tarball.name,
        "n_pairs_found": len(pairs),
        "written": written,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", nargs="+", choices=sorted(TARBALLS), default=["vegas"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chips", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    cities = []
    for aoi in args.aoi:
        print(f"==== prepare {aoi} ====", flush=True)
        cities.append(_prepare_aoi(aoi, args.val_fraction, args.seed, args.max_chips))

    meta = {
        "source": "SpaceNet 2 buildings",
        "aois": [c["aoi"] for c in cities],
        "cities": cities,
        "chip_counts": {c["aoi"]: _chip_count(c["aoi"]) for c in cities},
    }
    DATASET.mkdir(parents=True, exist_ok=True)
    (DATASET / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
