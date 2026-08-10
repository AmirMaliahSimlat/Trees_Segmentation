"""Download OAM-TCD, downsample 10cm→~30cm, write train/val PNG pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from tree_seg.hf_auth import ensure_hf_auth

ensure_hf_auth(verbose=True)

from datasets import load_dataset


def _to_rgb(image) -> np.ndarray:
    arr = np.array(image.convert("RGB") if hasattr(image, "convert") else image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr.astype(np.uint8)


def _to_binary_mask(annotation) -> np.ndarray:
    """Any non-background annotation pixel becomes tree (1)."""
    arr = np.array(annotation)
    if arr.ndim == 3:
        # RGB-encoded panoptic / class map — non-black = tree
        mask = (arr.max(axis=2) > 0).astype(np.uint8)
    else:
        mask = (arr > 0).astype(np.uint8)
    return mask


def downsample_pair(rgb: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    rgb_ds = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_ds = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return rgb_ds, mask_ds


def prepare_oam_tcd_30cm(
    output_dir: str | Path,
    *,
    source_gsd_m: float = 0.10,
    target_gsd_m: float = 0.30,
    val_fold: int = 0,
    max_train: int | None = None,
    max_val: int | None = None,
    seed: int = 42,
    dataset_name: str = "restor/tcd",
) -> dict:
    """
    Write ``output_dir/train|val/{images,masks}/*.png`` at ~target GSD.
    """
    output_dir = Path(output_dir)
    scale = source_gsd_m / target_gsd_m  # 10cm -> 30cm => 1/3

    print(f"Loading {dataset_name} (cached after first download)...")
    ds = load_dataset(dataset_name, split="train")

    train_idx = [i for i, v in enumerate(ds["validation_fold"]) if int(v) != val_fold and int(v) >= 0]
    val_idx = [i for i, v in enumerate(ds["validation_fold"]) if int(v) == val_fold]

    rng = random.Random(seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    if max_train is not None:
        train_idx = train_idx[:max_train]
    if max_val is not None:
        val_idx = val_idx[:max_val]

    meta = {
        "dataset": dataset_name,
        "source_gsd_m": source_gsd_m,
        "target_gsd_m": target_gsd_m,
        "scale": scale,
        "val_fold": val_fold,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }

    for split, indices in (("train", train_idx), ("val", val_idx)):
        img_dir = output_dir / split / "images"
        msk_dir = output_dir / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(indices, desc=f"Write {split}", unit="img"):
            row = ds[i]
            tile_id = f"oam_{int(row['image_id']):05d}"
            out_img = img_dir / f"{tile_id}.png"
            out_msk = msk_dir / f"{tile_id}.png"
            if out_img.exists() and out_msk.exists():
                continue

            rgb = _to_rgb(row["image"])
            mask = _to_binary_mask(row["annotation"])
            if mask.shape[:2] != rgb.shape[:2]:
                mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            rgb_ds, mask_ds = downsample_pair(rgb, mask, scale)
            Image.fromarray(rgb_ds).save(out_img)
            Image.fromarray((mask_ds * 255).astype(np.uint8)).save(out_msk)

    (output_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OAM-TCD downsampled to ~30cm/px")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/oam_tcd_30cm"),
        help="Output dataset root",
    )
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on train tiles")
    parser.add_argument("--max-val", type=int, default=None, help="Optional cap on val tiles")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_oam_tcd_30cm(
        args.output,
        val_fold=args.val_fold,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
