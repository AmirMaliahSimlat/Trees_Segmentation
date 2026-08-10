"""Build a tiny synthetic train set and run a 1-epoch fine-tune smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from tree_seg.annotate_export import import_corrected_masks
from tree_seg.train import train_segformer

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "outputs" / "smoke" / "ann"


def main() -> None:
    ortho = ROOT / "outputs" / "smoke" / "synthetic_ortho.tif"
    gt = ROOT / "outputs" / "smoke" / "synthetic_gt_mask.tif"
    img_dir = ANN / "images"
    msk_dir = ANN / "masks"
    hold = ANN / "masks_holdout"
    hold.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    msk_dir.mkdir(exist_ok=True)

    with rasterio.open(ortho) as ds:
        rgb = np.transpose(ds.read([1, 2, 3]), (1, 2, 0))
    with rasterio.open(gt) as ds:
        mask = ds.read(1)

    crops = [(0, 0), (256, 0), (0, 256), (256, 256), (400, 100), (100, 400)]
    for i, (r, c) in enumerate(crops):
        tile = rgb[r : r + 256, c : c + 256]
        m = mask[r : r + 256, c : c + 256]
        if tile.shape[0] < 256 or tile.shape[1] < 256:
            continue
        name = f"synth_{i}.png"
        Image.fromarray(tile).save(img_dir / name)
        Image.fromarray((m * 255).astype(np.uint8)).save(msk_dir / name)

    for name in ["synth_4.png", "synth_5.png"]:
        src = msk_dir / name
        if src.exists():
            src.replace(hold / name)

    ds_root = ROOT / "outputs" / "smoke" / "dataset"
    counts = import_corrected_masks(ANN, ds_root)
    print("imported", counts)

    best = train_segformer(
        train_images=ds_root / "train" / "images",
        train_masks=ds_root / "train" / "masks",
        val_images=ds_root / "val" / "images",
        val_masks=ds_root / "val" / "masks",
        output_dir=ROOT / "outputs" / "smoke" / "checkpoints",
        model_name="restor/tcd-segformer-mit-b5",
        image_size=256,
        batch_size=1,
        epochs=1,
        learning_rate=6e-5,
        num_workers=0,
        early_stopping_patience=5,
        device="cpu",
    )
    print("BEST", best)


if __name__ == "__main__":
    main()
