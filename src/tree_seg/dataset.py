"""Dataset and augmentations for fine-tuning."""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _pad(image_size: int) -> A.PadIfNeeded:
    # albumentations 1.x uses value/mask_value; 2.x uses fill/fill_mask
    try:
        return A.PadIfNeeded(
            image_size,
            image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        )
    except TypeError:
        return A.PadIfNeeded(
            image_size,
            image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        )


def build_train_transforms(image_size: int = 512) -> A.Compose:
    noise: A.BasicTransform
    try:
        noise = A.GaussNoise(std_range=(0.02, 0.08), p=0.2)
    except TypeError:
        noise = A.GaussNoise(var_limit=(5.0, 25.0), p=0.2)

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            _pad(image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            noise,
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
    )


def build_val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            _pad(image_size),
        ]
    )


class TreeMaskDataset(Dataset):
    """Paired RGB PNG + binary mask PNG dataset."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: A.Compose | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.ids = sorted(
            p.stem
            for p in self.images_dir.glob("*.png")
            if (self.masks_dir / f"{p.stem}.png").exists()
        )
        if not self.ids:
            raise FileNotFoundError(
                f"No image/mask pairs found in {self.images_dir} and {self.masks_dir}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        stem = self.ids[index]
        image = cv2.cvtColor(cv2.imread(str(self.images_dir / f"{stem}.png")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(stem)
        mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # ImageNet-style normalization expected by SegFormer processor;
        # we feed float CHW in [0,1] and let training code run processor OR
        # normalize here consistently with HF SegformerImageProcessor defaults.
        image_f = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_f = (image_f - mean) / std
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return {"pixel_values": image_t, "labels": mask_t, "id": stem}


def split_pairs(
    images_dir: Path,
    masks_dir: Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    ids = sorted(
        p.stem
        for p in images_dir.glob("*.png")
        if (masks_dir / f"{p.stem}.png").exists()
    )
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction))) if len(ids) > 1 else 0
    val_ids = ids[:n_val]
    train_ids = ids[n_val:] or ids
    return train_ids, val_ids
