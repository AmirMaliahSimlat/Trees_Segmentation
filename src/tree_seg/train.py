"""Fine-tune SegFormer on corrected canopy tiles."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from tree_seg.dataset import TreeMaskDataset, build_train_transforms, build_val_transforms
from tree_seg.metrics import binary_confusion, mean_metrics
from tree_seg.model import load_segformer, resolve_device


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on tree class (index 1) or sigmoid if single channel."""
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)[:, 0]
    else:
        probs = torch.softmax(logits, dim=1)[:, 1]
    targets_f = targets.float()
    dims = (1, 2)
    inter = (probs * targets_f).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets_f.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    # Upsample logits to label resolution
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    if logits.shape[1] == 1:
        bce = F.binary_cross_entropy_with_logits(logits[:, 0], targets.float())
    else:
        bce = F.cross_entropy(logits, targets.long())
    d = dice_loss(logits, targets)
    return bce_weight * bce + dice_weight * d


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    metrics = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pixel_values).logits
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            pred = (torch.sigmoid(logits)[:, 0] >= 0.5).long()
        else:
            pred = logits.argmax(dim=1)
        for i in range(pred.shape[0]):
            metrics.append(
                binary_confusion(
                    pred[i].cpu().numpy(),
                    labels[i].cpu().numpy(),
                )
            )
    return mean_metrics(metrics)


def train_segformer(
    train_images: str | Path,
    train_masks: str | Path,
    output_dir: str | Path,
    *,
    val_images: str | Path | None = None,
    val_masks: str | Path | None = None,
    model_name: str = "restor/tcd-segformer-mit-b5",
    image_size: int = 512,
    batch_size: int = 2,
    epochs: int = 30,
    learning_rate: float = 6e-5,
    weight_decay: float = 0.01,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    num_workers: int = 0,
    val_fraction: float = 0.2,
    seed: int = 42,
    early_stopping_patience: int = 8,
    device: str | None = None,
    num_labels: int = 2,
    id2label: dict[int, str] | None = None,
) -> Path:
    """Fine-tune and save best checkpoint directory. Returns path to best model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = resolve_device(device)

    train_ds_full = TreeMaskDataset(
        train_images, train_masks, transform=build_train_transforms(image_size)
    )

    if val_images and val_masks and Path(val_images).exists() and any(Path(val_images).glob("*.png")):
        val_ds = TreeMaskDataset(val_images, val_masks, transform=build_val_transforms(image_size))
        train_ds = train_ds_full
    else:
        # Split from train folder
        n = len(train_ds_full)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
        val_idx = indices[:n_val]
        train_idx = indices[n_val:] or indices
        # Val without heavy augmentations
        base_val = TreeMaskDataset(train_images, train_masks, transform=build_val_transforms(image_size))
        train_ds = Subset(train_ds_full, train_idx)
        val_ds = Subset(base_val, val_idx) if n_val else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_t.type == "cuda",
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device_t.type == "cuda",
        )

    bundle = load_segformer(
        model_name=model_name,
        num_labels=num_labels,
        id2label=id2label,
        device=str(device_t),
    )
    model: SegformerForSemanticSegmentation = bundle.model
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_iou = -1.0
    best_dir = output_dir / "best"
    patience_left = early_stopping_patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device_t)
            labels = batch["labels"].to(device_t)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixel_values).logits
            loss = segmentation_loss(
                logits, labels, bce_weight=bce_weight, dice_weight=dice_weight
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")

        scheduler.step()
        train_loss = float(np.mean(losses)) if losses else 0.0
        record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device_t)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            score = val_metrics.get("iou", 0.0)
            print(
                f"epoch {epoch}: loss={train_loss:.4f} "
                f"iou={val_metrics.get('iou', 0):.4f} f1={val_metrics.get('f1', 0):.4f} "
                f"fpr={val_metrics.get('fpr', 0):.4f}"
            )
            if score > best_iou:
                best_iou = score
                patience_left = early_stopping_patience
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                bundle.processor.save_pretrained(best_dir)
                (best_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stopping.")
                    history.append(record)
                    break
        else:
            # No val — save last
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            bundle.processor.save_pretrained(best_dir)
            print(f"epoch {epoch}: loss={train_loss:.4f}")

        history.append(record)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    last_dir = output_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)
    bundle.processor.save_pretrained(last_dir)
    return best_dir


def train_from_config(cfg: dict[str, Any], dataset_root: str | Path, output_dir: str | Path | None = None) -> Path:
    t = cfg.get("train", {})
    m = cfg.get("model", {})
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir or t.get("checkpoint_dir", "outputs/checkpoints"))
    id2label = m.get("id2label")
    if isinstance(id2label, dict):
        id2label = {int(k): str(v) for k, v in id2label.items()}
    else:
        id2label = None
    return train_segformer(
        train_images=dataset_root / "train" / "images",
        train_masks=dataset_root / "train" / "masks",
        val_images=dataset_root / "val" / "images",
        val_masks=dataset_root / "val" / "masks",
        output_dir=output_dir,
        model_name=m.get("name", "restor/tcd-segformer-mit-b5"),
        num_labels=int(m.get("num_labels", 2)),
        id2label=id2label,
        image_size=int(t.get("image_size", 512)),
        batch_size=int(t.get("batch_size", 2)),
        epochs=int(t.get("epochs", 30)),
        learning_rate=float(t.get("learning_rate", 6e-5)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        dice_weight=float(t.get("dice_weight", 1.0)),
        bce_weight=float(t.get("bce_weight", 1.0)),
        num_workers=int(t.get("num_workers", 0)),
        val_fraction=float(t.get("val_fraction", 0.2)),
        seed=int(t.get("seed", 42)),
        early_stopping_patience=int(t.get("early_stopping_patience", 8)),
    )
