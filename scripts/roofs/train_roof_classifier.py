#!/usr/bin/env python
"""Fine-tune ImageNet ResNet-50 on Bonn roof-type chips.

Does not touch tree canopy or orchard SegFormer checkpoints.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def _tf(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    per_cls: dict[int, list[int]] = {}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
        for t, p in zip(y.tolist(), pred.tolist()):
            per_cls.setdefault(t, [0, 0])
            per_cls[t][1] += 1
            per_cls[t][0] += int(t == p)
    acc = correct / max(total, 1)
    cls_acc = {str(k): v[0] / max(v[1], 1) for k, v in per_cls.items()}
    return {"acc": acc, "n": total, "per_class_acc": cls_acc}


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    epochs: int = 12,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    patience: int = 4,
    seed: int = 42,
) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = datasets.ImageFolder(data_dir / "train", transform=_tf(image_size, True))
    val_dir = data_dir / "val"
    if not val_dir.is_dir() or not any(val_dir.iterdir()):
        raise FileNotFoundError(val_dir)
    val_ds = datasets.ImageFolder(val_dir, transform=_tf(image_size, False))
    if train_ds.classes != val_ds.classes:
        print(f"warn: train classes {train_ds.classes} vs val {val_ds.classes}", flush=True)

    counts = Counter(train_ds.targets)
    weights = [1.0 / counts[y] for y in train_ds.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_path = out_dir / "best.pt"
    wait = patience
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")
        sched.step()
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(record)
        print(f"epoch {epoch}: loss={record['train_loss']:.4f} acc={metrics['acc']:.4f}", flush=True)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            wait = patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": train_ds.classes,
                    "image_size": image_size,
                    "metrics": record,
                },
                best_path,
            )
        else:
            wait -= 1
            if wait <= 0:
                print("Early stopping.", flush=True)
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "classes.json").write_text(json.dumps(train_ds.classes, indent=2), encoding="utf-8")
    print(f"Saved roof-type checkpoint: {best_path} acc={best_acc:.4f}", flush=True)
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "shared" / "datasets" / "roof_types")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "checkpoints" / "roof_types")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.dataset, args.output, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
