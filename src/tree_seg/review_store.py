"""Review session store: statuses, RGB/mask previews, corrected masks."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

Status = Literal["pending", "correct", "incorrect"]


@dataclass
class ReviewItem:
    tile_id: str
    source: str
    image: str  # relative path under session dir
    mask: str
    status: Status = "pending"
    corrected: bool = False
    notes: str = ""


class ReviewSession:
    """Filesystem-backed review session under ``session_dir``."""

    def __init__(self, session_dir: str | Path) -> None:
        self.root = Path(session_dir)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.corrected_dir = self.root / "corrected"
        self.manifest_path = self.root / "manifest.json"
        for d in (self.images_dir, self.masks_dir, self.corrected_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.items: list[ReviewItem] = []
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.items = [ReviewItem(**row) for row in data.get("items", [])]

    def save(self) -> None:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self.items),
            "items": [asdict(i) for i in self.items],
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def index_of(self, tile_id: str) -> int:
        for i, item in enumerate(self.items):
            if item.tile_id == tile_id:
                return i
        raise KeyError(tile_id)

    def get(self, index: int) -> ReviewItem:
        return self.items[index]

    def upsert_item(
        self,
        tile_id: str,
        source: str | Path,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> ReviewItem:
        """Add or refresh image/mask previews; preserve status if already reviewed."""
        img_rel = f"images/{tile_id}.png"
        msk_rel = f"masks/{tile_id}.png"
        Image.fromarray(rgb).save(self.root / img_rel)
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        Image.fromarray(mask_u8).save(self.root / msk_rel)

        existing = None
        for item in self.items:
            if item.tile_id == tile_id:
                existing = item
                break
        if existing is None:
            item = ReviewItem(tile_id=tile_id, source=str(source), image=img_rel, mask=msk_rel)
            self.items.append(item)
        else:
            existing.source = str(source)
            existing.image = img_rel
            existing.mask = msk_rel
            item = existing
        self.save()
        return item

    def set_status(self, index: int, status: Status) -> ReviewItem:
        item = self.items[index]
        item.status = status
        self.save()
        return item

    def save_corrected_mask(self, index: int, mask: np.ndarray) -> ReviewItem:
        item = self.items[index]
        mask_u8 = _normalize_mask(mask)
        Image.fromarray(mask_u8).save(self.root / item.mask)
        corr_path = self.corrected_dir / f"{item.tile_id}.png"
        Image.fromarray(mask_u8).save(corr_path)
        item.corrected = True
        item.status = "incorrect"
        self.save()
        return item

    def load_rgb(self, index: int) -> np.ndarray:
        return np.array(Image.open(self.root / self.items[index].image).convert("RGB"))

    def load_mask(self, index: int) -> np.ndarray:
        arr = np.array(Image.open(self.root / self.items[index].mask).convert("L"))
        return (arr > 127).astype(np.uint8)

    def counts(self) -> dict[str, int]:
        out = {"pending": 0, "correct": 0, "incorrect": 0, "corrected": 0, "total": len(self.items)}
        for item in self.items:
            out[item.status] = out.get(item.status, 0) + 1
            if item.corrected:
                out["corrected"] += 1
        return out

    def build_finetune_dataset(
        self,
        dataset_dir: str | Path,
        *,
        include_correct: bool = True,
        include_corrected_only: bool = False,
        val_fraction: float = 0.15,
        seed: int = 42,
    ) -> dict[str, int]:
        """
        Copy reviewed tiles into train/val image-mask pairs.

        - include_corrected_only=True: only tiles with hand-fixed masks
        - else: incorrect+corrected always; correct included if include_correct
        """
        import random

        dataset_dir = Path(dataset_dir)
        selected: list[ReviewItem] = []
        for item in self.items:
            if include_corrected_only:
                if item.corrected:
                    selected.append(item)
            else:
                if item.status == "incorrect" and item.corrected:
                    selected.append(item)
                elif item.status == "correct" and include_correct:
                    selected.append(item)

        if not selected:
            raise ValueError(
                "No reviewed tiles ready for fine-tuning. "
                "Mark tiles Correct, or fix Incorrect masks first."
            )

        rng = random.Random(seed)
        ids = selected[:]
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_fraction))) if len(ids) >= 5 else 0
        val_set = set(i.tile_id for i in ids[:n_val])

        for split in ("train", "val"):
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "masks").mkdir(parents=True, exist_ok=True)

        n_train = n_val_count = 0
        for item in selected:
            split = "val" if item.tile_id in val_set else "train"
            shutil.copy2(self.root / item.image, dataset_dir / split / "images" / f"{item.tile_id}.png")
            shutil.copy2(self.root / item.mask, dataset_dir / split / "masks" / f"{item.tile_id}.png")
            if split == "train":
                n_train += 1
            else:
                n_val_count += 1

        meta = {
            "train": n_train,
            "val": n_val_count,
            "include_correct": include_correct,
            "include_corrected_only": include_corrected_only,
            "tile_ids": [i.tile_id for i in selected],
        }
        (dataset_dir / "finetune_selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"train": n_train, "val": n_val_count}


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        # ImageEditor may return RGBA/RGB paint — treat bright/greenish as tree
        if mask.shape[2] == 4:
            alpha = mask[..., 3]
            rgb = mask[..., :3]
            painted = (alpha > 32) & (rgb.max(axis=2) > 32)
            return painted.astype(np.uint8) * 255
        gray = mask.max(axis=2)
        return (gray > 127).astype(np.uint8) * 255
    return (mask > 127).astype(np.uint8) * 255


def overlay_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = (mask > 0).astype(np.float32)[..., None] * alpha
    color = np.zeros_like(out)
    color[..., 1] = 255.0
    return np.clip(out * (1.0 - m) + color * m, 0, 255).astype(np.uint8)


def summarize_session(session: ReviewSession) -> str:
    c = session.counts()
    return (
        f"Total: {c['total']} | Pending: {c['pending']} | "
        f"Correct: {c['correct']} | Incorrect: {c['incorrect']} | "
        f"Hand-fixed: {c['corrected']}"
    )
