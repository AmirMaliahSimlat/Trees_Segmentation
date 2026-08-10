"""SegFormer model loading and single-tile inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


@dataclass
class SegFormerBundle:
    model: SegformerForSemanticSegmentation
    processor: SegformerImageProcessor
    device: torch.device
    id2label: dict[int, str]


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_segformer(
    model_name: str = "restor/tcd-segformer-mit-b5",
    *,
    num_labels: int = 2,
    id2label: dict[int, str] | None = None,
    device: str | None = None,
    local_checkpoint: str | None = None,
) -> SegFormerBundle:
    """Load pretrained OAM-TCD SegFormer or a fine-tuned local checkpoint."""
    id2label = id2label or {0: "background", 1: "tree"}
    # HF may store string keys
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    source = local_checkpoint or model_name
    processor = SegformerImageProcessor.from_pretrained(source)
    model = SegformerForSemanticSegmentation.from_pretrained(
        source,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    dev = resolve_device(device)
    model.to(dev)
    model.eval()
    return SegFormerBundle(model=model, processor=processor, device=dev, id2label=id2label)


@torch.inference_mode()
def predict_tile_proba(
    bundle: SegFormerBundle,
    rgb: np.ndarray,
    *,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Predict tree probability map for one RGB uint8 HxWx3 tile.

    Returns float32 HxW in [0, 1] at the original (or target_size) resolution.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")

    h, w = rgb.shape[:2]
    out_h, out_w = target_size if target_size else (h, w)

    inputs = bundle.processor(images=rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(bundle.device)
    outputs = bundle.model(pixel_values=pixel_values)
    logits = outputs.logits  # B, C, h', w'
    logits = F.interpolate(logits, size=(out_h, out_w), mode="bilinear", align_corners=False)

    if logits.shape[1] == 1:
        proba = torch.sigmoid(logits)[0, 0]
    else:
        # Prefer tree class index 1 when present
        tree_idx = 1 if logits.shape[1] > 1 else 0
        proba = torch.softmax(logits, dim=1)[0, tree_idx]

    return proba.detach().cpu().numpy().astype(np.float32)


def bundle_from_config(cfg: dict[str, Any], *, checkpoint: str | None = None, device: str | None = None) -> SegFormerBundle:
    model_cfg = cfg.get("model", {})
    id2label = model_cfg.get("id2label", {0: "background", 1: "tree"})
    return load_segformer(
        model_name=model_cfg.get("name", "restor/tcd-segformer-mit-b5"),
        num_labels=int(model_cfg.get("num_labels", 2)),
        id2label=id2label,
        device=device,
        local_checkpoint=checkpoint,
    )
