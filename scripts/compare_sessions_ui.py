#!/usr/bin/env python
"""Side-by-side Gradio UI: pretrained vs fine-tuned review sessions."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from tree_seg.review_store import overlay_rgb

REPO = Path(__file__).resolve().parents[1]
BEFORE = REPO / "data" / "review_pretrained"
AFTER = REPO / "data" / "review"


def _load_rgb_mask(session: Path, tile_id: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(session / "images" / f"{tile_id}.png").convert("RGB"))
    mask = np.asarray(Image.open(session / "masks" / f"{tile_id}.png"))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return rgb, mask


def _tile_ids() -> list[str]:
    ids = sorted(p.stem for p in (AFTER / "images").glob("*.png"))
    if not ids:
        ids = sorted(p.stem for p in (BEFORE / "images").glob("*.png"))
    return ids


def build_gallery() -> list[tuple[np.ndarray, str]]:
    items: list[tuple[np.ndarray, str]] = []
    for tile_id in _tile_ids():
        rgb_b, mask_b = _load_rgb_mask(BEFORE, tile_id)
        rgb_a, mask_a = _load_rgb_mask(AFTER, tile_id)
        # Prefer after RGB (same source imagery) for both overlays
        rgb = rgb_a if rgb_a.shape == mask_a.shape[:2] + (3,) else rgb_b
        if rgb.shape[:2] != mask_b.shape[:2]:
            mask_b = np.asarray(
                Image.fromarray(mask_b).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
            )
        if rgb.shape[:2] != mask_a.shape[:2]:
            mask_a = np.asarray(
                Image.fromarray(mask_a).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
            )
        before = overlay_rgb(rgb, mask_b)
        after = overlay_rgb(rgb, mask_a)
        items.append((before, f"{tile_id} — BEFORE fine-tune"))
        items.append((after, f"{tile_id} — AFTER fine-tune"))
    return items


def main() -> None:
    gallery = build_gallery()
    with gr.Blocks(title="Before vs after fine-tune") as demo:
        gr.Markdown(
            "# Tree masks: before vs after fine-tune\n"
            f"**Before:** `{BEFORE}` (pretrained HF) · "
            f"**After:** `{AFTER}` (fine-tuned checkpoint)\n\n"
            "Gallery order: each tile appears twice — before, then after."
        )
        gr.Gallery(
            value=gallery,
            columns=2,
            rows=5,
            height="auto",
            object_fit="contain",
            label="All 10 overlays (5 tiles × before/after)",
        )
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)


if __name__ == "__main__":
    main()
