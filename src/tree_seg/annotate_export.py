"""HITL annotation batch preparation and Label Studio helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from tree_seg.io_geotiff import (
    iter_tile_specs,
    open_rgb_geotiff,
    read_tile_rgb,
    save_mask_png,
    save_tile_png,
)
from tree_seg.model import SegFormerBundle, predict_tile_proba


def uncertainty_score(proba: np.ndarray) -> float:
    """
    Mean binary entropy — high when the model is unsure (near 0.5).

    Also elevated near decision boundary on grass/tree mix.
    """
    p = np.clip(proba.astype(np.float64), 1e-6, 1.0 - 1e-6)
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(ent.mean())


def make_overlay(rgb: np.ndarray, proba: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Green overlay proportional to tree probability."""
    overlay = rgb.astype(np.float32).copy()
    color = np.zeros_like(overlay)
    color[..., 1] = 255.0  # green
    w = np.clip(proba, 0, 1)[..., None] * alpha
    out = overlay * (1.0 - w) + color * w
    return np.clip(out, 0, 255).astype(np.uint8)


def prepare_annotation_batch(
    input_path: str | Path,
    output_dir: str | Path,
    bundle: SegFormerBundle,
    *,
    tile_size: int = 1024,
    sample_stride: int | None = None,
    max_tiles: int = 40,
    uncertainty_top_k: int = 30,
    overlay_alpha: float = 0.45,
    threshold: float = 0.5,
) -> Path:
    """
    Sample tiles, score uncertainty, keep the hardest ones for human correction.

    Writes under ``output_dir``:
      images/*.png
      proposals/*.png   (binary proposal masks)
      overlays/*.png
      manifest.json
      label_studio_tasks.json
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    prop_dir = output_dir / "proposals"
    overlay_dir = output_dir / "overlays"
    for d in (img_dir, prop_dir, overlay_dir):
        d.mkdir(parents=True, exist_ok=True)

    overlap = 0.0
    if sample_stride is not None and sample_stride < tile_size:
        overlap = 1.0 - (sample_stride / tile_size)

    candidates: list[dict[str, Any]] = []

    with open_rgb_geotiff(input_path) as ds:
        eff_tile = min(tile_size, ds.height, ds.width)
        specs = list(iter_tile_specs(ds.height, ds.width, tile_size=eff_tile, overlap=overlap))
        # Cap candidate pool before scoring if huge
        if len(specs) > max_tiles * 8:
            step = max(1, len(specs) // (max_tiles * 8))
            specs = specs[::step]

        for i, spec in enumerate(tqdm(specs, desc="Score tiles", unit="tile")):
            rgb = read_tile_rgb(ds, spec, eff_tile)
            # Skip nearly empty / nodata-ish tiles
            if float(rgb.mean()) < 5.0:
                continue
            proba = predict_tile_proba(bundle, rgb, target_size=(eff_tile, eff_tile))
            score = uncertainty_score(proba)
            # Prefer tiles that contain both classes (more informative)
            tree_frac = float((proba >= threshold).mean())
            if tree_frac < 0.01 or tree_frac > 0.95:
                score *= 0.5
            candidates.append(
                {
                    "index": i,
                    "row_off": spec.row_off,
                    "col_off": spec.col_off,
                    "height": spec.height,
                    "width": spec.width,
                    "uncertainty": score,
                    "tree_frac": tree_frac,
                    "rgb": rgb,
                    "proba": proba,
                    "tile_size": eff_tile,
                }
            )

    candidates.sort(key=lambda c: c["uncertainty"], reverse=True)
    selected = candidates[: min(uncertainty_top_k, max_tiles, len(candidates))]

    manifest: list[dict[str, Any]] = []
    ls_tasks: list[dict[str, Any]] = []

    for rank, item in enumerate(selected):
        tile_id = f"{input_path.stem}_r{item['row_off']}_c{item['col_off']}"
        img_name = f"{tile_id}.png"
        mask_name = f"{tile_id}.png"

        save_tile_png(img_dir / img_name, item["rgb"])
        proposal = (item["proba"] >= threshold).astype(np.uint8)
        save_mask_png(prop_dir / mask_name, proposal)
        overlay = make_overlay(item["rgb"], item["proba"], alpha=overlay_alpha)
        save_tile_png(overlay_dir / img_name, overlay)

        # Seed masks/ for correction workflow (copy of proposal to edit)
        seed_dir = output_dir / "masks"
        seed_dir.mkdir(parents=True, exist_ok=True)
        save_mask_png(seed_dir / mask_name, proposal)

        entry = {
            "tile_id": tile_id,
            "rank": rank,
            "image": f"images/{img_name}",
            "proposal": f"proposals/{mask_name}",
            "mask": f"masks/{mask_name}",
            "overlay": f"overlays/{img_name}",
            "row_off": item["row_off"],
            "col_off": item["col_off"],
            "uncertainty": item["uncertainty"],
            "tree_frac": item["tree_frac"],
            "source_geotiff": str(input_path),
        }
        manifest.append(entry)

        # Label Studio brush pre-annotation (RLE-like via image path reference)
        ls_tasks.append(
            {
                "data": {
                    "image": f"images/{img_name}",
                    "tile_id": tile_id,
                },
                "meta": {
                    "proposal": f"proposals/{mask_name}",
                    "uncertainty": item["uncertainty"],
                },
            }
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": str(input_path),
                "tile_size": selected[0].get("tile_size", tile_size) if selected else tile_size,
                "count": len(manifest),
                "tiles": manifest,
                "instructions": (
                    "Edit masks/*.png (white=tree, black=background). "
                    "Focus on removing grass/crops false positives. "
                    "Keep 5-10 tiles aside as holdout (move to masks_holdout/)."
                ),
            },
            f,
            indent=2,
        )

    ls_path = output_dir / "label_studio_tasks.json"
    with ls_path.open("w", encoding="utf-8") as f:
        json.dump(ls_tasks, f, indent=2)

    # Label Studio labeling config snippet
    config_path = output_dir / "label_studio_config.xml"
    config_path.write_text(
        """<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="tag" toName="image">
    <Label value="tree" background="#00FF00"/>
  </BrushLabels>
</View>
""",
        encoding="utf-8",
    )

    readme = output_dir / "README_ANNOTATION.md"
    readme.write_text(
        f"""# Annotation batch ({len(manifest)} tiles)

## Quick path (no Label Studio)

1. Open `overlays/` to see model proposals (green = predicted tree).
2. Edit `masks/*.png` in any editor (GIMP, Photoshop, Paint.NET):
   - White (255) = tree canopy
   - Black (0) = not tree
3. Prioritize deleting grass / lawn / crops wrongly marked as tree.
4. Move 5–10 finished tiles to `masks_holdout/` (+ matching images) for validation.
5. Remaining `images/` + `masks/` pairs go to fine-tuning.

Budget target: **20–30 train tiles** + **5–10 holdout** (cap ~50).

## Label Studio path

1. Create a project with `label_studio_config.xml`.
2. Import `label_studio_tasks.json` (serve `images/` as local files).
3. Export brush masks and place PNGs under `masks/` with the same filenames.

Source: `{input_path}`
""",
        encoding="utf-8",
    )

    return manifest_path


def import_corrected_masks(
    annotation_dir: str | Path,
    dataset_dir: str | Path,
    *,
    holdout_dir_name: str = "masks_holdout",
) -> dict[str, int]:
    """
    Copy corrected image/mask pairs into train and val dataset folders.

    Expects:
      annotation_dir/images/*.png
      annotation_dir/masks/*.png
      annotation_dir/masks_holdout/*.png (optional)
    """
    import shutil

    annotation_dir = Path(annotation_dir)
    dataset_dir = Path(dataset_dir)
    train_img = dataset_dir / "train" / "images"
    train_msk = dataset_dir / "train" / "masks"
    val_img = dataset_dir / "val" / "images"
    val_msk = dataset_dir / "val" / "masks"
    for d in (train_img, train_msk, val_img, val_msk):
        d.mkdir(parents=True, exist_ok=True)

    images = annotation_dir / "images"
    masks = annotation_dir / "masks"
    holdout = annotation_dir / holdout_dir_name

    n_train = n_val = 0
    for mask_path in sorted(masks.glob("*.png")):
        img_path = images / mask_path.name
        if not img_path.exists():
            continue
        shutil.copy2(img_path, train_img / mask_path.name)
        shutil.copy2(mask_path, train_msk / mask_path.name)
        n_train += 1

    if holdout.exists():
        for mask_path in sorted(holdout.glob("*.png")):
            img_path = images / mask_path.name
            if not img_path.exists():
                # also allow holdout images folder
                alt = annotation_dir / "images_holdout" / mask_path.name
                img_path = alt if alt.exists() else img_path
            if not img_path.exists():
                continue
            shutil.copy2(img_path, val_img / mask_path.name)
            shutil.copy2(mask_path, val_msk / mask_path.name)
            n_val += 1

    return {"train": n_train, "val": n_val}
