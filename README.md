# Binary tree canopy semantic segmentation

Pixel-level **tree vs not-tree** masks on orthographic GeoTIFFs (~30 cm/px), using Restor’s OAM-TCD **SegFormer**. Output is a **shapefile of tree-area footprints** (polygons), not individual crown IDs.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -e .

# Hugging Face auth (higher rate limits / faster Hub downloads)
copy .env.example .env
# Edit .env and set HF_TOKEN=... from https://huggingface.co/settings/tokens
```

Put your tile images in `data/<map>/Imagery/` (GeoTIFF preferred; PNG/JPG also work for review/train).

### Large single-map GeoTIFF (e.g. 35 GB)

Do **not** run predict on the whole file. Split it first into ~10k×10k tiles (streaming, low RAM):

```bash
python scripts/geo/split_geotiff.py -i path/to/huge_map.tif -o data/MAP_NAME/Imagery --tile-size 10240
```

Then batch-predict the tile folder as usual:

```bash
python scripts/geo/batch_predict.py -i data/MAP_NAME/Imagery --checkpoint outputs/checkpoints/oam_tcd_30cm/best --no-skip
```

### Checkpoints (roll-back)

| Path | Purpose |
|------|---------|
| `outputs/checkpoints/oam_tcd_30cm/best` | Baseline fine-tuned on OAM-TCD downsampled to ~30 cm — **keep / roll back here** |
| `outputs/checkpoints/gfk_finetune/` | Later domain / UI fine-tunes on your imagery |

```bash
# Prepare OAM-TCD @ 30cm (once)
python scripts/trees/prepare_oam_tcd_30cm.py -o data/shared/datasets/oam_tcd_30cm --max-train 1200 --max-val 150

# Train isolated baseline
python scripts/trees/train_oam_tcd_30cm.py
```

### Main workflow (recommended)

```bash
# Launch review UI (batch predict + review + fine-tune)
python scripts/trees/review_ui.py
```

Open http://127.0.0.1:7860 then:

1. **Run batch predict** on `data/<map>/Imagery/`
2. Browse tiles — toggle **Show mask overlay**
3. **Mark Correct** or **Mark Incorrect**
4. If incorrect: paint the mask (white = tree) → **Save correction**
5. After reviewing many tiles → **Fine-tune on reviewed tiles** (does not run after each fix)

Then re-run batch predict with the new checkpoint path filled in the UI, or:

```bash
python scripts/geo/batch_predict.py -i data/GFK/Imagery --checkpoint outputs/checkpoints/best
```

### Export shapefile footprints

Per tile (folder with `.shp` + sidecars + clean `.tif`):

```bash
python scripts/geo/export_shapefile.py -i path/to/tile_tree_mask.tif -o outputs/footprints/GFK/trees
```

Whole map (all tile masks → per-tile folders + **one combined shapefile**):

```bash
python scripts/geo/export_map_shapefile.py -i data/GFK/Review/geotiff_masks -o outputs/footprints/GFK/trees --map-name GFK
```

Or merge existing per-tile shapefiles only:

```bash
python scripts/geo/merge_shapefiles.py -i outputs/footprints/GFK/trees -o outputs/footprints/GFK/trees/GFK_tree_footprints.shp
```

Combined output: `GFK_tree_footprints.shp` (EPSG from source GeoTIFFs) for QGIS.

### Single-file CLI predict

```bash
python scripts/geo/predict_geotiff.py -i data/Nablus/Imagery/one_tile.tif -o outputs/pred
```

## Config

See [`configs/default.yaml`](configs/default.yaml).

## Model notes

- Default weights: [`restor/tcd-segformer-mit-b5`](https://huggingface.co/restor/tcd-segformer-mit-b5)
- SegFormer is under NVIDIA research license — fine for internal tooling
- At 30 cm/px, zero-shot may confuse grass; fine-tune on corrected tiles after review

## Layout

```
data/<map>/Imagery/          orthophotos / TMS tiles
data/<map>/Buildings/        drop building-footprint shapefiles here
data/<map>/Review/           review sessions for that map
data/shared/datasets/        training chips (not map-specific)
data/shared/sources/         Bonn zip, EuroCrops
outputs/checkpoints/         trained weights (do not mix tasks)
outputs/footprints/<map>/    trees | orchards | roofs shapefiles
outputs/pred/<map>/          raster masks
outputs/logs/                pipeline / train logs
scripts/trees|orchards|roofs|geo|dev
```

Maps: `Fort_Riley`, `GFK`, `Nablus`.

Building roof classification (after dropping shapefiles into `data/<map>/Buildings/`):

```bash
python scripts/roofs/classify_building_roofs.py `
  --shapefile data/Nablus/Buildings/buildings.shp `
  --raster data/Nablus/Imagery/<tile>.tif `
  -o outputs/footprints/Nablus/roofs/buildings_with_roof_type.shp
```
