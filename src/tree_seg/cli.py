"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from tree_seg.hf_auth import ensure_hf_auth

ensure_hf_auth(verbose=False)

from tree_seg.annotate_export import import_corrected_masks, prepare_annotation_batch
from tree_seg.config import load_config
from tree_seg.model import bundle_from_config
from tree_seg.postprocess import export_from_config
from tree_seg.predict import predict_from_config
from tree_seg.train import train_from_config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | mps")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional fine-tuned model directory",
    )


def predict_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict tree canopy mask from GeoTIFF")
    _add_common(parser)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input GeoTIFF")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    bundle = bundle_from_config(
        cfg,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )
    paths = predict_from_config(args.input, args.output, cfg, bundle)
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


def prepare_annotation_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare lean HITL annotation batch")
    _add_common(parser)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input GeoTIFF")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Annotation batch dir")
    parser.add_argument(
        "--import-to",
        type=Path,
        default=None,
        help="If set, import corrected masks from --output into this dataset root",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    a = cfg.get("annotation", {})
    p = cfg.get("predict", {})

    if args.import_to:
        counts = import_corrected_masks(args.output, args.import_to)
        print(f"Imported train={counts['train']} val={counts['val']} into {args.import_to}")
        return

    bundle = bundle_from_config(
        cfg,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )
    manifest = prepare_annotation_batch(
        args.input,
        args.output,
        bundle,
        tile_size=int(a.get("tile_size", 1024)),
        sample_stride=int(a.get("sample_stride", 512)),
        max_tiles=int(a.get("max_tiles", 40)),
        uncertainty_top_k=int(a.get("uncertainty_top_k", 30)),
        overlay_alpha=float(a.get("overlay_alpha", 0.45)),
        threshold=float(p.get("threshold", 0.5)),
    )
    print(f"Annotation batch ready: {manifest}")
    print(f"See {Path(args.output) / 'README_ANNOTATION.md'}")


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer on corrected tiles")
    _add_common(parser)
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        required=True,
        help="Dataset root with train/images, train/masks, optional val/",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Checkpoint output directory",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    best = train_from_config(cfg, args.dataset, args.output)
    print(f"Best checkpoint: {best}")


def export_shapefile_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export tree-area footprint polygons as an ESRI Shapefile"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Probability or mask GeoTIFF",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--geojson",
        action="store_true",
        help="Also write a GeoJSON copy of the footprints",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.geojson:
        cfg.setdefault("postprocess", {})["write_geojson"] = True
    paths = export_from_config(args.input, args.output, cfg)
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


def merge_shapefiles_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-tile footprint shapefiles into one map shapefile"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Folder containing per-tile footprint shapefiles (searched recursively)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output combined .shp path (e.g. outputs/footprints/GFK/trees/GFK_tree_footprints.shp)",
    )
    parser.add_argument(
        "--dissolve",
        action="store_true",
        help="Union touching/overlapping polygons into one layer geometry set",
    )
    args = parser.parse_args(argv)

    from tree_seg.postprocess import merge_footprint_shapefiles

    out = merge_footprint_shapefiles(args.input, args.output, dissolve=args.dissolve)
    print(f"Wrote combined shapefile: {out}")


def export_map_shapefile_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export all mask GeoTIFFs in a folder to per-tile shapefiles + one map shapefile"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Folder of *_tree_mask.tif files (e.g. data/GFK/Review/geotiff_masks)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output map folder (e.g. outputs/footprints/GFK/trees)",
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default=None,
        help="Name for combined shapefile prefix (default: output folder name)",
    )
    args = parser.parse_args(argv)

    from tree_seg.postprocess import export_map_footprints

    cfg = load_config(args.config)
    result = export_map_footprints(
        args.input,
        args.output,
        cfg,
        map_name=args.map_name or Path(args.output).name,
    )
    print(f"Tiles exported: {result['n_tiles']}")
    print(f"Combined map shapefile: {result['map_shapefile']}")


def batch_predict_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch-predict a folder of tiles into a review session")
    _add_common(parser)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Folder of tile images")
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Review session directory (default: data/GFK/Review)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-predict tiles even if already in the session",
    )
    parser.add_argument(
        "--full-resolution",
        action="store_true",
        help="Run full tiled GeoTIFF inference (needed for accurate shapefile export)",
    )
    args = parser.parse_args(argv)

    from tree_seg.batch_predict import batch_predict_folder
    from tree_seg.review_store import summarize_session

    cfg = load_config(args.config)
    bundle = bundle_from_config(
        cfg,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )
    root = Path(__file__).resolve().parents[2]
    session_dir = args.session or (root / "data" / "GFK" / "Review")
    session = batch_predict_folder(
        args.input,
        session_dir,
        bundle,
        cfg,
        skip_existing=not args.no_skip,
        full_resolution=args.full_resolution,
    )
    print(summarize_session(session))
    print(f"Session: {session_dir}")


def split_geotiff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split a large GeoTIFF into smaller georeferenced tiles (streaming)"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Large input GeoTIFF")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output tile folder")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=10240,
        help="Tile edge length in pixels (default: 10240, same scale as GFK tiles)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap in pixels between tiles (default: 0; predict adds its own overlap)",
    )
    parser.add_argument("--prefix", type=str, default=None, help="Output filename prefix")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep nearly empty / nodata tiles (skipped by default)",
    )
    args = parser.parse_args(argv)

    from tree_seg.split_geotiff import split_geotiff

    summary = split_geotiff(
        args.input,
        args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        skip_empty=not args.keep_empty,
        prefix=args.prefix,
    )
    print(
        f"Wrote {summary['tiles_written']} tiles "
        f"({summary.get('tiles_resumed', 0)} resumed, "
        f"{summary['tiles_skipped_empty']} empty skipped) → {args.output}"
    )
    print(f"Manifest: {Path(args.output) / 'manifest.json'}")


def review_ui_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch tree canopy review UI")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--session", type=Path, default=None, help="Review session folder")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    from tree_seg.review_ui import launch_review_ui

    launch_review_ui(
        session_dir=args.session,
        config_path=args.config,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


# Backwards-compatible alias
export_unreal_main = export_shapefile_main


if __name__ == "__main__":
    predict_main()
