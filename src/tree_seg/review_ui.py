"""Gradio review UI: overlay toggle, correct/incorrect, mask edit, fine-tune."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from tree_seg.batch_predict import batch_predict_folder
from tree_seg.config import load_config
from tree_seg.hf_auth import ensure_hf_auth
from tree_seg.model import bundle_from_config
from tree_seg.review_store import ReviewSession, overlay_rgb, summarize_session
from tree_seg.train import train_from_config

ensure_hf_auth(verbose=False)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_paths() -> dict[str, str]:
    return {
        "raw": str(REPO_ROOT / "data" / "raw" / "GFK"),
        "session": str(REPO_ROOT / "data" / "review"),
        "dataset": str(REPO_ROOT / "data" / "tiles"),
        "checkpoints": str(REPO_ROOT / "outputs" / "checkpoints" / "gfk_finetune"),
        "config": str(REPO_ROOT / "configs" / "default.yaml"),
    }


def _view_image(session: ReviewSession, index: int, show_mask: bool) -> np.ndarray:
    rgb = session.load_rgb(index)
    if not show_mask:
        return rgb
    mask = session.load_mask(index)
    return overlay_rgb(rgb, mask)


def _status_label(session: ReviewSession, index: int) -> str:
    item = session.get(index)
    flag = " (hand-fixed)" if item.corrected else ""
    return f"{index + 1}/{len(session.items)}  |  {item.tile_id}  |  {item.status}{flag}"


def build_app(
    *,
    session_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> gr.Blocks:
    defaults = _default_paths()
    session_dir = Path(session_dir or defaults["session"])
    config_path = Path(config_path or defaults["config"])

    with gr.Blocks(title="Tree canopy review") as demo:
        gr.Markdown(
            """
            # Tree canopy review
            1. Put tiles in `data/raw/`
            2. Click **Run batch predict**
            3. Review each tile — toggle mask, mark **Correct** or **Incorrect**
            4. If incorrect, edit the mask (paint tree = white) and **Save correction**
            5. When done reviewing, click **Fine-tune** (does not run after each fix)
            """
        )

        state_index = gr.State(0)
        state_session = gr.State(str(session_dir))

        with gr.Row():
            raw_dir = gr.Textbox(label="Images folder", value=defaults["raw"])
            session_box = gr.Textbox(label="Review session folder", value=str(session_dir))
            checkpoint = gr.Textbox(
                label="Checkpoint for predict (blank = pretrained HF)",
                value=str(REPO_ROOT / "outputs" / "checkpoints" / "oam_tcd_30cm" / "best"),
                info="Use oam_tcd_30cm/best as the roll-back baseline; UI fine-tunes write to gfk_finetune/",
            )

        with gr.Row():
            btn_predict = gr.Button("Run batch predict", variant="primary")
            btn_reload = gr.Button("Reload session")
            skip_existing = gr.Checkbox(label="Skip already predicted", value=True)

        summary = gr.Textbox(label="Session summary", interactive=False)
        status = gr.Textbox(label="Current tile", interactive=False)

        with gr.Row():
            show_mask = gr.Checkbox(label="Show mask overlay", value=True)
            viewer = gr.Image(label="Tile preview", type="numpy", height=512)

        with gr.Row():
            btn_prev = gr.Button("Previous")
            btn_next = gr.Button("Next")
            btn_correct = gr.Button("Mark Correct", variant="primary")
            btn_incorrect = gr.Button("Mark Incorrect")

        gr.Markdown("### Fix mask (only if Incorrect)")
        gr.Markdown(
            "Paint **tree canopy in white** (or any bright color). Background stays black. "
            "Then click **Save correction**."
        )
        mask_editor = gr.ImageEditor(
            label="Mask editor",
            type="numpy",
            image_mode="RGB",
            brush=gr.Brush(default_size=20, colors=["#ffffff", "#00ff00"], color_mode="fixed"),
            layers=False,
            height=512,
        )
        btn_save_fix = gr.Button("Save correction")

        with gr.Accordion("Fine-tune (run once after reviewing)", open=True):
            include_correct = gr.Checkbox(
                label="Include tiles marked Correct (recommended)",
                value=True,
            )
            corrected_only = gr.Checkbox(
                label="Only hand-fixed (Incorrect) tiles",
                value=False,
            )
            dataset_dir = gr.Textbox(label="Dataset output", value=defaults["dataset"])
            ckpt_dir = gr.Textbox(label="Checkpoint output", value=defaults["checkpoints"])
            epochs = gr.Number(label="Epochs", value=20, precision=0)
            btn_finetune = gr.Button("Fine-tune on reviewed tiles", variant="stop")
            finetune_log = gr.Textbox(label="Fine-tune log", lines=8)

        # ---- helpers bound inside closures ----
        def _session(path: str) -> ReviewSession:
            return ReviewSession(path)

        def load_session(sess_path: str):
            session = _session(sess_path)
            if not session.items:
                return (
                    0,
                    sess_path,
                    summarize_session(session),
                    "No tiles yet — run batch predict.",
                    None,
                    None,
                )
            idx = 0
            rgb = _view_image(session, idx, True)
            mask_rgb = np.stack([session.load_mask(idx) * 255] * 3, axis=-1)
            return (
                idx,
                sess_path,
                summarize_session(session),
                _status_label(session, idx),
                rgb,
                {"background": mask_rgb, "layers": [], "composite": mask_rgb},
            )

        def run_predict(raw: str, sess_path: str, ckpt: str, skip: bool):
            cfg = load_config(config_path)
            ckpt_path = ckpt.strip()
            if ckpt_path and not Path(ckpt_path).exists():
                print(f"Checkpoint not found ({ckpt_path}); using pretrained HF weights.")
                ckpt_path = ""
            bundle = bundle_from_config(
                cfg,
                checkpoint=ckpt_path or None,
                device=None,
            )
            batch_predict_folder(
                raw,
                sess_path,
                bundle,
                cfg,
                skip_existing=skip,
            )
            return load_session(sess_path)

        def nav(delta: int, idx: int, sess_path: str, show: bool):
            session = _session(sess_path)
            if not session.items:
                return idx, summarize_session(session), "Empty session", None, None
            idx = int(np.clip(idx + delta, 0, len(session.items) - 1))
            rgb = _view_image(session, idx, show)
            mask_rgb = np.stack([session.load_mask(idx) * 255] * 3, axis=-1)
            return (
                idx,
                summarize_session(session),
                _status_label(session, idx),
                rgb,
                {"background": mask_rgb, "layers": [], "composite": mask_rgb},
            )

        def refresh_view(idx: int, sess_path: str, show: bool):
            session = _session(sess_path)
            if not session.items:
                return None, "Empty session", summarize_session(session)
            idx = int(np.clip(idx, 0, len(session.items) - 1))
            return (
                _view_image(session, idx, show),
                _status_label(session, idx),
                summarize_session(session),
            )

        def mark_correct(idx: int, sess_path: str, show: bool):
            session = _session(sess_path)
            if not session.items:
                return idx, summarize_session(session), "Empty", None, None
            session.set_status(idx, "correct")
            next_idx = min(idx + 1, len(session.items) - 1)
            rgb = _view_image(session, next_idx, show)
            mask_rgb = np.stack([session.load_mask(next_idx) * 255] * 3, axis=-1)
            return (
                next_idx,
                summarize_session(session),
                _status_label(session, next_idx),
                rgb,
                {"background": mask_rgb, "layers": [], "composite": mask_rgb},
            )

        def mark_incorrect(idx: int, sess_path: str, show: bool):
            session = _session(sess_path)
            if not session.items:
                return idx, summarize_session(session), "Empty", None, None
            session.set_status(idx, "incorrect")
            rgb = _view_image(session, idx, show)
            mask_rgb = np.stack([session.load_mask(idx) * 255] * 3, axis=-1)
            return (
                idx,
                summarize_session(session),
                _status_label(session, idx),
                rgb,
                {"background": mask_rgb, "layers": [], "composite": mask_rgb},
            )

        def save_fix(editor_data: Any, idx: int, sess_path: str, show: bool):
            session = _session(sess_path)
            if not session.items:
                return summarize_session(session), "", None

            mask = None
            if isinstance(editor_data, dict):
                for key in ("composite", "background", "image"):
                    if editor_data.get(key) is not None:
                        mask = np.array(editor_data[key])
                        break
                if mask is None and editor_data.get("layers"):
                    mask = np.array(editor_data["layers"][0])
            elif editor_data is not None:
                mask = np.array(editor_data)

            if mask is None:
                raise gr.Error("No mask to save — paint on the editor first.")

            session.save_corrected_mask(idx, mask)
            rgb = _view_image(session, idx, show)
            return summarize_session(session), _status_label(session, idx), rgb

        def do_finetune(
            sess_path: str,
            ds_dir: str,
            out_ckpt: str,
            inc_correct: bool,
            only_corr: bool,
            n_epochs: float,
        ):
            session = _session(sess_path)
            try:
                counts = session.build_finetune_dataset(
                    ds_dir,
                    include_correct=inc_correct,
                    include_corrected_only=only_corr,
                )
            except ValueError as exc:
                return str(exc)

            cfg = load_config(config_path)
            cfg.setdefault("train", {})["epochs"] = int(n_epochs)
            best = train_from_config(cfg, ds_dir, out_ckpt)
            return (
                f"Dataset train={counts['train']} val={counts['val']}\n"
                f"Best checkpoint: {best}\n"
                f"Use this checkpoint in batch predict / predict_geotiff --checkpoint"
            )

        out_load = [state_index, state_session, summary, status, viewer, mask_editor]
        btn_predict.click(
            run_predict,
            inputs=[raw_dir, session_box, checkpoint, skip_existing],
            outputs=out_load,
        )
        btn_reload.click(load_session, inputs=[session_box], outputs=out_load)
        btn_prev.click(
            lambda i, s, show: nav(-1, i, s, show),
            inputs=[state_index, session_box, show_mask],
            outputs=[state_index, summary, status, viewer, mask_editor],
        )
        btn_next.click(
            lambda i, s, show: nav(1, i, s, show),
            inputs=[state_index, session_box, show_mask],
            outputs=[state_index, summary, status, viewer, mask_editor],
        )
        show_mask.change(
            refresh_view,
            inputs=[state_index, session_box, show_mask],
            outputs=[viewer, status, summary],
        )
        btn_correct.click(
            mark_correct,
            inputs=[state_index, session_box, show_mask],
            outputs=[state_index, summary, status, viewer, mask_editor],
        )
        btn_incorrect.click(
            mark_incorrect,
            inputs=[state_index, session_box, show_mask],
            outputs=[state_index, summary, status, viewer, mask_editor],
        )
        btn_save_fix.click(
            save_fix,
            inputs=[mask_editor, state_index, session_box, show_mask],
            outputs=[summary, status, viewer],
        )
        btn_finetune.click(
            do_finetune,
            inputs=[
                session_box,
                dataset_dir,
                ckpt_dir,
                include_correct,
                corrected_only,
                epochs,
            ],
            outputs=[finetune_log],
        )

        demo.load(load_session, inputs=[session_box], outputs=out_load)

    return demo


def launch_review_ui(
    *,
    session_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    app = build_app(session_dir=session_dir, config_path=config_path)
    app.launch(server_name=server_name, server_port=server_port, share=share)
