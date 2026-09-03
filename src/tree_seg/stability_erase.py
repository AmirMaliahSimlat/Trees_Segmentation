"""Stability AI Erase (v2beta) — object removal; the API fills the hole.

Does not call the Inpaint endpoint. Local LaMa/PatchMatch stay in ground_fill.py.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image

from tree_seg.hf_auth import load_dotenv_files
from tree_seg.io_geotiff import iter_tile_specs

ERASE_URL = "https://api.stability.ai/v2beta/stable-image/edit/erase"
# Official cap is 4_194_304 px (~2048^2). Stay under it.
DEFAULT_MAX_PIXELS = 1920 * 1920


def stability_api_key() -> str:
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except ImportError:
        load_dotenv_files()
    key = (
        os.environ.get("STABILITY_API_KEY")
        or os.environ.get("SAI_API_KEY")
        or os.environ.get("STABILITY_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "STABILITY_API_KEY is missing. Put it in the project .env "
            "(see .env.example). Get a key at https://platform.stability.ai/account/keys"
        )
    return key


def _png_rgb(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
    return buf.getvalue()


def _png_mask(mask: np.ndarray) -> bytes:
    m = np.where(mask > 0, 255, 0).astype(np.uint8)
    rgb = np.stack([m, m, m], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def erase_image(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    api_key: str | None = None,
    grow_mask: int = 0,
    seed: int = 0,
    timeout: float = 180.0,
) -> np.ndarray:
    """One Erase request. ``rgb`` HxWx3 uint8, ``mask`` HxW (nonzero = remove)."""
    if not np.any(mask):
        return rgb
    key = api_key or stability_api_key()
    data = {"output_format": "png"}
    if int(grow_mask) > 0:
        data["grow_mask"] = str(int(grow_mask))
    if int(seed) > 0:
        data["seed"] = str(int(seed))
    resp = requests.post(
        ERASE_URL,
        headers={"Authorization": f"Bearer {key}", "Accept": "image/*"},
        files={
            "image": ("image.png", _png_rgb(rgb), "image/png"),
            "mask": ("mask.png", _png_mask(mask), "image/png"),
        },
        data=data,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Stability erase HTTP {resp.status_code}: {resp.text[:800]}")
    out = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
    if out.shape[:2] != rgb.shape[:2]:
        out = cv2.resize(out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    hole = mask > 0
    return np.where(hole[..., None], out, rgb)


def pick_object_window(
    obj: np.ndarray,
    side: int,
    *,
    step: int | None = None,
    min_frac: float = 0.12,
    max_frac: float = 0.62,
) -> tuple[int, int, float]:
    """Return (row, col, object_fraction) for a ``side``×``side`` crop with mixed objects + ground."""
    h, w = obj.shape[:2]
    side = int(min(side, h, w))
    if side < 1:
        return 0, 0, 0.0
    integ = cv2.integral((obj > 0).astype(np.float64))
    step = int(step) if step is not None else max(32, side // 20)
    best: tuple[float, int, int, float] | None = None
    area = float(side * side)
    for r in range(0, h - side + 1, step):
        for c in range(0, w - side + 1, step):
            s = float(integ[r + side, c + side] - integ[r, c + side] - integ[r + side, c] + integ[r, c])
            frac = s / area
            in_band = min_frac <= frac <= max_frac
            score = frac if in_band else (frac * 0.25 if frac < min_frac else (1.0 - frac) * 0.2)
            if best is None or score > best[0]:
                best = (score, r, c, frac)
    assert best is not None
    return best[1], best[2], best[3]


class StabilityEraseFiller:
    """Drop-in filler: Erase API. Refuses oversized images unless ``allow_tile`` is set."""

    guard_sources = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        grow_mask: int = 0,
        seed: int = 0,
        timeout: float = 180.0,
        allow_tile: bool = False,
    ) -> None:
        self.api_key = api_key
        self.max_pixels = int(max_pixels)
        self.grow_mask = int(grow_mask)
        self.seed = int(seed)
        self.timeout = float(timeout)
        self.allow_tile = bool(allow_tile)
        self.backend = "stability_erase"
        self.calls = 0

    def fill(self, rgb: np.ndarray, mask: np.ndarray, source_ban: np.ndarray | None = None) -> np.ndarray:
        if not np.any(mask):
            return rgb
        h, w = rgb.shape[:2]
        n = h * w
        if n > self.max_pixels:
            if not self.allow_tile:
                raise RuntimeError(
                    f"Image is {w}x{h} ({n} px); Stability Erase max is {self.max_pixels} px. "
                    "Crop first, or set allow_tile=true to split into blocks (each block is 5 credits)."
                )
            return self._fill_tiled(rgb, mask)
        self.calls += 1
        return erase_image(
            rgb,
            mask,
            api_key=self.api_key,
            grow_mask=self.grow_mask,
            seed=self.seed,
            timeout=self.timeout,
        )

    def _fill_tiled(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        side = int(self.max_pixels**0.5)
        h, w = rgb.shape[:2]
        out = rgb.copy()
        acc = np.zeros((h, w, 3), dtype=np.float32)
        wt = np.zeros((h, w), dtype=np.float32)
        for spec in iter_tile_specs(h, w, tile_size=side, overlap=0.08):
            r0, c0, hh, ww = spec.row_off, spec.col_off, spec.height, spec.width
            m = mask[r0 : r0 + hh, c0 : c0 + ww]
            if not np.any(m):
                continue
            tile = rgb[r0 : r0 + hh, c0 : c0 + ww]
            self.calls += 1
            filled = erase_image(
                tile,
                m,
                api_key=self.api_key,
                grow_mask=self.grow_mask,
                seed=self.seed,
                timeout=self.timeout,
            )
            wy = np.hanning(max(hh, 2)).astype(np.float32)
            wx = np.hanning(max(ww, 2)).astype(np.float32)
            if hh == 1:
                wy[:] = 1.0
            if ww == 1:
                wx[:] = 1.0
            wwgt = wy[:hh, None] * wx[None, :ww]
            acc[r0 : r0 + hh, c0 : c0 + ww] += filled.astype(np.float32) * wwgt[..., None]
            wt[r0 : r0 + hh, c0 : c0 + ww] += wwgt
        use = wt > 1e-6
        if use.any():
            denom = np.maximum(wt[use], 1e-6)[:, None]
            blended = np.zeros_like(rgb)
            blended[use] = np.clip(acc[use] / denom, 0, 255).astype(np.uint8)
            hole = mask > 0
            out = np.where(hole[..., None], blended, rgb)
        return out


def make_stability_filler(inp: dict[str, Any] | None = None) -> StabilityEraseFiller:
    inp = inp or {}
    return StabilityEraseFiller(
        max_pixels=int(inp.get("stability_max_pixels", DEFAULT_MAX_PIXELS)),
        grow_mask=int(inp.get("stability_grow_mask", 0)),
        seed=int(inp.get("stability_seed", 0)),
        timeout=float(inp.get("stability_timeout", 180)),
        allow_tile=bool(inp.get("stability_allow_tile", False)),
    )
