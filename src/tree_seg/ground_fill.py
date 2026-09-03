"""Segment buildings/trees (OpenEarthMap) and inpaint them with LaMa.

Does not use building shapefiles. Does not touch tree/orchard/roof checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from tree_seg.io_geotiff import (
    iter_tile_specs,
    open_rgb_geotiff,
    read_rgb,
    write_geotiff,
)

OBJECT_NAME_HINTS = ("tree", "building")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


OEM8 = {
    0: "bareland",
    1: "grass",
    2: "pavement",
    3: "road",
    4: "tree",
    5: "water",
    6: "cropland",
    7: "building",
}


def load_landcover(hf_id: str, device: torch.device | None = None):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    device = device or _device()
    processor = AutoImageProcessor.from_pretrained(hf_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(hf_id)
    model.to(device)
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if all(str(v).startswith("label_") for v in id2label.values()) and set(id2label) == set(range(8)):
        id2label = dict(OEM8)
    object_ids = {
        i for i, name in id2label.items() if any(h in name for h in OBJECT_NAME_HINTS)
    }
    if not object_ids:
        object_ids = {4, 7}
    return processor, model, id2label, object_ids, device


class LamaFiller:
    """TorchScript Big-LaMa; OpenCV Telea if the weights are missing."""

    def __init__(self, weights: Path | None = None, device: torch.device | None = None) -> None:
        self.device = device or _device()
        self._lama = None
        self.backend = "opencv"
        default = Path(__file__).resolve().parents[2] / "outputs" / "checkpoints" / "ground_fill" / "big-lama.pt"
        path = Path(weights) if weights else default
        if path.is_file():
            try:
                self._lama = torch.jit.load(str(path), map_location=self.device)
                self._lama.eval()
                self.backend = "lama"
            except Exception:
                self._lama = None

    def fill(self, rgb: np.ndarray, mask: np.ndarray, source_ban: np.ndarray | None = None) -> np.ndarray:
        if not mask.any():
            return rgb
        if self._lama is not None:
            return self._lama_fill(rgb, mask)
        m = (mask > 0).astype(np.uint8)
        return cv2.inpaint(rgb, m, 3, cv2.INPAINT_TELEA)

    def _lama_fill(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        img = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        m = (mask > 0).astype(np.float32)[None, ...]
        img_p = _pad_to_modulo(img, 8)
        m_p = _pad_to_modulo(m, 8)
        with torch.inference_mode():
            out = self._lama(
                torch.from_numpy(img_p)[None].to(self.device),
                torch.from_numpy(m_p)[None].to(self.device),
            )[0]
        arr = out.permute(1, 2, 0).detach().cpu().numpy()
        arr = np.clip(arr[:h, :w] * 255.0, 0, 255).astype(np.uint8)
        return arr


SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SDXL_PROMPT = (
    "aerial orthophoto, empty ground, grass, lawn, pavement, asphalt, dirt, "
    "bare soil, no objects"
)
SDXL_NEGATIVE = (
    "tree, trees, forest, canopy, building, house, roof, vehicle, car, truck, "
    "person, people, shadow, text, watermark, painting, blur"
)


class SdxlInpaintFiller:
    """SDXL inpainting. Weights live under outputs/checkpoints/ground_fill/hf."""

    def __init__(
        self,
        hf_id: str | None = None,
        device: torch.device | None = None,
        *,
        steps: int = 24,
        guidance: float = 6.5,
        strength: float = 0.99,
        cache_dir: Path | None = None,
    ) -> None:
        from diffusers import AutoPipelineForInpainting

        self.device = device or _device()
        self.backend = "sdxl"
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.strength = float(strength)
        root = Path(__file__).resolve().parents[2]
        cache = Path(cache_dir) if cache_dir else root / "outputs" / "checkpoints" / "ground_fill" / "hf"
        cache.mkdir(parents=True, exist_ok=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        kwargs = dict(torch_dtype=dtype, cache_dir=str(cache))
        try:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                hf_id or SDXL_INPAINT_ID, variant="fp16", **kwargs
            )
        except Exception:
            self.pipe = AutoPipelineForInpainting.from_pretrained(hf_id or SDXL_INPAINT_ID, **kwargs)
        self.pipe.to(self.device)
        if self.device.type == "cuda":
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(1)

    def fill(self, rgb: np.ndarray, mask: np.ndarray, source_ban: np.ndarray | None = None) -> np.ndarray:
        if not mask.any():
            return rgb
        h, w = rgb.shape[:2]
        side = 1024
        scale = min(side / max(h, 1), side / max(w, 1))
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        m = cv2.resize((mask > 0).astype(np.uint8) * 255, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        mc = np.zeros((side, side), dtype=np.uint8)
        canvas[:nh, :nw] = img
        mc[:nh, :nw] = m
        with torch.inference_mode():
            out = self.pipe(
                prompt=SDXL_PROMPT,
                negative_prompt=SDXL_NEGATIVE,
                image=Image.fromarray(canvas),
                mask_image=Image.fromarray(mc),
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                strength=self.strength,
                generator=self._gen,
            ).images[0]
        arr = np.asarray(out)[:nh, :nw]
        if (nh, nw) != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.where((mask > 0)[..., None], arr, rgb)


class HybridFiller:
    """LaMa on small holes, SDXL on larger crops (the ones that smear)."""

    def __init__(self, lama: LamaFiller, sdxl: SdxlInpaintFiller, *, min_px: int = 512) -> None:
        self.lama = lama
        self.sdxl = sdxl
        self.min_px = int(min_px)
        self.backend = f"hybrid_sdxl{self.min_px}"

    def fill(self, rgb: np.ndarray, mask: np.ndarray, source_ban: np.ndarray | None = None) -> np.ndarray:
        if not mask.any():
            return rgb
        h, w = mask.shape[:2]
        if max(h, w) < self.min_px:
            return self.lama.fill(rgb, mask)
        return self.sdxl.fill(rgb, mask)


class PatchMatchFiller:
    """Barnes SIGGRAPH 2009 PatchMatch (PyPatchMatch). Copies nearby ground patches.

    Other remaining objects in the crop are banned as sources so cars/roofs
    are not cloned. Large holes or crops with too little visible ground fall
    back to LaMa instead of running unconstrained PatchMatch.
    """

    guard_sources = True

    def __init__(
        self,
        lama: LamaFiller | None = None,
        *,
        patch_size: int = 7,
        max_px: int = 320,
        min_source: float = 0.18,
    ) -> None:
        from patchmatch import patch_match

        if not getattr(patch_match, "patchmatch_available", False):
            raise RuntimeError("PyPatchMatch failed to load")
        self.lama = lama or LamaFiller()
        self.tiled_fallback = self.lama
        self.patch_size = max(3, int(patch_size) | 1)
        self.max_px = int(max_px)
        self.min_source = float(min_source)
        self.backend = f"patchmatch{self.patch_size}"
        self._pm = patch_match
        self._pm.set_random_seed(1212)

    def fill(self, rgb: np.ndarray, mask: np.ndarray, source_ban: np.ndarray | None = None) -> np.ndarray:
        if not mask.any():
            return rgb
        h, w = rgb.shape[:2]
        hole = mask > 0
        ban = source_ban > 0 if source_ban is not None else np.zeros((h, w), dtype=bool)
        source_ok = (~hole) & (~ban)
        if max(h, w) > self.max_px or float(source_ok.mean()) < self.min_source:
            return self.lama.fill(rgb, hole.astype(np.uint8))
        img = np.ascontiguousarray(rgb)
        hole_u8 = np.ascontiguousarray((hole.astype(np.uint8)) * 255)
        try:
            if ban.any():
                gmask = np.ascontiguousarray((ban.astype(np.uint8)) * 255)
                out = self._pm.inpaint(img, hole_u8, global_mask=gmask, patch_size=self.patch_size)
            else:
                out = self._pm.inpaint(img, hole_u8, patch_size=self.patch_size)
        except Exception:
            return self.lama.fill(rgb, hole.astype(np.uint8))
        filled = out[..., :3] if out.ndim == 3 else out
        return np.where(hole[..., None], filled, rgb)


class MatFiller:
    """MAT (Places-512) via IOPaint's MAT class. Erase model; pads to 512."""

    def __init__(self, device: torch.device | None = None) -> None:
        import sys
        import types

        import iopaint
        from iopaint.schema import InpaintRequest

        model_dir = Path(iopaint.__file__).resolve().parent / "model"
        if "iopaint.model" not in sys.modules or not getattr(sys.modules["iopaint.model"], "__path__", None):
            pkg = types.ModuleType("iopaint.model")
            pkg.__path__ = [str(model_dir)]
            pkg.__package__ = "iopaint.model"
            sys.modules["iopaint.model"] = pkg
        from iopaint.model.mat import MAT

        self.device = device or _device()
        self.backend = "mat"
        self._model = MAT(self.device)
        self._cfg = InpaintRequest()

    def fill(self, rgb: np.ndarray, mask: np.ndarray, source_ban: np.ndarray | None = None) -> np.ndarray:
        if not mask.any():
            return rgb
        m = (mask > 0).astype(np.uint8) * 255
        bgr = self._model(rgb, m, self._cfg)
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.where((mask > 0)[..., None], out, rgb)


GROUND_CLASS_IDS = (0, 1, 2, 3, 5, 6)


def nearest_ground_class(classes: np.ndarray, source: np.ndarray) -> np.ndarray:
    """For every pixel, land-cover class of the nearest unmasked ground pixel."""
    h, w = classes.shape
    best = np.full((h, w), 1e9, dtype=np.float32)
    target = np.zeros((h, w), dtype=np.uint8)
    src = source.astype(bool)
    for cid in GROUND_CLASS_IDS:
        valid = (src & (classes == cid)).astype(np.uint8)
        if not valid.any():
            continue
        dist = cv2.distanceTransform(1 - valid, cv2.DIST_L2, 5)
        better = dist < best
        target[better] = cid
        best[better] = dist[better]
    return target


def landcover_fill(
    rgb: np.ndarray,
    mask: np.ndarray,
    classes: np.ndarray,
    *,
    patch: int = 48,
    stride: int | None = None,
) -> np.ndarray:
    """Fill holes with real pixels of the nearest ground class (grass/pavement/…).

    Does not search object pixels. Uses overlapping class-matched patches.
    """
    hole = mask > 0
    if not hole.any():
        return rgb
    h, w = hole.shape
    source = ~hole
    target = nearest_ground_class(classes, source)
    step = max(4, patch // 6)
    coords: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cid in GROUND_CLASS_IDS:
        ys, xs = np.where(source & (classes == cid))
        if ys.size:
            coords[cid] = (ys[::step], xs[::step])
    if not coords:
        ys, xs = np.where(source)
        coords[1] = (ys[::step], xs[::step]) if ys.size else (ys, xs)

    rng = np.random.default_rng(1)
    stride = int(stride or max(8, patch // 2))
    acc = np.zeros((h, w, 3), dtype=np.float32)
    wt = np.zeros((h, w), dtype=np.float32)
    hann_full = _hann2d(patch, patch)
    ys_h, xs_h = np.where(hole)
    r0, r1 = int(ys_h.min()), int(ys_h.max()) + 1
    c0, c1 = int(xs_h.min()), int(xs_h.max()) + 1
    half = patch // 2
    for y in range(r0, r1, stride):
        for x in range(c0, c1, stride):
            y1 = min(y + patch, h)
            x1 = min(x + patch, w)
            sl = hole[y:y1, x:x1]
            if not sl.any():
                continue
            ph, pw = sl.shape
            maj = int(np.bincount(target[y:y1, x:x1][sl].ravel(), minlength=8).argmax())
            pool = coords.get(maj) or next(iter(coords.values()))
            pys, pxs = pool
            if pys.size == 0:
                continue
            got = None
            for _ in range(12):
                i = int(rng.integers(0, pys.size))
                cy, cx = int(pys[i]), int(pxs[i])
                sy, sx = cy - half, cx - half
                if sy < 0 or sx < 0 or sy + ph > h or sx + pw > w:
                    continue
                if not source[sy : sy + ph, sx : sx + pw].all():
                    continue
                got = rgb[sy : sy + ph, sx : sx + pw]
                break
            if got is None:
                i = int(rng.integers(0, pys.size))
                cy, cx = int(pys[i]), int(pxs[i])
                got = np.broadcast_to(rgb[cy, cx], (ph, pw, 3)).copy()
            ww = hann_full[:ph, :pw]
            acc[y:y1, x:x1] += got.astype(np.float32) * ww[..., None]
            wt[y:y1, x:x1] += ww
    out = rgb.copy()
    use = hole & (wt > 1e-6)
    if use.any():
        out[use] = np.clip(acc[use] / wt[use][:, None], 0, 255).astype(np.uint8)
    leftover = hole & ~use
    if leftover.any():
        for cid, (pys, pxs) in coords.items():
            sel = leftover & (target == cid)
            if not sel.any() or pys.size == 0:
                continue
            n = int(sel.sum())
            pick = rng.integers(0, pys.size, size=n)
            out[sel] = rgb[pys[pick], pxs[pick]]
    return out


def make_filler(inp: dict[str, Any] | None = None, device: torch.device | None = None):
    """Build the inpaint backend from the ground_fill.yaml inpaint block."""
    inp = inp or {}
    device = device or _device()
    backend = str(inp.get("backend", "lama")).lower()
    lama = LamaFiller(device=device)
    if backend in ("lama", "opencv"):
        return lama
    if backend in ("patchmatch", "pm"):
        return PatchMatchFiller(
            lama,
            patch_size=int(inp.get("patch_size", 7)),
            max_px=int(inp.get("patchmatch_max_px", 320)),
            min_source=float(inp.get("patchmatch_min_source", 0.18)),
        )
    if backend == "mat":
        return MatFiller(device=device)
    if backend in ("stability", "stability_erase", "erase"):
        from tree_seg.stability_erase import make_stability_filler

        return make_stability_filler(inp)
    sdxl = SdxlInpaintFiller(
        hf_id=inp.get("sdxl_id") or SDXL_INPAINT_ID,
        device=device,
        steps=int(inp.get("sdxl_steps", 24)),
        guidance=float(inp.get("sdxl_guidance", 6.5)),
        strength=float(inp.get("sdxl_strength", 0.99)),
    )
    if backend == "sdxl":
        return sdxl
    return HybridFiller(lama, sdxl, min_px=int(inp.get("sdxl_min_px", 512)))


def _pad_to_modulo(arr: np.ndarray, mod: int) -> np.ndarray:
    """Pad CxHxW to H,W multiples of mod."""
    _, h, w = arr.shape
    ph = 0 if h % mod == 0 else mod - (h % mod)
    pw = 0 if w % mod == 0 else mod - (w % mod)
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="symmetric")


def _predict_tile_classes(processor, model, rgb: np.ndarray, device) -> np.ndarray:
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    h, w = rgb.shape[:2]
    pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]
    return pred.detach().cpu().numpy().astype(np.uint8)


def segment_landcover(
    rgb: np.ndarray,
    processor,
    model,
    object_ids: set[int],
    device,
    *,
    tile_size: int = 512,
    overlap: float = 0.125,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (object mask, class map). Class ids follow OEM8."""
    h, w = rgb.shape[:2]
    obj_votes = np.zeros((h, w), dtype=np.uint16)
    counts = np.zeros((h, w), dtype=np.uint16)
    classes = np.zeros((h, w), dtype=np.uint8)
    specs = list(iter_tile_specs(h, w, tile_size=tile_size, overlap=overlap))
    for spec in tqdm(specs, desc="landcover", leave=False):
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        patch = rgb[spec.row_off : spec.row_off + spec.height, spec.col_off : spec.col_off + spec.width]
        tile[: spec.height, : spec.width] = patch
        if float(patch.mean()) < 2:
            continue
        cls = _predict_tile_classes(processor, model, tile, device)
        valid = cls[: spec.height, : spec.width]
        obj = np.isin(valid, list(object_ids)).astype(np.uint16)
        r0, c0 = spec.row_off, spec.col_off
        rh, cw = spec.height, spec.width
        obj_votes[r0 : r0 + rh, c0 : c0 + cw] += obj
        counts[r0 : r0 + rh, c0 : c0 + cw] += 1
        classes[r0 : r0 + rh, c0 : c0 + cw] = np.clip(valid, 0, 7)
    counts = np.maximum(counts, 1)
    obj_mask = (obj_votes * 2 >= counts).astype(np.uint8)
    return obj_mask, classes


def segment_objects(
    rgb: np.ndarray,
    processor,
    model,
    object_ids: set[int],
    device,
    *,
    tile_size: int = 512,
    overlap: float = 0.125,
) -> np.ndarray:
    obj, _ = segment_landcover(
        rgb, processor, model, object_ids, device, tile_size=tile_size, overlap=overlap
    )
    return obj


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    k = 2 * int(dilate_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return cv2.dilate(closed, kernel)


def _luma(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def grow_attached_shadows(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    max_px: int = 96,
    luma_ratio: float = 0.55,
) -> np.ndarray:
    """Expand object mask into connected dark pixels (building/tree shadows).

    Uses a local max-luminance reference so sunlit dark asphalt is less likely
    to be eaten than attached shadows.
    """
    grow = (mask > 0).astype(np.uint8)
    if max_px <= 0 or not grow.any():
        return grow
    y = _luma(rgb)
    win = 2 * min(int(max_px), 32) + 1
    local_max = cv2.dilate(y, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win)))
    dark = (y < float(luma_ratio) * np.maximum(local_max, 1.0)) & (y >= 3.0)
    limit = np.maximum(grow, dark.astype(np.uint8))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(int(max_px)):
        nxt = cv2.dilate(grow, k3) & limit
        if not np.any(nxt != grow):
            break
        grow = nxt
    return grow


def lift_context_shadows(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    luma_ratio: float = 0.55,
    win: int = 31,
) -> np.ndarray:
    """Brighten remaining *known* shadows so LaMa does not copy them into holes."""
    y = _luma(rgb)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(win), int(win)))
    local_y = cv2.dilate(y, k)
    known = mask == 0
    dark = known & (y < float(luma_ratio) * np.maximum(local_y, 1.0)) & (y >= 3.0)
    if not dark.any():
        return rgb
    scale = np.ones_like(y)
    scale[dark] = local_y[dark] / np.maximum(y[dark], 1.0)
    scale = np.clip(scale, 1.0, 3.0)
    lifted = np.clip(rgb.astype(np.float32) * scale[..., None], 0, 255).astype(np.uint8)
    return np.where(dark[..., None], lifted, rgb)


def relight_inpainted(
    bare: np.ndarray,
    mask: np.ndarray,
    *,
    win: int = 64,
    min_ratio: float = 0.72,
) -> np.ndarray:
    """If a filled pixel is still much darker than nearby known ground, lift it."""
    m = mask > 0
    known = ~m
    if not m.any() or not known.any():
        return bare
    y = _luma(bare)
    known_f = known.astype(np.float32)
    k = 2 * (int(win) // 2) + 1
    kernel = np.ones((k, k), np.float32)
    local = cv2.filter2D(y * known_f, -1, kernel, borderType=cv2.BORDER_REFLECT)
    count = cv2.filter2D(known_f, -1, kernel, borderType=cv2.BORDER_REFLECT)
    gmed = float(np.median(y[known]))
    local = np.where(count > 8.0, local / np.maximum(count, 1.0), gmed)
    too_dark = m & (y < float(min_ratio) * local) & (y > 1.0)
    if not too_dark.any():
        return bare
    scale = np.ones_like(y)
    scale[too_dark] = (0.92 * local[too_dark]) / np.maximum(y[too_dark], 1.0)
    scale = np.clip(scale, 1.0, 3.5)
    lifted = np.clip(bare.astype(np.float32) * scale[..., None], 0, 255).astype(np.uint8)
    return np.where(m[..., None], lifted, bare)


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(max(h, 2)).astype(np.float32)
    wx = np.hanning(max(w, 2)).astype(np.float32)
    if h == 1:
        wy[:] = 1.0
    if w == 1:
        wx[:] = 1.0
    return wy[:, None] * wx[None, :]


def blend_local_grain(
    filled: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    *,
    cell: int = 128,
    sigma: float = 1.25,
    strength: float = 0.8,
) -> np.ndarray:
    """Add surrounding high-frequency grain without cloning objects or patterns."""
    m = mask > 0
    if not m.any():
        return filled
    grain = _shuffled_highpass(original, m, cell=cell, sigma=sigma)
    out = filled.astype(np.float32)
    out[m] = np.clip(out[m] + float(strength) * grain[m], 0, 255)
    return out.astype(np.uint8)


def _shuffled_highpass(
    original: np.ndarray,
    mask: np.ndarray,
    *,
    cell: int,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """High-pass residual from nearby unmasked pixels, shuffled so objects are not cloned."""
    orig_f = original.astype(np.float32)
    hp = orig_f - cv2.GaussianBlur(original, ksize=(0, 0), sigmaX=float(sigma)).astype(np.float32)
    grain = np.zeros_like(hp)
    h, w = mask.shape
    rng = rng or np.random.default_rng(1)
    known = ~mask
    c = max(8, int(cell))
    for y0 in range(0, h, c):
        y1 = min(y0 + c, h)
        for x0 in range(0, w, c):
            x1 = min(x0 + c, w)
            hole = mask[y0:y1, x0:x1]
            if not hole.any():
                continue
            yy0, xx0 = max(y0 - c, 0), max(x0 - c, 0)
            yy1, xx1 = min(y1 + c, h), min(x1 + c, w)
            kn = known[yy0:yy1, xx0:xx1]
            if not kn.any():
                continue
            kys, kxs = np.where(kn)
            hys, hxs = np.where(hole)
            pick = rng.integers(0, len(kys), size=len(hys))
            grain[y0 + hys, x0 + hxs] = hp[yy0 + kys[pick], xx0 + kxs[pick]]
    return grain


def _box_sum(arr: np.ndarray, ksize: int) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    return cv2.boxFilter(arr, ddepth=-1, ksize=(k, k), normalize=False, borderType=cv2.BORDER_REFLECT)


def _local_mean_std(channel: np.ndarray, weight: np.ndarray, ksize: int, min_count: float = 12.0):
    wsum = _box_sum(weight, ksize)
    csum = _box_sum(channel * weight, ksize)
    c2sum = _box_sum(channel * channel * weight, ksize)
    valid = wsum >= float(min_count)
    mean = np.where(valid, csum / np.maximum(wsum, 1.0), np.nan)
    var = np.where(valid, c2sum / np.maximum(wsum, 1.0) - mean * mean, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std, valid


def _coalesce_stats(small, large, global_value: float) -> np.ndarray:
    out = np.where(np.isfinite(small), small, large)
    return np.where(np.isfinite(out), out, float(global_value))


def refine_inpaint(
    filled: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    *,
    chroma_win: int = 97,
    chroma_win_large: int = 257,
    chroma_strength: float = 1.0,
    grain_scales: tuple[float, ...] = (1.25, 4.0, 9.0),
    grain_strengths: tuple[float, ...] = (0.42, 0.40, 0.28),
    grain_cells: tuple[int, ...] = (128, 192, 256),
    sharpen: float = 0.28,
    sharpen_edge_boost: float = 0.55,
) -> np.ndarray:
    """Match nearby ground chroma, add multi-scale residual grain, then edge-aware sharpen.

    Touches masked pixels only. Keeps LaMa luminance; grain is luma-only so it
    does not inject magenta/green speckle. Does not clone patches.
    """
    m = mask > 0
    if not m.any():
        return filled

    known = ~m
    known_f = known.astype(np.float32)
    hole_f = m.astype(np.float32)
    lab_f = cv2.cvtColor(filled, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_o = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    out_lab = lab_f.copy()

    if known.any() and float(chroma_strength) > 0:
        for ch in (1, 2):
            fill_ch = lab_f[..., ch]
            orig_ch = lab_o[..., ch]
            mu_o_s, _, _ = _local_mean_std(orig_ch, known_f, chroma_win)
            mu_o_l, _, _ = _local_mean_std(orig_ch, known_f, chroma_win_large)
            mu_f_s, _, _ = _local_mean_std(fill_ch, hole_f, chroma_win)
            mu_f_l, _, _ = _local_mean_std(fill_ch, hole_f, chroma_win_large)
            g_mu_o = float(orig_ch[known].mean())
            g_mu_f = float(fill_ch[m].mean())
            mu_o = _coalesce_stats(mu_o_s, mu_o_l, g_mu_o)
            mu_f = _coalesce_stats(mu_f_s, mu_f_l, g_mu_f)
            out_lab[..., ch] = fill_ch + float(chroma_strength) * (mu_o - mu_f)

    luma_orig = np.stack([_luma(original).astype(np.uint8)] * 3, axis=-1)
    rng = np.random.default_rng(1)
    grain_l = np.zeros(m.shape, dtype=np.float32)
    for sigma, strength, cell in zip(grain_scales, grain_strengths, grain_cells):
        if strength <= 0:
            continue
        g = _shuffled_highpass(luma_orig, m, cell=cell, sigma=sigma, rng=rng)
        grain_l += float(strength) * g[..., 0]
    out_lab[..., 0] = np.clip(out_lab[..., 0] + grain_l, 0, 255)

    if float(sharpen) > 0:
        L = out_lab[..., 0]
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        p95 = float(np.percentile(grad[m], 95)) if int(m.sum()) else 1.0
        edge_w = np.clip(grad / max(p95, 1.0), 0.0, 1.0)
        blur = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=1.05)
        amount = float(sharpen) * (0.42 + float(sharpen_edge_boost) * edge_w)
        out_lab[..., 0] = np.clip(L + amount * (L - blur), 0, 255)

    rgb = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out = filled.copy()
    out[m] = rgb[m]
    return out


def object_label_map(mask: np.ndarray, erode_px: int = 12) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """Split a dilated object mask into per-object labels.

    Erodes to break dilation bridges, then dilates each seed back into the
    original hole. Returns (label_map, [(label, r0, c0, r1, c1), ...]).
    Unassigned leftover pixels stay 0 for a final fill pass.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    if not m.any():
        return np.zeros(m.shape, dtype=np.int32), []
    k = max(0, int(erode_px))
    if k <= 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        boxes = []
        for lab in range(1, n):
            x, y, bw, bh, _area = stats[lab]
            boxes.append((lab, y, x, y + bh, x + bw))
        return labels.astype(np.int32), boxes
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    seeds = cv2.erode(m, ker)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    if n <= 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        boxes = []
        for lab in range(1, n):
            x, y, bw, bh, _area = stats[lab]
            boxes.append((lab, y, x, y + bh, x + bw))
        return labels.astype(np.int32), boxes
    out = np.zeros(m.shape, dtype=np.int32)
    pad = k + 2
    boxes: list[tuple[int, int, int, int, int]] = []
    for lab in range(1, n):
        x, y, bw, bh, _area = stats[lab]
        r0, c0 = max(y - pad, 0), max(x - pad, 0)
        r1, c1 = min(y + bh + pad, h), min(x + bw + pad, w)
        crop = (labels[r0:r1, c0:c1] == lab).astype(np.uint8)
        crop = cv2.dilate(crop, ker)
        dest = out[r0:r1, c0:c1]
        hit = (crop > 0) & (m[r0:r1, c0:c1] > 0) & (dest == 0)
        if not hit.any():
            continue
        dest[hit] = lab
        out[r0:r1, c0:c1] = dest
        ys, xs = np.where(hit)
        boxes.append((lab, r0 + int(ys.min()), c0 + int(xs.min()), r0 + int(ys.max()) + 1, c0 + int(xs.max()) + 1))
    return out, boxes


def object_holes_with_context_gaps(
    mask: np.ndarray,
    *,
    min_distance_px: int = 10,
    gap_px: int = 2,
    open_px: int = 1,
) -> np.ndarray:
    """Keep object pixels as holes but force a thin unmasked gap between instances.

    Uses distance-transform peaks so touching tree crowns / roofs become separate
    holes. Ground, roads, and yards stay visible as context for a single Erase call.
    """
    m = (mask > 0).astype(np.uint8)
    if open_px > 0:
        k = 2 * int(open_px) + 1
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if not m.any():
        return m
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    min_d = max(3, int(min_distance_px))
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    peaks = peak_local_max(dist, min_distance=min_d, labels=m.astype(bool), exclude_border=False)
    markers = np.zeros(m.shape, dtype=np.int32)
    for i, (rr, cc) in enumerate(peaks, start=1):
        markers[int(rr), int(cc)] = i
    if int(markers.max()) < 2:
        n, labels, _st, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
        inst = labels.astype(np.int32)
    else:
        inst = watershed(-dist, markers, mask=m.astype(bool)).astype(np.int32)
    gap = max(0, int(gap_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1)) if gap else None
    out = np.zeros_like(m)
    nlab = int(inst.max())
    for lab in range(1, nlab + 1):
        piece = (inst == lab).astype(np.uint8)
        if not piece.any():
            continue
        shrunk = cv2.erode(piece, ker) if ker is not None else piece
        if shrunk.any():
            out = np.maximum(out, shrunk)
        else:
            out = np.maximum(out, piece)
    return out


def expand_instances_keep_gaps(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Grow each hole, but leave pixels claimed by two instances unmasked."""
    m = (mask > 0).astype(np.uint8)
    if int(dilate_px) <= 0 or not m.any():
        return m
    n, labels, _st, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1))
    votes = np.zeros(m.shape, dtype=np.uint16)
    for lab in range(1, n):
        votes += cv2.dilate((labels == lab).astype(np.uint8), ker)
    return (votes == 1).astype(np.uint8)


def union_dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Dilate the combined hole (instances may merge)."""
    m = (mask > 0).astype(np.uint8)
    if int(dilate_px) <= 0:
        return m
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1))
    return cv2.dilate(m, ker)


def _fill_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    filler,
    *,
    tile_size: int,
    overlap: float,
    source_ban: np.ndarray | None = None,
) -> np.ndarray:
    if not np.any(mask):
        return rgb
    h, w = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)
    if h <= tile_size and w <= tile_size:
        return filler.fill(rgb, m, source_ban=source_ban)
    fallback = getattr(filler, "tiled_fallback", filler)
    return inpaint_tiled(rgb, m, fallback, tile_size=tile_size, overlap=overlap)


def inpaint_per_object(
    rgb: np.ndarray,
    mask: np.ndarray,
    filler,
    *,
    pad_px: int = 96,
    tile_size: int = 1024,
    overlap: float = 0.125,
    split_erode_px: int = 12,
) -> np.ndarray:
    """Fill each object hole with a ring of surrounding pixels still visible.

    Small components first so later (larger) holes see already-filled ground.
    Only the current object's pixels are masked in each fill call. Remaining
    objects in the crop can be banned as PatchMatch sources.
    """
    m = mask > 0
    if not m.any():
        return rgb
    labels, boxes = object_label_map(mask, erode_px=split_erode_px)
    current = rgb.copy()
    h, w = m.shape
    pad = max(0, int(pad_px))
    already = np.zeros((h, w), dtype=bool)
    guard = bool(getattr(filler, "guard_sources", False))
    boxes_sorted = sorted(
        boxes,
        key=lambda b: int((labels[b[1] : b[3], b[2] : b[4]] == b[0]).sum()),
    )
    print(f"per-object units={len(boxes_sorted)} leftover={(m & (labels == 0)).mean():.4f}", flush=True)
    for lab, br0, bc0, br1, bc1 in tqdm(boxes_sorted, desc="objects", leave=False):
        r0, r1 = max(br0 - pad, 0), min(br1 + pad, h)
        c0, c1 = max(bc0 - pad, 0), min(bc1 + pad, w)
        crop = current[r0:r1, c0:c1]
        hole = labels[r0:r1, c0:c1] == lab
        if not hole.any():
            continue
        source_ban = None
        if guard:
            blocked = m[r0:r1, c0:c1] & (~already[r0:r1, c0:c1]) & (~hole)
            source_ban = blocked.astype(np.uint8)
        filled = _fill_crop(
            crop, hole, filler, tile_size=tile_size, overlap=overlap, source_ban=source_ban
        )
        current[r0:r1, c0:c1] = np.where(hole[..., None], filled, crop)
        already[r0:r1, c0:c1] |= hole
    leftover = m & (labels == 0)
    if leftover.any():
        fallback = getattr(filler, "tiled_fallback", filler)
        filled_left = inpaint_tiled(
            current, leftover.astype(np.uint8), fallback, tile_size=tile_size, overlap=overlap
        )
        current = np.where(leftover[..., None], filled_left, current)
    return current


def inpaint_tiled(
    rgb: np.ndarray,
    mask: np.ndarray,
    filler,
    *,
    tile_size: int = 1024,
    overlap: float = 0.125,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = rgb.copy()
    acc = np.zeros((h, w, 3), dtype=np.float32)
    wt = np.zeros((h, w), dtype=np.float32)
    specs = list(iter_tile_specs(h, w, tile_size=tile_size, overlap=overlap))
    for spec in tqdm(specs, desc="inpaint", leave=False):
        r0, c0, hh, ww = spec.row_off, spec.col_off, spec.height, spec.width
        m = mask[r0 : r0 + hh, c0 : c0 + ww]
        if not m.any():
            continue
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        mt = np.zeros((tile_size, tile_size), dtype=np.uint8)
        tile[:hh, :ww] = rgb[r0 : r0 + hh, c0 : c0 + ww]
        mt[:hh, :ww] = m
        filled = filler.fill(tile, mt)[:hh, :ww]
        wwgt = _hann2d(hh, ww)
        acc[r0 : r0 + hh, c0 : c0 + ww] += filled.astype(np.float32) * wwgt[..., None]
        wt[r0 : r0 + hh, c0 : c0 + ww] += wwgt
    use = wt > 1e-6
    if use.any():
        blended = np.zeros_like(rgb)
        denom = np.maximum(wt[use], 1e-6)[:, None]
        blended[use] = np.clip(acc[use] / denom, 0, 255).astype(np.uint8)
        out = np.where(mask[..., None].astype(bool), blended, rgb)
    return out


def ground_fill_paths(output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Return (objects tif, bare tif) under output_dir/objects and output_dir/bare."""
    output_dir = Path(output_dir)
    objects_dir = output_dir / "objects"
    bare_dir = output_dir / "bare"
    objects_dir.mkdir(parents=True, exist_ok=True)
    bare_dir.mkdir(parents=True, exist_ok=True)
    return objects_dir / f"{stem}_objects.tif", bare_dir / f"{stem}_bare.tif"


def fill_geotiff(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    processor,
    model,
    object_ids: set[int],
    device,
    filler=None,
    tile_size: int = 512,
    overlap: float = 0.125,
    dilate_px: int = 12,
    inpaint_tile: int = 1024,
    shadow_grow_px: int = 96,
    shadow_luma_ratio: float = 0.55,
    lift_context: bool = False,
    relight: bool = False,
    reuse_mask: bool = False,
    skip_existing: bool = True,
    per_object: bool = True,
    object_pad_px: int = 96,
    split_erode_px: int = 12,
) -> dict[str, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    stem = input_path.stem
    mask_path, bare_path = ground_fill_paths(output_dir, stem)
    if skip_existing and mask_path.is_file() and bare_path.is_file():
        return {"mask": mask_path, "bare": bare_path, "skipped": True}

    with open_rgb_geotiff(input_path) as ds:
        rgb = read_rgb(ds)
        transform = ds.transform
        crs = ds.crs

    if float(rgb.mean()) < 2:
        write_geotiff(mask_path, np.zeros(rgb.shape[:2], dtype=np.uint8), transform, crs, nodata=0, dtype="uint8")
        write_geotiff(bare_path, np.transpose(rgb, (2, 0, 1)), transform, crs, dtype="uint8")
        return {"mask": mask_path, "bare": bare_path, "empty": True}

    reused = False
    if reuse_mask and mask_path.is_file():
        import rasterio

        with rasterio.open(mask_path) as mds:
            obj = mds.read(1)
        reused = True
    else:
        obj = segment_objects(
            rgb, processor, model, object_ids, device, tile_size=tile_size, overlap=overlap
        )
        obj = dilate_mask(obj, dilate_px)
        obj = grow_attached_shadows(rgb, obj, max_px=shadow_grow_px, luma_ratio=shadow_luma_ratio)

    if filler is None:
        filler = LamaFiller(device=device)
    context = lift_context_shadows(rgb, obj, luma_ratio=shadow_luma_ratio) if lift_context else rgb
    if per_object:
        bare = inpaint_per_object(
            context,
            obj,
            filler,
            pad_px=object_pad_px,
            tile_size=inpaint_tile,
            overlap=overlap,
            split_erode_px=split_erode_px,
        )
    else:
        bare = inpaint_tiled(context, obj, filler, tile_size=inpaint_tile, overlap=overlap)
    bare = np.where(obj[..., None].astype(bool), bare, rgb)
    bare = blend_local_grain(bare, rgb, obj)
    if relight:
        bare = relight_inpainted(bare, obj)

    write_geotiff(mask_path, obj, transform, crs, nodata=0, dtype="uint8")
    write_geotiff(bare_path, np.transpose(bare, (2, 0, 1)), transform, crs, dtype="uint8")
    return {
        "mask": mask_path,
        "bare": bare_path,
        "object_frac": float(obj.mean()),
        "backend": filler.backend if filler else "lama",
        "reused_mask": reused,
        "per_object": per_object,
    }


def fill_from_config(input_path: str | Path, output_dir: str | Path, cfg: dict[str, Any], **kwargs):
    m = cfg.get("model", {})
    inp = cfg.get("inpaint", {})
    device = _device()
    processor, model, id2label, object_ids, device = load_landcover(m.get("hf_id"), device)
    filler = make_filler(inp, device)
    extra = {
        "processor": processor,
        "model": model,
        "object_ids": object_ids,
        "device": device,
        "filler": filler,
        "tile_size": int(m.get("tile_size", 512)),
        "overlap": float(m.get("overlap", 0.125)),
        "dilate_px": int(inp.get("dilate_px", 12)),
        "inpaint_tile": int(inp.get("tile_size", 1024)),
        "shadow_grow_px": int(inp.get("shadow_grow_px", 96)),
        "shadow_luma_ratio": float(inp.get("shadow_luma_ratio", 0.55)),
        "lift_context": bool(inp.get("lift_context", True)),
        "relight": bool(inp.get("relight", False)),
        "reuse_mask": bool(inp.get("reuse_mask", False)),
        "per_object": bool(inp.get("per_object", True)),
        "object_pad_px": int(inp.get("object_pad_px", 96)),
        "split_erode_px": int(inp.get("split_erode_px", 12)),
    }
    extra.update(kwargs)
    result = fill_geotiff(input_path, output_dir, **extra)
    result["id2label"] = id2label
    result["object_ids"] = sorted(object_ids)
    return result
