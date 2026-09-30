"""
MiDaS relative depth inference — SECONDARY model, used by Stage 2e.

Historical note: this was Stage 1's primary backbone until the backbone
swap to Depth Anything (see app/services/depth_anything_model.py, which
is now the default for the live /api/depth/estimate path). MiDaS itself
was NOT removed — it's kept specifically for the Stage 2e confidence/
uncertainty ensemble (app/services/confidence.py): MiDaS vs Depth
Anything disagreement is a more meaningful uncertainty signal than
comparing two Depth Anything sizes against each other, since MiDaS was
trained on different data with a different method (two Depth Anything
sizes share training data/method, so their disagreement mostly reflects
model capacity, not real scene ambiguity). Stage 2e is not yet wired in,
so this module is currently unused by the live API — kept ready for
that step.

Loads the pretrained MiDaS backbone via torch.hub (official repo:
github.com/isl-org/MiDaS) and runs monocular depth estimation on an RGB
image. The output is a *relative* inverse-depth map (larger value = the
model thinks that point is closer to the camera) with no metric scale
attached.

Caveat worth stating plainly (also called out in the README): MiDaS was
trained on ground-level/oblique photos, not nadir (straight-down)
satellite/aerial imagery. Treating higher inverse-depth (model's "closer
to camera") as "taller" is a reasonable proxy for top-down shots but is
an approximation — the same caveat applies to Depth Anything, which is
why Stage 2b's LoRA fine-tuning on real aerial RGB+DSM pairs is what's
meant to correct this properly.

The model is loaded once per process (singleton) since torch.hub.load()
plus the first-run weight download is expensive.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

from app.config import settings

_model = None
_transform = None
_device: torch.device | None = None
_lock = threading.Lock()


@dataclass
class DepthResult:
    relative_depth: np.ndarray  # (H, W) float32, raw MiDaS output (inverse depth, arbitrary units)
    normalized: np.ndarray  # (H, W) float32 in [0, 1] — display + mesh-generation friendly
    model_type: str
    device: str


def _load_model() -> None:
    global _model, _transform, _device
    if _model is not None:
        return

    with _lock:
        if _model is not None:  # re-check inside the lock (another thread may have won the race)
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.hub.load("intel-isl/MiDaS", settings.midas_model_type)
        model.to(device)
        model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if settings.midas_model_type in ("DPT_Large", "DPT_Hybrid"):
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform

        _model, _transform, _device = model, transform, device


def is_model_loaded() -> bool:
    return _model is not None


def run_inference(rgb: np.ndarray) -> DepthResult:
    """
    rgb: (H, W, 3) uint8 RGB array (NOT BGR — caller is responsible for
    the conversion if the source used OpenCV).

    Returns depth resampled back to the ORIGINAL input resolution so it
    lines up pixel-for-pixel with the source image for texturing.
    """
    _load_model()
    assert _model is not None and _transform is not None and _device is not None

    original_h, original_w = rgb.shape[:2]
    input_batch = _transform(rgb).to(_device)

    with torch.no_grad():
        prediction = _model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(original_h, original_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy().astype(np.float32)

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        normalized = np.zeros_like(depth)
    else:
        normalized = (depth - d_min) / (d_max - d_min)

    return DepthResult(
        relative_depth=depth,
        normalized=normalized,
        model_type=settings.midas_model_type,
        device=str(_device),
    )
