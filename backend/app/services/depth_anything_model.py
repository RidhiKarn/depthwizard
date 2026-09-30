"""
Stage 1 (PRIMARY as of the backbone swap) — Depth Anything relative
depth inference, served via ONNX Runtime rather than PyTorch.

Why ONNX Runtime instead of the transformers/PyTorch pipeline this
originally used: deployed on Render's free tier (512MB RAM), the
PyTorch version crashed with "Ran out of memory" on the very first
image upload — confirmed via Render's own logs, not a guess. Root
cause, confirmed by direct measurement on this project's dev machine:
merely `import torch` (before loading any model at all) already carries
a large baseline memory footprint. ONNX Runtime runs the identical
trained model with a much smaller footprint, no large tensor-autograd
runtime to load.

The .onnx file this loads (app/ml_models/depth_anything_vits_quantized.onnx)
was produced once, offline, by scripts/export_onnx.py — see that
script's docstring for exactly how, and for the verification that its
output still matches the original PyTorch model (correlation 0.9996 on
a test image; not bit-identical due to int8 quantization, but the
relative depth structure — the only thing this app actually uses, since
output gets min-max normalized regardless — is preserved).

Two consequences of this switch, both intentional:
    - Only the "vits" (ViT-S) variant has been exported/committed —
      it's what the live demo path used by default anyway. "vitl"
      would need running scripts/export_onnx.py --variant vitl once.
    - Preprocessing (resize + ImageNet normalization) is done here in
      plain PIL/numpy instead of transformers' AutoImageProcessor,
      since pulling in transformers just for that would defeat the
      point. Resize is a fixed 518x518 square (see export script's
      EXPORT_RESOLUTION comment for why dynamic sizes aren't safe for
      this architecture) rather than the original aspect-ratio-
      preserving behavior — a minor accuracy trade-off, not a
      correctness bug (output is still resampled back to the original
      image's exact resolution afterward).

torch/transformers are NOT imports of this module at all anymore, on
purpose — see app/services/midas_model.py and
app/services/calibration/lora_calibration.py for where they're still
used (both now import torch lazily, only when actually invoked, so
merely starting the app / routing a request through depth.py doesn't
load PyTorch into memory unless one of those specific opt-in/bonus
features is actually used).
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import settings

HF_MODEL_IDS = {
    "vits": "LiheYoung/depth-anything-small-hf",
    "vitl": "LiheYoung/depth-anything-large-hf",
}

ONNX_MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models"
EXPORT_RESOLUTION = 518  # must match scripts/export_onnx.py

# ImageNet normalization stats — the exact values transformers'
# AutoImageProcessor used for this model (verified against its
# image_mean/image_std attributes before switching away from it).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_sessions: dict = {}
_lock = threading.Lock()


@dataclass
class DepthResult:
    relative_depth: np.ndarray  # (H, W) float32, raw model output (inverse depth, arbitrary units)
    normalized: np.ndarray  # (H, W) float32 in [0, 1] — display + mesh-generation friendly
    model_type: str
    device: str


def _get_session(variant: str):
    if variant not in HF_MODEL_IDS:
        raise ValueError(
            f"Unknown Depth Anything variant '{variant}'. Expected one of {list(HF_MODEL_IDS)}."
        )

    if variant in _sessions:
        return _sessions[variant]

    with _lock:
        if variant in _sessions:  # re-check inside the lock (race with another thread)
            return _sessions[variant]

        import onnxruntime as ort

        model_path = ONNX_MODEL_DIR / f"depth_anything_{variant}_quantized.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run `python scripts/export_onnx.py "
                f"--variant {variant}` to generate it (see that script's docstring)."
            )

        # Single-threaded, arena allocator disabled: this runs on
        # memory-capped free-tier hosts. Thread pools and ONNX Runtime's
        # own memory arena (which pre-reserves and reuses larger memory
        # blocks for speed) both trade RAM for performance we don't need
        # at this request volume. Measured on this project's own dev
        # machine, not a guess: stable per-request RSS for the whole
        # FastAPI process dropped from ~420MB to ~190MB with these three
        # options set, for this exact model.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False

        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        _sessions[variant] = session
        return session


def is_model_loaded(variant: str = "vits") -> bool:
    return variant in _sessions


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    """rgb (H, W, 3) uint8 -> (1, 3, EXPORT_RESOLUTION, EXPORT_RESOLUTION) float32,
    resized + ImageNet-normalized, matching what AutoImageProcessor did
    (verified against its image_mean/image_std before this rewrite)."""
    resized = Image.fromarray(rgb).resize((EXPORT_RESOLUTION, EXPORT_RESOLUTION), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0  # (H, W, 3), [0, 1]
    array = (array - _MEAN) / _STD
    array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(array, axis=0).astype(np.float32)  # add batch dim


def run_inference(rgb: np.ndarray, variant: str | None = None) -> DepthResult:
    """
    rgb: (H, W, 3) uint8 RGB array.
    variant: "vits" (fast, live-demo default) or "vitl" (accuracy,
        offline benchmark runs — needs its own exported .onnx file, see
        module docstring). Defaults to settings.depth_anything_live_variant.

    Returns depth resampled back to the ORIGINAL input resolution so it
    lines up pixel-for-pixel with the source image for texturing.
    """
    variant = variant or settings.depth_anything_live_variant
    session = _get_session(variant)

    pixel_values = _preprocess(rgb)
    (raw_output,) = session.run(None, {"pixel_values": pixel_values})
    depth_small = raw_output[0]  # (EXPORT_RESOLUTION, EXPORT_RESOLUTION)

    original_h, original_w = rgb.shape[:2]
    depth = cv2.resize(depth_small, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    depth = depth.astype(np.float32)

    del raw_output, depth_small, pixel_values
    gc.collect()

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        normalized = np.zeros_like(depth)
    else:
        normalized = (depth - d_min) / (d_max - d_min)

    return DepthResult(
        relative_depth=depth,
        normalized=normalized,
        model_type=f"depth-anything-{variant}",
        device="cpu",
    )
