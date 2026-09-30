"""
Stage 1 (PRIMARY as of the backbone swap) — Depth Anything relative
depth inference.

Loads a pretrained Depth Anything checkpoint via Hugging Face
`transformers` (the `depth-estimation` pipeline, backed by
`DepthAnythingForDepthEstimation`) and runs monocular depth estimation
on an RGB image.

Two size variants are configured in app/config.py:
    - "vits" — LiheYoung/depth-anything-small-hf (ViT-S/14 encoder,
      24.8M params). Default for the live web demo / real-time user
      uploads, where inference latency matters.
    - "vitl" — LiheYoung/depth-anything-large-hf (ViT-L/14 encoder,
      335.3M params). For offline benchmark runs / best-accuracy report
      figures. Not used on the live request path unless explicitly
      requested.

Why this replaced MiDaS as Stage 1's default: Depth Anything is trained
on 1.5M labeled + 62M unlabeled images (vs MiDaS's ~2M labeled) and
outperforms MiDaS across every published benchmark cited in the project
spec (e.g. DDAD AbsRel improves 0.251 -> 0.230, KITTI delta1 improves
0.850 -> 0.947). MiDaS is NOT removed from the codebase — see
app/services/midas_model.py, kept for the Stage 2e confidence ensemble.

Output is a *relative* inverse-depth map (same convention as MiDaS —
larger value = closer to camera), with no metric scale attached.
Swapping the backbone does not by itself change Stage 1's
"relative_uncalibrated" status; metric scale still comes from
Stage 2a/2b/2d once those land.

Each variant's pipeline is loaded once per process (singleton, keyed by
variant) since the Hugging Face model download + pipeline construction
is expensive.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from app.config import settings

# Deployment reality check (see README > Deployment): free-tier hosts
# (Render's free web service, in particular) cap memory at 512MB, which
# a full PyTorch + transformers stack can exceed just from its own
# baseline footprint before any model is even loaded. torch.set_num_threads(1)
# trims some of that (each BLAS/OMP thread pool allocates its own
# buffers) — a real, measurable reduction, not a full fix for a
# genuinely tight memory budget, but free and safe to always apply.
torch.set_num_threads(1)

HF_MODEL_IDS = {
    "vits": "LiheYoung/depth-anything-small-hf",
    "vitl": "LiheYoung/depth-anything-large-hf",
}

_pipelines: dict = {}
_lock = threading.Lock()


@dataclass
class DepthResult:
    relative_depth: np.ndarray  # (H, W) float32, raw model output (inverse depth, arbitrary units)
    normalized: np.ndarray  # (H, W) float32 in [0, 1] — display + mesh-generation friendly
    model_type: str
    device: str


def _get_pipeline(variant: str):
    if variant not in HF_MODEL_IDS:
        raise ValueError(
            f"Unknown Depth Anything variant '{variant}'. Expected one of {list(HF_MODEL_IDS)}."
        )

    if variant in _pipelines:
        return _pipelines[variant]

    with _lock:
        if variant in _pipelines:  # re-check inside the lock (race with another thread)
            return _pipelines[variant]

        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        # low_cpu_mem_usage=True avoids from_pretrained's default
        # behavior of allocating the model twice (once with random
        # init, then again while copying in the real weights) — cuts
        # peak RAM during load by roughly the model's own weight size.
        # Matters a lot on a memory-capped host (see module docstring's
        # deployment note); free and behavior-neutral everywhere else.
        pipe = hf_pipeline(
            task="depth-estimation",
            model=HF_MODEL_IDS[variant],
            device=device,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        _pipelines[variant] = pipe
        return pipe


def is_model_loaded(variant: str = "vits") -> bool:
    return variant in _pipelines


def run_inference(rgb: np.ndarray, variant: str | None = None) -> DepthResult:
    """
    rgb: (H, W, 3) uint8 RGB array.
    variant: "vits" (fast, live-demo default) or "vitl" (accuracy,
        offline benchmark runs). Defaults to
        settings.depth_anything_live_variant.

    Returns depth resampled back to the ORIGINAL input resolution so it
    lines up pixel-for-pixel with the source image for texturing.
    """
    variant = variant or settings.depth_anything_live_variant
    pipe = _get_pipeline(variant)

    image = Image.fromarray(rgb)
    # no_grad: the pipeline already runs in eval mode, but this makes it
    # explicit and guarantees no autograd graph gets retained — on a
    # memory-capped host, an accidentally-kept graph is pure waste.
    with torch.no_grad():
        output = pipe(image)

        # Use the raw "predicted_depth" tensor (model-native resolution,
        # full float precision) rather than the pipeline's convenience
        # "depth" PIL image, which is already quantized to uint8 — we
        # want our own float32 normalization for the mesh/height_grid
        # math, same as midas_model.py does.
        depth_tensor = output["predicted_depth"]
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.squeeze(0)

        original_h, original_w = rgb.shape[:2]
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0),
            size=(original_h, original_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = depth_tensor.cpu().numpy().astype(np.float32)

    # Drop the tensor references and reclaim memory immediately rather
    # than waiting for Python's next GC cycle — each request's tensors
    # are otherwise the kind of thing that accumulates right up to the
    # memory ceiling under back-to-back requests.
    del output, depth_tensor
    gc.collect()

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        normalized = np.zeros_like(depth)
    else:
        normalized = (depth - d_min) / (d_max - d_min)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return DepthResult(
        relative_depth=depth,
        normalized=normalized,
        model_type=f"depth-anything-{variant}",
        device=device_str,
    )
