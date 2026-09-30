"""
Stage 2b — Learned semantic-prior calibration via LoRA fine-tuning.

This is the project's primary technical differentiator: it lets the
model output a plausible absolute height for images that have NO
location metadata at all (PNG/JPG), beyond the official PS's minimum
requirement (relative-only for that case).

STATUS: the inference-loading side (this module) is implemented and
verified — see tests/test_lora_calibration.py, which round-trips a real
(structurally valid but untrained) adapter through save -> load ->
forward pass against the actual `LiheYoung/depth-anything-small-hf`
model. What is NOT done yet is producing a genuinely trained adapter:
that needs registered training datasets (ISPRS/DFC — see
training/README.md) and a GPU this dev environment doesn't have. The
training script itself is in training/train_lora.py, meant to be run
on Colab/Kaggle/any CUDA box; once it produces a checkpoint, drop it at
DEPTHWIZARD_LORA_ADAPTER_DIR (default backend/data/lora_adapter/) and
this module picks it up automatically — no code changes needed.

Recipe (matches training/train_lora.py exactly, verified against the
real model architecture, not guessed):
  - LoRA target_modules=["query", "value"] on the Dinov2 backbone's
    self-attention (confirmed these names exist in the real model and
    collide with nothing outside app.backbone).
  - modules_to_save=["neck", "head"] keeps the depth decoder fully
    trainable and bundles it into the same saved checkpoint.
  - Loss (in the training script) is a masked Huber/SmoothL1 against
    real DSM meters, not scale-invariant log — the whole point of this
    stage is to teach the model to emit metric values directly, not
    just improve its relative ordering.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.config import settings

DEFAULT_TRAINING_RESOLUTION = 518  # must match train_lora.py's --resolution default

_peft_model = None
_device: Optional[torch.device] = None
_training_resolution: Optional[int] = None
_lock = threading.Lock()


@dataclass
class LoraDepthResult:
    absolute_depth: np.ndarray  # (H, W) float32, meters — the model's DIRECT output, no post-hoc regression
    model_id: str
    adapter_dir: str


class LoraCalibrationError(RuntimeError):
    pass


def is_lora_available() -> bool:
    """True once a trained adapter has been dropped at
    settings.lora_adapter_dir (see module docstring) — always False
    until then, which is the current state of this repo."""
    return (settings.lora_adapter_dir / "adapter_config.json").exists()


def _load_model() -> None:
    global _peft_model, _device, _training_resolution
    if _peft_model is not None:
        return

    with _lock:
        if _peft_model is not None:  # re-check inside the lock
            return
        if not is_lora_available():
            raise LoraCalibrationError(
                f"No LoRA adapter found at {settings.lora_adapter_dir}. "
                "Train one with training/train_lora.py (see training/README.md) "
                "and drop the output directory there."
            )

        from peft import PeftModel
        from transformers import AutoModelForDepthEstimation

        from app.services.depth_anything_model import HF_MODEL_IDS

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = HF_MODEL_IDS[settings.depth_anything_live_variant]
        base_model = AutoModelForDepthEstimation.from_pretrained(model_id)
        peft_model = PeftModel.from_pretrained(base_model, str(settings.lora_adapter_dir))
        peft_model.to(device)
        peft_model.eval()

        _peft_model, _device = peft_model, device
        _training_resolution = _read_training_resolution()


def _read_training_resolution() -> int:
    """train_lora.py saves training_report.json (with the --resolution
    it was actually run at) alongside the adapter weights. Fall back to
    the script's own default if that file isn't present (e.g. an
    adapter assembled by hand, like the test fixture)."""
    import json

    report_path = settings.lora_adapter_dir / "training_report.json"
    if report_path.exists():
        try:
            return int(json.loads(report_path.read_text())["resolution"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return DEFAULT_TRAINING_RESOLUTION


def run_inference(rgb: np.ndarray) -> LoraDepthResult:
    """
    rgb: (H, W, 3) uint8 RGB array, ANY size — resized here (square, to
    whatever resolution the adapter was actually trained at, read from
    its training_report.json) before inference, same as
    depth_anything_model.run_inference() / midas_model.run_inference()
    resample their output back to the ORIGINAL input resolution, so all
    three modules are drop-in compatible for the router.

    Unlike those two, the output here needs NO further calibration: the
    whole point of Stage 2b is that the fine-tuned model emits meters
    directly.
    """
    _load_model()
    assert _peft_model is not None and _device is not None and _training_resolution is not None

    from transformers import AutoImageProcessor
    from PIL import Image

    from app.services.depth_anything_model import HF_MODEL_IDS

    original_h, original_w = rgb.shape[:2]
    model_id = HF_MODEL_IDS[settings.depth_anything_live_variant]
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    pil_image = Image.fromarray(rgb).resize(
        (_training_resolution, _training_resolution), Image.BILINEAR
    )
    pixel_values = image_processor(images=pil_image, return_tensors="pt", do_resize=False)[
        "pixel_values"
    ].to(_device)

    with torch.no_grad():
        predicted_depth = _peft_model(pixel_values=pixel_values).predicted_depth
        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_h, original_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = predicted_depth.cpu().numpy().astype(np.float32)
    return LoraDepthResult(
        absolute_depth=depth,
        model_id=model_id,
        adapter_dir=str(settings.lora_adapter_dir),
    )
