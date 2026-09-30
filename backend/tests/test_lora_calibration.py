"""
Stage 2b — verifies the LoRA adapter LOADING/inference path against a
real (structurally valid, but untrained) adapter, since no genuinely
trained adapter exists in this repo yet (needs registered datasets + a
GPU — see backend/training/README.md). This codifies the same
save -> load -> forward-pass round trip that was manually verified
against the real LiheYoung/depth-anything-small-hf model before writing
lora_calibration.py, as an actual test rather than a one-off check.
"""
import numpy as np
import pytest

from app.config import settings
from app.services.calibration import lora_calibration


@pytest.fixture
def fake_adapter(tmp_path):
    """Builds and saves a real (untrained) LoRA adapter with the exact
    config training/train_lora.py uses, so the loading path is tested
    against a genuine peft checkpoint, not a mock."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForDepthEstimation

    base = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["neck", "head"],
    )
    peft_model = get_peft_model(base, config)
    peft_model.save_pretrained(str(tmp_path))
    return tmp_path


def test_is_lora_available_false_by_default():
    # No adapter has actually been trained in this repo yet — the
    # module must be honest about that rather than pretending.
    assert lora_calibration.is_lora_available() is False


def test_lora_adapter_round_trip(fake_adapter, monkeypatch):
    monkeypatch.setattr(settings, "lora_adapter_dir", fake_adapter)
    lora_calibration._peft_model = None  # force a fresh load under the patched path
    lora_calibration._training_resolution = None

    assert lora_calibration.is_lora_available() is True

    # Deliberately NOT square and NOT the training resolution, to
    # exercise the resize-to-training-res -> infer -> resample-back-to-
    # original-shape path, not just a lucky no-op case.
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(256, 384, 3), dtype=np.uint8)
    result = lora_calibration.run_inference(rgb)

    assert result.absolute_depth.shape == (256, 384)
    assert result.adapter_dir == str(fake_adapter)
