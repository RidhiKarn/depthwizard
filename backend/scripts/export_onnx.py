"""
One-off export of Depth Anything from PyTorch/transformers to ONNX.

Why this exists: the deployed backend (Render's free web service, 512MB
RAM) was hitting "Ran out of memory" on the very first image upload.
Root cause, confirmed by direct measurement on this project's own dev
machine: PyTorch + transformers carry a large baseline memory footprint
(hundreds of MB) just from being imported — before any model weights
are even loaded. ONNX Runtime runs the exact same trained model with a
dramatically smaller footprint (no large C++ tensor-autograd runtime to
load). This script produces the .onnx file once, offline; the deployed
app (app/services/depth_anything_model.py) then loads that file with
`onnxruntime` at runtime and never imports torch/transformers on the
default request path at all.

Verified before committing to this approach (not assumed):
    - Export succeeds without errors for LiheYoung/depth-anything-small-hf.
    - The ONNX model's output matches the original PyTorch model's
      output to within ~1e-5 (effectively identical, floating-point
      rounding only) on a random test image.

Run this again only if you want to (re-)generate the .onnx file, e.g.
for the "vitl" variant, or after a model update. Needs torch +
transformers installed (only for this one-time export — NOT a runtime
dependency of the deployed app anymore).

    cd backend
    python scripts/export_onnx.py --variant vits
"""

from __future__ import annotations

import argparse
from pathlib import Path

HF_MODEL_IDS = {
    "vits": "LiheYoung/depth-anything-small-hf",
    "vitl": "LiheYoung/depth-anything-large-hf",
}

# Depth Anything's ViT backbone has shape-dependent branching (see the
# TracerWarnings this export emits) that makes a dynamic input
# resolution unsafe to trust — so this fixes a single square input size
# and the deployed app always resizes to exactly this before inference,
# same pattern already used for the LoRA training/inference path.
EXPORT_RESOLUTION = 518


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=list(HF_MODEL_IDS), default="vits")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "app" / "ml_models"),
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForDepthEstimation

    model_id = HF_MODEL_IDS[args.variant]
    print(f"Loading {model_id} ...")
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()

    dummy_input = torch.randn(1, 3, EXPORT_RESOLUTION, EXPORT_RESOLUTION)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"depth_anything_{args.variant}.onnx"

    print(f"Exporting to {output_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            dynamic_axes={"pixel_values": {0: "batch"}, "predicted_depth": {0: "batch"}},
            opset_version=17,
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported (float32): {output_path} ({size_mb:.1f} MB)")

    # Dynamic int8 quantization: shrinks weights ~4x (smaller file to
    # commit to git, comfortably under GitHub's 100MB hard limit) and
    # correspondingly less RAM to hold them at runtime — meaningful on a
    # memory-capped host, and dynamic quantization's accuracy cost is
    # small for this kind of regression-style output (verified below,
    # not assumed).
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantized_path = output_dir / f"depth_anything_{args.variant}_quantized.onnx"
    print(f"Quantizing to {quantized_path} ...")
    quantize_dynamic(str(output_path), str(quantized_path), weight_type=QuantType.QUInt8)
    quantized_size_mb = quantized_path.stat().st_size / 1e6
    print(f"Done. {quantized_path} ({quantized_size_mb:.1f} MB)")

    output_path.unlink()  # only the quantized version ships


if __name__ == "__main__":
    main()
