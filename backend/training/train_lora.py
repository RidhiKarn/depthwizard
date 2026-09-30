"""
Stage 2b — LoRA fine-tuning for Depth Anything, using the recipe from
the Depth Anything paper (as specified for this project):
  - encoder learning rate = decoder learning rate / 50
  - batch size 16
  - ~5 epochs to convergence
  - training resolution a multiple of 14 (ViT patch size)

Everything in this file was checked against the REAL downloaded
`LiheYoung/depth-anything-small-hf` model before being written — not
guessed. Specifically verified in this environment (CPU, no training
data, just architecture/API checks):
  - LoRA target_modules=["query", "value"] matches real Linear layer
    names inside model.backbone (Dinov2 attention: attention.query,
    attention.key, attention.value) with zero collisions outside the
    backbone.
  - modules_to_save=["neck", "head"] correctly keeps the depth decoder
    fully trainable (not LoRA-adapted) and bundles it into the same
    saved adapter checkpoint.
  - A full save -> fresh-load -> forward-pass round trip works via
    peft.PeftModel.from_pretrained().
  - The model's own AutoImageProcessor preserves aspect ratio by
    default (not suitable for fixed-shape batching), so this script
    resizes to an exact square itself and calls the processor with
    do_resize=False, keeping its verified ImageNet normalization stats
    (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).
  - A (1, 3, 518, 518) input produces a (1, 518, 518) predicted_depth
    — already exactly the input resolution, no extra resampling needed
    before computing the loss.

What is NOT verified: actual training convergence / accuracy, because
this dev environment has no GPU and no training data (see
README.md in this directory for how to get both). Run this on Colab/
Kaggle/any CUDA box with a real manifest, and report the numbers
train_report.json prints — don't assume they'll match any particular
target.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DepthCalibrationDataset


def build_model(model_id: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    base_model = AutoModelForDepthEstimation.from_pretrained(model_id)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=["neck", "head"],
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model, image_processor


def build_optimizer(peft_model, decoder_lr: float, encoder_lr_ratio: float):
    """Two param groups: LoRA-adapted encoder at decoder_lr * encoder_lr_ratio
    (the recipe's "encoder LR = decoder LR / 50" -> encoder_lr_ratio=1/50),
    fully-unfrozen decoder (neck+head, via modules_to_save) at decoder_lr.
    Verified against real parameter names: LoRA params always contain
    "lora_" in their dotted name; modules_to_save params never do."""
    encoder_lr = decoder_lr * encoder_lr_ratio
    lora_params, decoder_params = [], []
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        (lora_params if "lora_" in name else decoder_params).append(param)

    return torch.optim.AdamW(
        [
            {"params": lora_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]
    )


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return pred.sum() * 0.0  # keep the graph valid, contribute nothing
    diff = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return diff[mask].mean()


@torch.no_grad()
def evaluate(peft_model, loader, device) -> dict:
    """Overall + per-category RMSE/MAE/correlation, matching the
    project's stated evaluation criteria (accuracy tested separately
    per terrain category)."""
    peft_model.eval()
    errors_by_category: dict[str, list] = {}
    preds_by_category: dict[str, list] = {}
    targets_by_category: dict[str, list] = {}

    for pixel_values, height_target, valid_mask, categories in loader:
        pixel_values = pixel_values.to(device)
        height_target = height_target.to(device)
        valid_mask = valid_mask.to(device)

        predicted_depth = peft_model(pixel_values=pixel_values).predicted_depth

        for i, category in enumerate(categories):
            mask_i = valid_mask[i]
            if mask_i.sum() == 0:
                continue
            p = predicted_depth[i][mask_i].detach().cpu().numpy()
            t = height_target[i][mask_i].detach().cpu().numpy()
            errors_by_category.setdefault(category, []).append(p - t)
            preds_by_category.setdefault(category, []).append(p)
            targets_by_category.setdefault(category, []).append(t)

    def summarize(errors, preds, targets) -> dict:
        errors = np.concatenate(errors)
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        corr = float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else float("nan")
        return {"rmse_m": rmse, "mae_m": mae, "correlation": corr, "sample_pixel_count": int(len(errors))}

    report = {"overall": None, "by_category": {}}
    all_errors = [e for v in errors_by_category.values() for e in v]
    all_preds = [p for v in preds_by_category.values() for p in v]
    all_targets = [t for v in targets_by_category.values() for t in v]
    if all_errors:
        report["overall"] = summarize(all_errors, all_preds, all_targets)
    for category in errors_by_category:
        report["by_category"][category] = summarize(
            errors_by_category[category], preds_by_category[category], targets_by_category[category]
        )
    return report


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    peft_model, image_processor = build_model(args.model_id, args.lora_r, args.lora_alpha, args.lora_dropout)
    peft_model.to(device)

    train_dataset = DepthCalibrationDataset(args.manifest, image_processor, resolution=args.resolution)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )

    val_loader = None
    if args.val_manifest:
        val_dataset = DepthCalibrationDataset(args.val_manifest, image_processor, resolution=args.resolution)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

    optimizer = build_optimizer(peft_model, args.decoder_lr, args.encoder_lr_ratio)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        peft_model.train()
        epoch_start = time.time()
        running_loss = 0.0
        batch_count = 0

        for pixel_values, height_target, valid_mask, _categories in train_loader:
            pixel_values = pixel_values.to(device)
            height_target = height_target.to(device)
            valid_mask = valid_mask.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                predicted_depth = peft_model(pixel_values=pixel_values).predicted_depth
                loss = masked_huber_loss(predicted_depth, height_target, valid_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / max(batch_count, 1)
        epoch_report = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "seconds": time.time() - epoch_start,
        }

        if val_loader is not None:
            epoch_report["validation"] = evaluate(peft_model, val_loader, device)

        history.append(epoch_report)
        print(json.dumps(epoch_report, indent=2))

    peft_model.save_pretrained(str(output_dir))
    with open(output_dir / "training_report.json", "w") as f:
        json.dump(
            {
                "model_id": args.model_id,
                "resolution": args.resolution,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "decoder_lr": args.decoder_lr,
                "encoder_lr_ratio": args.encoder_lr_ratio,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Saved adapter + training_report.json to {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV manifest of training pairs (see dataset.py)")
    p.add_argument("--val-manifest", default=None, help="CSV manifest of validation pairs (optional but recommended)")
    p.add_argument("--output-dir", default="lora_adapter_out")
    p.add_argument(
        "--model-id",
        default="LiheYoung/depth-anything-small-hf",
        help="ViT-S for a fast run; LiheYoung/depth-anything-large-hf for the higher-accuracy report run",
    )
    p.add_argument("--resolution", type=int, default=518, help="Must be a multiple of 14 (ViT patch size)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--decoder-lr", type=float, default=1e-4)
    p.add_argument("--encoder-lr-ratio", type=float, default=1 / 50, help="encoder_lr = decoder_lr * this")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        print(
            "WARNING: running on CPU. This recipe (batch 16, 5 epochs) is "
            "designed for GPU (Colab/Kaggle free tier or better) — see README.md. "
            "CPU will work but may take hours-to-days depending on dataset size."
        )
    train(args)
