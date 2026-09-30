"""
Generic RGB + height-raster paired dataset for Stage 2b LoRA fine-tuning.

Deliberately source-agnostic: rather than hardcoding ISPRS/DFC2018/
DFC2019/self-built folder layouts (which differ from each other, and
none of which are downloaded in this repo yet — see README.md in this
directory for the registration steps), this reads a flat manifest CSV
that maps each RGB image to its matching height raster. Converting any
source dataset into this manifest format is a short one-off script you
write once you actually have the data (walk that dataset's real folder
structure, emit a manifest.csv row per pair) — this loader itself never
needs to change per dataset.

Manifest columns (header row required):
    rgb_path       - path to an RGB image (any format PIL can open)
    height_path    - path to a single-band height raster in meters
                     (.tif/.tiff via rasterio, or .npy)
    height_scale   - optional, default 1.0. raw_value * scale + offset = meters
    height_offset  - optional, default 0.0
    category       - optional free-text terrain category ("urban",
                     "hilly", "sparse", "forested", ...) — used only for
                     reporting per-category validation metrics, matching
                     the project's evaluation criteria (RMSE/MAE/
                     correlation tested separately per terrain type).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ManifestRow:
    rgb_path: Path
    height_path: Path
    height_scale: float
    height_offset: float
    category: str


def load_manifest(manifest_path: Union[str, Path]) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with open(manifest_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                ManifestRow(
                    rgb_path=Path(r["rgb_path"]),
                    height_path=Path(r["height_path"]),
                    height_scale=float(r.get("height_scale") or 1.0),
                    height_offset=float(r.get("height_offset") or 0.0),
                    category=(r.get("category") or "unknown").strip() or "unknown",
                )
            )
    if not rows:
        raise ValueError(f"Manifest {manifest_path} has no rows.")
    return rows


def _load_height_raster(path: Path, scale: float, offset: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
    elif suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported height raster format '{suffix}' for {path}")

    return arr * scale + offset


class DepthCalibrationDataset(Dataset):
    """
    Returns (pixel_values, height_target, valid_mask, category) per
    sample:
      - pixel_values: (3, resolution, resolution) float tensor, resized
        to an exact SQUARE via PIL (required for fixed-shape batching —
        the model's own AutoImageProcessor preserves aspect ratio by
        default, which produces variable shapes across a batch, so we
        resize ourselves and only use the processor for its verified
        normalization stats: do_resize=False, image_mean/std unchanged
        from the pretrained checkpoint).
      - height_target: (resolution, resolution) float tensor, meters,
        NaNs replaced with 0 (excluded from loss via valid_mask instead).
      - valid_mask: (resolution, resolution) bool tensor.
      - category: str, for per-terrain-category metric reporting.
    """

    def __init__(self, manifest_path, image_processor, resolution: int = 518):
        if resolution % 14 != 0:
            raise ValueError(
                f"resolution must be a multiple of 14 (ViT encoder patch size), got {resolution}"
            )
        self.rows = load_manifest(manifest_path)
        self.image_processor = image_processor
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        rgb = Image.open(row.rgb_path).convert("RGB")
        rgb = rgb.resize((self.resolution, self.resolution), Image.BILINEAR)
        pixel_values = self.image_processor(
            images=rgb, return_tensors="pt", do_resize=False
        )["pixel_values"][0]

        height = _load_height_raster(row.height_path, row.height_scale, row.height_offset)
        height_tensor = torch.from_numpy(height).unsqueeze(0).unsqueeze(0)
        height_tensor = torch.nn.functional.interpolate(
            height_tensor,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        valid_mask = torch.isfinite(height_tensor)
        height_tensor = torch.nan_to_num(height_tensor, nan=0.0)

        return pixel_values, height_tensor, valid_mask, row.category
