"""
Stage 2e — Uncertainty / confidence mapping (bonus layer).

STATUS: not yet implemented. Last build-priority item before polish/deploy.

Deliberately pairs MiDaS against Depth Anything rather than comparing
two Depth Anything sizes (ViT-S vs ViT-L) against each other: the two
Depth Anything sizes share the same training data and method, so their
disagreement would mostly reflect model capacity, not real scene
ambiguity. MiDaS (app/services/midas_model.py) was trained on different
data with a different method, so pairing it against Depth Anything
(app/services/depth_anything_model.py, Stage 1's primary backbone)
produces a more meaningful disagreement signal — and costs nothing extra
since MiDaS is already integrated (kept in the codebase for exactly this
reason).

Planned approach:
1. Run MiDaS on the same image already fed to Depth Anything for Stage 1
   (both modules already exist — this step is just calling both and
   keeping the outputs).
2. Normalize both depth maps to a common scale (min-max, or the
   DEM/LoRA calibrated scale once those stages exist).
3. Compute a per-pixel disagreement score, e.g.
       confidence = 1 - clip(|depth_midas - depth_anything| / disagreement_norm, 0, 1)
   where disagreement_norm is a robust scale (e.g. 95th percentile of
   |diff| across the image) so the map isn't dominated by a few outliers.
4. Return this as a separate grayscale heatmap the frontend can toggle
   on/off as an overlay (see TerrainViewer.js confidence toggle).

This module will expose:
    compute_confidence_map(depth_a: np.ndarray, depth_b: np.ndarray) -> np.ndarray
"""

from __future__ import annotations

import numpy as np


def compute_confidence_map(depth_a: np.ndarray, depth_b: np.ndarray) -> np.ndarray:
    """
    Pure-math step, implemented now even though the two-model call
    (MiDaS + Depth Anything on the same image) isn't wired up yet, so
    it's ready to plug in. Returns a (H, W) float32 array in [0, 1],
    1 = models agree.
    """
    diff = np.abs(depth_a - depth_b)
    norm = np.percentile(diff, 95) or 1e-6
    confidence = 1.0 - np.clip(diff / norm, 0.0, 1.0)
    return confidence.astype(np.float32)
