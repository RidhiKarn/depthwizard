"""
Stage 2c — Shadow-length + sun-angle cross-check.

Independent of the AI depth model — used as a cross-validation signal,
not the primary estimate. Two parts:

1. Shadow detection (`detect_shadow_candidates`): a deliberately
   simplified contour-based detector, NOT a full object-detector
   pipeline (e.g. YOLO trained on building footprints) — noted here and
   in the README as a "future work" item. It thresholds the image for
   dark blobs (Otsu, adaptive per-image rather than a fixed brightness
   cutoff) and treats each large-enough dark contour as a shadow
   candidate, without trying to separately identify the building that
   cast it. Each candidate's pixel extent along the sun-azimuth
   direction (shadows point directly away from the sun) is taken as its
   shadow length. This will false-positive on any dark area that isn't
   actually a shadow (dark roofs, water, tree canopy shade) — acceptable
   for a cross-check signal, not acceptable as a sole source of truth,
   which is exactly why it's a cross-check and not the primary estimate.

2. Geometry (`estimate_height_from_shadow`): the textbook formula,
       height = shadow_length_meters * tan(solar_elevation_degrees)
   Converting a measured pixel length to meters needs the image's
   ground sample distance (GSD, meters/pixel) — derived automatically
   from GeoTIFF metadata when available (see routers/shadow.py), or
   supplied by the user for non-georeferenced images.

Solar angle input: if a capture timestamp + location were available,
`compute_solar_elevation`/`compute_solar_azimuth` (pysolar) would give
exact angles. In practice most uploads (GeoTIFF or not) don't carry a
reliable capture timestamp, so the current wiring in routers/shadow.py
always takes sun elevation/azimuth as explicit user input, with a
visible "reduced accuracy if guessed" disclaimer in the frontend —
matching the spec's own documented fallback for exactly this case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ShadowGeometryError(RuntimeError):
    pass


@dataclass
class ShadowCandidate:
    length_px: float
    area_px: float
    centroid_x: float
    centroid_y: float


@dataclass
class ShadowHeightEstimate:
    centroid_x: float
    centroid_y: float
    shadow_length_px: float
    shadow_length_m: float
    estimated_height_m: float


def estimate_height_from_shadow(
    shadow_length_px: float,
    gsd_meters_per_px: float,
    solar_elevation_deg: float,
) -> float:
    """height = shadow_length_meters * tan(solar_elevation_degrees)."""
    shadow_length_m = shadow_length_px * gsd_meters_per_px
    return shadow_length_m * math.tan(math.radians(solar_elevation_deg))


def detect_shadow_candidates(
    rgb: np.ndarray,
    sun_azimuth_deg: float,
    max_candidates: int = 20,
    min_area_px: int = 30,
) -> list[ShadowCandidate]:
    """
    sun_azimuth_deg: compass bearing (0=N, 90=E, 180=S, 270=W) the sun is
    coming FROM. Shadows fall in the opposite direction.

    Returns up to max_candidates ShadowCandidate, largest-shadow-first,
    each with its extent (in pixels) measured along the away-from-sun
    direction and its pixel centroid (for the frontend to place a marker
    on the source image / mesh).
    """
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Otsu adapts the brightness cutoff per-image rather than using a
    # fixed threshold, which would fail across differently-lit photos.
    _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shadow_dir_rad = math.radians(sun_azimuth_deg + 180)
    # Compass bearing -> image-space direction: +x = east (right),
    # +y = south (down — image rows increase downward).
    dir_x = math.sin(shadow_dir_rad)
    dir_y = -math.cos(shadow_dir_rad)

    candidates: list[ShadowCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue
        points = contour.reshape(-1, 2).astype(np.float64)  # (N, 2) as (x, y)
        projections = points[:, 0] * dir_x + points[:, 1] * dir_y
        length_px = float(projections.max() - projections.min())
        centroid = points.mean(axis=0)
        candidates.append(
            ShadowCandidate(
                length_px=length_px,
                area_px=float(area),
                centroid_x=float(centroid[0]),
                centroid_y=float(centroid[1]),
            )
        )

    candidates.sort(key=lambda c: c.length_px, reverse=True)
    return candidates[:max_candidates]


def estimate_heights_from_shadows(
    rgb: np.ndarray,
    solar_elevation_deg: float,
    solar_azimuth_deg: float,
    gsd_meters_per_px: float,
    max_candidates: int = 20,
) -> list[ShadowHeightEstimate]:
    if solar_elevation_deg <= 0:
        raise ShadowGeometryError(
            "Sun elevation is <= 0 degrees (at or below the horizon) — no shadows are geometrically possible."
        )
    if gsd_meters_per_px <= 0:
        raise ShadowGeometryError("gsd_meters_per_px must be positive.")

    candidates = detect_shadow_candidates(rgb, solar_azimuth_deg, max_candidates=max_candidates)
    return [
        ShadowHeightEstimate(
            centroid_x=c.centroid_x,
            centroid_y=c.centroid_y,
            shadow_length_px=c.length_px,
            shadow_length_m=c.length_px * gsd_meters_per_px,
            estimated_height_m=estimate_height_from_shadow(
                c.length_px, gsd_meters_per_px, solar_elevation_deg
            ),
        )
        for c in candidates
    ]


def compute_solar_elevation(lat: float, lon: float, when) -> float:
    """Thin wrapper around pysolar. Not yet wired into automatic
    timestamp-based estimation (see module docstring) — exposed for
    when GeoTIFF capture-time extraction lands."""
    from pysolar.solar import get_altitude  # local import: optional heavy dep

    return get_altitude(lat, lon, when)


def compute_solar_azimuth(lat: float, lon: float, when) -> float:
    from pysolar.solar import get_azimuth  # local import: optional heavy dep

    return get_azimuth(lat, lon, when)
