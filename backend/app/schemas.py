"""Pydantic response models for the public API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class GeoMetadataResponse(BaseModel):
    crs: str
    bounds: tuple[float, float, float, float]
    bounds_lonlat: Optional[tuple[float, float, float, float]] = None


class DepthEstimateResponse(BaseModel):
    # "model_type"/"model_" fields below collide with pydantic's default
    # protected namespace (meant to guard actual ML "model_" attributes
    # on the BaseModel class itself, not our field names) — disable it.
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    filename: str
    width: int
    height: int
    is_georeferenced: bool
    geo: Optional[GeoMetadataResponse] = None

    # Which calibration stage actually produced this response:
    # "relative_uncalibrated" (Stage 1 only), "dem_calibrated" (Stage 2a
    # succeeded, georeferenced images), or "lora_calibrated" (Stage 2b
    # succeeded — only possible once a trained adapter exists; see
    # app/services/calibration/lora_calibration.py) — so the frontend/
    # README can state honestly what a given result is.
    calibration_stage: str

    model_type: str
    device: str

    image_url: str
    depth_heatmap_url: str

    # Stage 2e (bonus, opt-in via include_confidence=true on the
    # request): MiDaS-vs-Depth-Anything disagreement heatmap. None when
    # not requested, or if the second model pass failed.
    confidence_heatmap_url: Optional[str] = None

    # Downsampled (height_grid_resolution x height_grid_resolution) grid
    # of normalized [0, 1] relative depth values, row-major, used by the
    # Three.js viewer to build the displacement mesh without shipping a
    # full-resolution array over JSON.
    height_grid: list[list[float]]
    height_grid_resolution: int

    # Real meters instead of normalized [0, 1], same shape/resolution as
    # height_grid — populated when calibration_stage is "dem_calibrated"
    # or "lora_calibrated", None for "relative_uncalibrated".
    absolute_height_grid: Optional[list[list[float]]] = None

    # Stage 2a: regression fit's honesty check (slope/intercept/R^2/
    # sample_count) when it succeeded; dem_calibration_error explains
    # why when it was attempted but failed (calibration_stage then stays
    # "relative_uncalibrated" unless Stage 2b picks it up instead).
    dem_calibration_report: Optional[dict] = None
    dem_calibration_error: Optional[str] = None

    # Full-resolution GeoTIFF of the calibrated DSM, real CRS + transform
    # embedded, usable directly in GIS software. Only set when Stage 2a
    # succeeded (calibration_stage == "dem_calibrated") — LoRA-calibrated
    # results have no CRS to embed, so no GeoTIFF is offered for those.
    dsm_geotiff_url: Optional[str] = None

    # Full-resolution, single-band float32 TIFF with NO crs/transform —
    # the rDSM deliverable for inputs with no location data. Set exactly
    # when dsm_geotiff_url isn't (i.e. calibration_stage != "dem_calibrated"):
    # relative [0, 1] depth for "relative_uncalibrated", real meters
    # (just without a CRS) for "lora_calibrated".
    relative_dsm_tif_url: Optional[str] = None

    # Stage 2b: only ever set if a trained adapter existed AND inference
    # with it failed — is_lora_available()==False (the default, no
    # adapter trained yet) means this stage is simply never attempted,
    # not an error.
    lora_calibration_error: Optional[str] = None

    generated_at: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class ShadowHeightItem(BaseModel):
    # Pixel centroid of the detected shadow blob, in the coordinate
    # space of the image the request was made against — the frontend
    # can use this to place a marker.
    x: float
    y: float
    shadow_length_px: float
    shadow_length_m: float
    estimated_height_m: float


class ShadowEstimateResponse(BaseModel):
    # Pixel dimensions of the image these coordinates are measured
    # against (loaded.rgb — the ORIGINAL upload resolution, not the
    # resized copy /api/depth/estimate uses) — the frontend needs these
    # to convert each estimate's x/y into percentage positions so
    # numbered markers line up correctly over whatever size the photo is
    # actually displayed at.
    image_width: int
    image_height: int

    candidate_count: int
    gsd_meters_per_px: float
    # "geotiff_transform" (auto-derived) or "user_supplied" — so the UI
    # can flag when ground-sample-distance, and therefore every height
    # estimate, is a user guess rather than a measured value.
    gsd_source: str
    sun_elevation_deg: float
    sun_azimuth_deg: float
    estimates: list[ShadowHeightItem]


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str


class MeResponse(BaseModel):
    email: str
