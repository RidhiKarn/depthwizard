"""
Stage 1 + 2a + 2e API surface: upload an image, get back a relative
depth map (always) plus a DEM-calibrated absolute DSM in meters when the
image is georeferenced (Stage 2a, on by default) and an optional
MiDaS-vs-Depth-Anything confidence heatmap (Stage 2e, opt-in).

Primary backbone is Depth Anything (app/services/depth_anything_model.py,
ViT-S by default — see app/config.py). MiDaS (app/services/midas_model.py)
is kept for the Stage 2e confidence ensemble.

Stage 2b (LoRA), 2c (shadow geometry), and 2d (Metric3D v2) are not yet
wired in — see app/services/calibration/lora_calibration.py,
app/services/shadow/shadow_geometry.py, and README > Roadmap.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from app.config import settings
from app.routers.auth import get_current_user
from app.schemas import DepthEstimateResponse, GeoMetadataResponse, HealthResponse
from app.services import auth as auth_service
from app.services import confidence as confidence_service
from app.services import midas_model
from app.services.calibration import lora_calibration
from app.services.calibration.dem_calibration import DemCalibrationError, calibrate_with_dem
from app.services.depth_anything_model import is_model_loaded, run_inference
from app.utils.image_io import load_image_any

router = APIRouter(prefix="/depth", tags=["depth"])

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=is_model_loaded())


@router.post("/estimate", response_model=DepthEstimateResponse)
async def estimate_depth(
    file: UploadFile = File(...),
    include_confidence: bool = Form(False),
    calibrate_dem: bool = Form(True),
    current_user: auth_service.User = Depends(get_current_user),
) -> DepthEstimateResponse:
    """
    Requires a signed-in session (Authorization: Bearer <token> from
    /api/auth/login or /api/auth/signup) — this is the product's core
    feature, gated for real accounts, not just cosmetically in the UI.

    include_confidence: Stage 2e (bonus). When true, also runs MiDaS on
    the same image and returns a MiDaS-vs-Depth-Anything disagreement
    heatmap (confidence_heatmap_url). Off by default because it roughly
    doubles inference time (a second model forward pass, plus a one-time
    MiDaS weight download on first use) — the frontend only requests it
    when the user explicitly toggles the confidence overlay on.

    calibrate_dem: Stage 2a. On by default — it's a no-op (near-zero
    cost) for non-georeferenced uploads, and is the project's core
    differentiator for georeferenced ones, so there's no reason to make
    the caller opt in. Fetches Copernicus DEM GLO-30 over the network;
    failures (no coverage, connectivity) degrade gracefully back to
    "relative_uncalibrated" rather than failing the request — see
    dem_calibration_error in the response when that happens.
    """
    filename = file.filename or "upload"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (25 MB max).")

    try:
        loaded = load_image_any(raw, filename)
    except Exception as exc:  # noqa: BLE001 — surface as a clean 400, not a 500
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    resized_rgb, _scale = _resize_longest_edge(loaded.rgb, settings.depth_map_max_dim)

    try:
        depth_result = run_inference(resized_rgb)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Depth inference failed: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    image_url = _save_png(resized_rgb, f"{job_id}_input.png")
    heatmap_url = _save_png(_colorize(depth_result.normalized), f"{job_id}_depth.png")
    height_grid = _downsample_grid(depth_result.normalized, settings.height_grid_resolution)

    confidence_heatmap_url = None
    if include_confidence:
        try:
            midas_result = midas_model.run_inference(resized_rgb)
            confidence_map = confidence_service.compute_confidence_map(
                depth_result.normalized, midas_result.normalized
            )
            # TURBO (not INFERNO, already used for the depth heatmap above)
            # so the two overlays are visually distinguishable at a glance.
            confidence_heatmap_url = _save_png(
                _colorize(confidence_map, cv2.COLORMAP_TURBO), f"{job_id}_confidence.png"
            )
        except Exception:  # noqa: BLE001 — confidence is a bonus layer, never fail the whole request over it
            confidence_heatmap_url = None

    geo_response = None
    if loaded.geo is not None:
        geo_response = GeoMetadataResponse(
            crs=loaded.geo.crs,
            bounds=loaded.geo.bounds,
            bounds_lonlat=loaded.geo.bounds_lonlat,
        )

    calibration_stage = "relative_uncalibrated"
    absolute_height_grid = None
    dem_calibration_report = None
    dem_calibration_error = None

    dsm_geotiff_url = None
    if calibrate_dem and loaded.is_georeferenced and loaded.geo is not None:
        try:
            dem_result = calibrate_with_dem(depth_result.normalized, loaded.geo)
            absolute_height_grid = _downsample_grid(
                dem_result.absolute_dsm, settings.height_grid_resolution
            )
            dem_calibration_report = dem_result.report
            calibration_stage = "dem_calibrated"
            # Full-resolution (not downsampled) DSM, real CRS + transform
            # embedded — usable directly in GIS software (QGIS, ArcGIS),
            # per the original spec's "Download DSM" deliverable.
            dsm_geotiff_url = _save_geotiff(
                dem_result.absolute_dsm, dem_result.transform, dem_result.crs, f"{job_id}_dsm.tif"
            )
        except DemCalibrationError as exc:
            dem_calibration_error = str(exc)
        except Exception as exc:  # noqa: BLE001 — DEM lookup is a cross-check, never fail the whole request over it
            dem_calibration_error = f"DEM calibration failed unexpectedly: {exc}"

    # Stage 2b: if DEM calibration didn't already produce an absolute
    # result (most commonly because the image ISN'T georeferenced — the
    # exact case this stage exists for), and a trained LoRA adapter is
    # available, use its direct metric output instead. No adapter is
    # trained in this repo yet (see app/services/calibration/lora_calibration.py),
    # so is_lora_available() is false today and this is a no-op — wired
    # in now so training/train_lora.py's output activates automatically
    # once it exists, no code changes needed at that point.
    lora_calibration_error = None
    full_res_lora_depth = None
    if calibration_stage == "relative_uncalibrated" and lora_calibration.is_lora_available():
        try:
            lora_result = lora_calibration.run_inference(resized_rgb)
            full_res_lora_depth = lora_result.absolute_depth
            absolute_height_grid = _downsample_grid(
                full_res_lora_depth, settings.height_grid_resolution
            )
            calibration_stage = "lora_calibrated"
        except Exception as exc:  # noqa: BLE001 — same graceful-degradation pattern as DEM calibration above
            lora_calibration_error = f"LoRA calibration failed unexpectedly: {exc}"

    # Raster export for the cases dsm_geotiff_url doesn't cover: no CRS
    # exists for non-georeferenced uploads, but the PS still requires a
    # DSM "in a standard geospatial format" for every accepted input
    # type, not just georeferenced ones. A plain (no CRS/transform)
    # single-band float32 GeoTIFF is that format for the rDSM case —
    # full resolution, not the downsampled height_grid used for the 3D
    # mesh. Skipped when dsm_geotiff_url already exists (georeferenced +
    # DEM-calibrated) since that's a strictly better output for that case.
    relative_dsm_tif_url = None
    if calibration_stage != "dem_calibrated":
        full_res_array = (
            full_res_lora_depth if calibration_stage == "lora_calibrated" else depth_result.normalized
        )
        relative_dsm_tif_url = _save_plain_tif(full_res_array, f"{job_id}_rdsm.tif")

    return DepthEstimateResponse(
        job_id=job_id,
        filename=filename,
        width=int(resized_rgb.shape[1]),
        height=int(resized_rgb.shape[0]),
        is_georeferenced=loaded.is_georeferenced,
        geo=geo_response,
        calibration_stage=calibration_stage,
        model_type=depth_result.model_type,
        device=depth_result.device,
        image_url=image_url,
        depth_heatmap_url=heatmap_url,
        confidence_heatmap_url=confidence_heatmap_url,
        height_grid=height_grid.tolist(),
        height_grid_resolution=settings.height_grid_resolution,
        absolute_height_grid=(
            absolute_height_grid.tolist() if absolute_height_grid is not None else None
        ),
        dem_calibration_report=dem_calibration_report,
        dem_calibration_error=dem_calibration_error,
        lora_calibration_error=lora_calibration_error,
        dsm_geotiff_url=dsm_geotiff_url,
        relative_dsm_tif_url=relative_dsm_tif_url,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _resize_longest_edge(rgb: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return rgb, 1.0
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _colorize(normalized: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    gray = (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray, colormap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _downsample_grid(normalized: np.ndarray, resolution: int) -> np.ndarray:
    return cv2.resize(normalized, (resolution, resolution), interpolation=cv2.INTER_AREA)


def _save_png(rgb: np.ndarray, name: str) -> str:
    path = settings.output_dir / name
    Image.fromarray(rgb).save(path)
    return f"/outputs/{name}"


def _save_plain_tif(array: np.ndarray, name: str) -> str:
    """A single-band float32 GeoTIFF with NO crs/transform — a valid,
    standard raster (openable in any GIS tool or rasterio/GDAL) for
    images with no location data to embed. Values are relative depth
    [0, 1] for "relative_uncalibrated" results, or real meters (just
    without a CRS) for "lora_calibrated" ones."""
    import rasterio

    path = settings.output_dir / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
    ) as dst:
        dst.write(array.astype(np.float32), 1)
    return f"/outputs/{name}"


def _save_geotiff(array: np.ndarray, transform_coeffs: tuple, crs: str, name: str) -> str:
    import rasterio
    from rasterio.transform import Affine

    path = settings.output_dir / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=Affine(*transform_coeffs),
    ) as dst:
        dst.write(array.astype(np.float32), 1)
    return f"/outputs/{name}"
