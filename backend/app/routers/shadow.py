"""
Stage 2c API surface: POST /api/shadow/estimate — independent
shadow-length + sun-angle cross-check (see
app/services/shadow/shadow_geometry.py for the algorithm and its
documented limitations).

Kept as a separate endpoint from /api/depth/estimate because it's
explicitly an INDEPENDENT cross-check (per the project spec) with its
own inputs (sun angle, optionally a manual GSD) that don't belong on
the main depth-estimation request.
"""

from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.routers.auth import get_current_user
from app.schemas import ShadowEstimateResponse, ShadowHeightItem
from app.services import auth as auth_service
from app.services.shadow.shadow_geometry import ShadowGeometryError, estimate_heights_from_shadows
from app.utils.image_io import GeoMetadata, load_image_any

router = APIRouter(prefix="/shadow", tags=["shadow"])

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_CANDIDATES = 20


@router.post("/estimate", response_model=ShadowEstimateResponse)
async def estimate_shadow_heights(
    file: UploadFile = File(...),
    sun_elevation_deg: float = Form(...),
    sun_azimuth_deg: float = Form(...),
    gsd_meters_per_px: Optional[float] = Form(None),
    current_user: auth_service.User = Depends(get_current_user),
) -> ShadowEstimateResponse:
    """
    Requires a signed-in session, same as /api/depth/estimate.

    sun_elevation_deg / sun_azimuth_deg: required, always user-supplied.
    There's no reliable way to extract a capture timestamp from
    arbitrary uploads (see shadow_geometry.py docstring for why the
    pysolar auto-path isn't wired up), so the frontend collects these
    directly with a "reduced accuracy if estimated" disclaimer —
    matching the spec's own documented fallback for this exact case.

    gsd_meters_per_px: required for non-georeferenced images (no way to
    convert pixel distances to meters otherwise). Ignored in favor of a
    value auto-derived from the GeoTIFF's own affine transform for
    georeferenced uploads.
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
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    if loaded.is_georeferenced and loaded.geo is not None:
        gsd = _gsd_from_geo(loaded.geo)
        gsd_source = "geotiff_transform"
    elif gsd_meters_per_px is not None and gsd_meters_per_px > 0:
        gsd = gsd_meters_per_px
        gsd_source = "user_supplied"
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "gsd_meters_per_px is required for non-georeferenced images "
                "(no way to convert pixel distances to meters otherwise)."
            ),
        )

    try:
        estimates = estimate_heights_from_shadows(
            loaded.rgb,
            solar_elevation_deg=sun_elevation_deg,
            solar_azimuth_deg=sun_azimuth_deg,
            gsd_meters_per_px=gsd,
            max_candidates=MAX_CANDIDATES,
        )
    except ShadowGeometryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_height, image_width = loaded.rgb.shape[:2]

    return ShadowEstimateResponse(
        image_width=image_width,
        image_height=image_height,
        candidate_count=len(estimates),
        gsd_meters_per_px=gsd,
        gsd_source=gsd_source,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        estimates=[
            ShadowHeightItem(
                x=e.centroid_x,
                y=e.centroid_y,
                shadow_length_px=e.shadow_length_px,
                shadow_length_m=e.shadow_length_m,
                estimated_height_m=e.estimated_height_m,
            )
            for e in estimates
        ],
    )


def _gsd_from_geo(geo_meta: GeoMetadata) -> float:
    """meters/pixel, derived from the GeoTIFF's own affine transform.
    Projected (metric) CRSs: pixel size is already in meters. Geographic
    (degree) CRSs: approximated via the standard meters-per-degree
    constants, scaled by cos(latitude) for longitude — accurate to
    within ~1% away from the poles."""
    import rasterio

    px_size_x = abs(geo_meta.transform[0])
    px_size_y = abs(geo_meta.transform[4])

    crs_obj = rasterio.crs.CRS.from_string(geo_meta.crs)
    if not crs_obj.is_geographic:
        return (px_size_x + px_size_y) / 2.0

    lat_center = (
        (geo_meta.bounds_lonlat[1] + geo_meta.bounds_lonlat[3]) / 2.0
        if geo_meta.bounds_lonlat is not None
        else 0.0
    )
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_center))
    gsd_x = px_size_x * meters_per_deg_lon
    gsd_y = px_size_y * meters_per_deg_lat
    return (gsd_x + gsd_y) / 2.0
