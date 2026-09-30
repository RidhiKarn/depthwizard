"""
Stage 2a — DEM-based absolute-scale calibration (GEOREFERENCED images only).

Fits a regression from the AI model's relative depth values to real
elevation, using Copernicus DEM GLO-30 as the reference. NASADEM/AW3D30
were the original preference (see README > Datasets), but Copernicus
DEM GLO-30 is what's actually wired up here because, of the three, it's
the only one readable with zero authentication: it's published as
Cloud-Optimized GeoTIFFs on a fully public, anonymous-read AWS S3 bucket
(s3://copernicus-dem-30m, mirrored at
https://copernicus-dem-30m.s3.amazonaws.com/ — verified reachable via a
plain HTTPS GET with no credentials). NASADEM via OpenTopography needs a
free API key and AW3D30 needs its own registration; swapping to either
later is a one-line change to `_dem_tile_urls()` plus a credentials
lookup, not a redesign — there's just no reason to add that friction for
a working demo when Copernicus DEM GLO-30 already has global coverage.

How it works:
1. `geo_meta.bounds_lonlat` (already extracted in image_io.py) gives the
   image's real-world bounding box in EPSG:4326.
2. `_dem_tile_urls()` works out which 1x1-degree Copernicus tiles cover
   that box (tiles are named e.g. Copernicus_DSM_COG_10_N52_00_E013_00_DEM
   — format confirmed against the bucket's own readme.html plus a live
   HEAD request against a real tile).
3. Each tile is opened directly over HTTP via GDAL's /vsicurl/ virtual
   filesystem (rasterio), so only the needed byte ranges are fetched —
   no full-tile download, no local caching needed for a single request.
4. `rasterio.merge.merge()` mosaics the (usually just one) tiles,
   cropped to the image's bounds.
5. A grid of sample pixels is taken from the AI model's relative-depth
   array; each sample's pixel coordinate is converted to its real-world
   position (`_resized_affine` — the depth map is computed on a resized
   copy of the source image, not the original full-resolution GeoTIFF,
   so the original affine transform has to be rescaled first) and
   reprojected to EPSG:4326, then used to look up the matching DEM
   elevation.
6. A linear regression (elevation ≈ a * relative_depth + b) is fit
   across all valid (non-nodata) sample pairs via numpy.polyfit — the
   original plan called for trying a degree-2 fit if residuals show
   curvature; not added yet, linear is what's implemented and tested.
7. The regression is applied to the full relative-depth array to
   produce an absolute DSM in meters, and the fit's R^2 plus sample
   count are returned in `calibration_report` so accuracy is reported
   honestly rather than assumed.

Caveat worth stating plainly: a single linear regression against 30m
reference data, using a depth model that was never trained on nadir
imagery, is a genuinely weak calibration — a real, working, but
approximate cross-check, not a substitute for Stage 2b's LoRA
fine-tuning, which is where the accuracy is actually meant to come from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.utils.image_io import GeoMetadata

COPERNICUS_DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
SAMPLE_GRID_SIZE = 40  # up to 40x40 = 1600 sample points for the regression
MIN_VALID_SAMPLES = 10


class DemCalibrationError(RuntimeError):
    """Raised when DEM tiles can't be fetched or too few valid samples remain."""


@dataclass
class DemCalibrationResult:
    absolute_dsm: np.ndarray  # (H, W) float32, meters
    report: dict
    # Affine transform (a, b, c, d, e, f) and CRS for absolute_dsm's own
    # pixel grid (the RESIZED image, not the original GeoTIFF) — kept
    # here so callers (the GeoTIFF export endpoint) can georeference the
    # array correctly without recomputing the resize math themselves.
    transform: tuple
    crs: str


def calibrate_with_dem(relative_depth: np.ndarray, geo_meta: GeoMetadata) -> DemCalibrationResult:
    """
    relative_depth: (H, W) float32, normalized [0, 1] output from
        depth_anything_model.run_inference(), computed on the RESIZED
        input image — H, W match the resized dims, NOT geo_meta.width/height.
    geo_meta: GeoMetadata from image_io.py for the ORIGINAL, full-resolution
        GeoTIFF (crs, bounds, transform, width, height).
    """
    if geo_meta.bounds_lonlat is None:
        raise DemCalibrationError(
            "GeoTIFF has a CRS but its bounds could not be reprojected to "
            "EPSG:4326 (see image_io._try_load_geotiff) — cannot look up "
            "a matching DEM tile."
        )

    from rasterio.transform import rowcol
    from rasterio.warp import transform as warp_transform

    resized_h, resized_w = relative_depth.shape
    a, b, c, d, e, f = _resized_affine_coeffs(geo_meta, resized_w, resized_h)

    dem_array, dem_transform, dem_nodata = _read_dem_window(geo_meta.bounds_lonlat)

    rows, cols = _sample_grid(resized_h, resized_w, SAMPLE_GRID_SIZE)
    xs = a * cols + b * rows + c
    ys = d * cols + e * rows + f
    lons, lats = warp_transform(geo_meta.crs, "EPSG:4326", xs.tolist(), ys.tolist())

    dem_rows, dem_cols = rowcol(dem_transform, lons, lats)
    dem_rows = np.asarray(dem_rows)
    dem_cols = np.asarray(dem_cols)

    in_bounds = (
        (dem_rows >= 0)
        & (dem_rows < dem_array.shape[0])
        & (dem_cols >= 0)
        & (dem_cols < dem_array.shape[1])
    )

    elevations = np.full(rows.shape, np.nan, dtype=np.float64)
    elevations[in_bounds] = dem_array[dem_rows[in_bounds], dem_cols[in_bounds]]
    if dem_nodata is not None:
        elevations[elevations == dem_nodata] = np.nan

    rel_values = relative_depth[rows, cols]
    valid = ~np.isnan(elevations) & np.isfinite(rel_values)

    sample_count = int(valid.sum())
    if sample_count < MIN_VALID_SAMPLES:
        raise DemCalibrationError(
            f"Only {sample_count} valid DEM samples overlapped the image bounds "
            f"(need >= {MIN_VALID_SAMPLES}) — the DEM tile may not cover this "
            "area, or the image footprint is mostly water/void in the reference data."
        )

    x = rel_values[valid].astype(np.float64)
    y = elevations[valid]

    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    absolute_dsm = (slope * relative_depth + intercept).astype(np.float32)

    report = {
        "source": "Copernicus DEM GLO-30",
        "sample_count": sample_count,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "elevation_min_m": float(y.min()),
        "elevation_max_m": float(y.max()),
    }
    return DemCalibrationResult(
        absolute_dsm=absolute_dsm,
        report=report,
        transform=(a, b, c, d, e, f),
        crs=geo_meta.crs,
    )


def _resized_affine_coeffs(
    geo_meta: GeoMetadata, resized_w: int, resized_h: int
) -> tuple[float, float, float, float, float, float]:
    """geo_meta.transform describes the ORIGINAL full-resolution GeoTIFF
    (pixel (col, row) -> native-CRS (x, y): x = a*col + b*row + c,
    y = d*col + e*row + f). The depth map was computed on a resized copy
    (see routers/depth.py:_resize_longest_edge), so the transform has to
    be rescaled to match the resized pixel grid before use."""
    from rasterio.transform import Affine

    orig_transform = Affine(*geo_meta.transform)
    sx = geo_meta.width / resized_w
    sy = geo_meta.height / resized_h
    resized = orig_transform @ Affine.scale(sx, sy)
    return resized.a, resized.b, resized.c, resized.d, resized.e, resized.f


def _sample_grid(height: int, width: int, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.linspace(0, height - 1, min(grid_size, height), dtype=int)
    cols = np.linspace(0, width - 1, min(grid_size, width), dtype=int)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    return row_grid.ravel(), col_grid.ravel()


def _dem_tile_urls(bounds_lonlat: tuple[float, float, float, float]) -> list[str]:
    """Copernicus DEM GLO-30 tiles sit on a 1x1-degree grid, named e.g.
    Copernicus_DSM_COG_10_N52_00_E013_00_DEM (2-digit zero-padded
    latitude, 3-digit zero-padded longitude)."""
    left, bottom, right, top = bounds_lonlat
    lat_start, lat_end = math.floor(bottom), math.floor(top)
    lon_start, lon_end = math.floor(left), math.floor(right)

    urls = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            urls.append(f"/vsicurl/{COPERNICUS_DEM_BUCKET}/{tile}/{tile}.tif")
    return urls


def _read_dem_window(bounds_lonlat: tuple[float, float, float, float]):
    """Opens only the tiles overlapping the bounds via GDAL's /vsicurl/
    (byte-range HTTP reads — no full-tile download) and mosaics them,
    cropped to the bounds. Returns (elevation_array, transform, nodata)."""
    import rasterio
    from rasterio.merge import merge as rasterio_merge

    datasets = []
    try:
        for url in _dem_tile_urls(bounds_lonlat):
            try:
                datasets.append(rasterio.open(url))
            except rasterio.errors.RasterioIOError:
                continue  # tile doesn't exist (e.g. open ocean) — skip, not fatal

        if not datasets:
            raise DemCalibrationError(
                f"No Copernicus DEM tiles found for bounds {bounds_lonlat} "
                "(area may be entirely open ocean, or outside global coverage)."
            )

        nodata = datasets[0].nodata
        mosaic, mosaic_transform = rasterio_merge(datasets, bounds=bounds_lonlat)
        return mosaic[0], mosaic_transform, nodata
    finally:
        for ds in datasets:
            ds.close()
