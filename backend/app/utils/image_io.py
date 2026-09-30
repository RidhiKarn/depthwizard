"""
Stage 1 — image ingestion + georeference detection.

Accepts PNG, JPG, or TIFF/GeoTIFF bytes and normalizes them into a plain
RGB uint8 array for the depth model, while separately reporting whether
the file carried embedded coordinate-system metadata (and if so, what
it was) so the router can branch into the right calibration path later
(Stage 2a needs `GeoMetadata`; Stage 1 only needs `rgb`).

Detection rule: a file is "georeferenced" iff rasterio can open it AND
`src.crs` is not None. A `.tif` with no CRS (plenty of plain TIFFs use
that extension) is treated as a normal, non-georeferenced image — this
mirrors the PS's own definition ("non-georeferenced" = no embedded
location metadata, regardless of container format).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

GEOTIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class GeoMetadata:
    crs: str
    bounds: tuple[float, float, float, float]  # left, bottom, right, top (native CRS units)
    bounds_lonlat: Optional[tuple[float, float, float, float]]  # same, reprojected to EPSG:4326
    transform: list[float]  # affine transform coefficients (a, b, c, d, e, f)
    width: int
    height: int


@dataclass
class LoadedImage:
    rgb: np.ndarray  # (H, W, 3) uint8
    is_georeferenced: bool
    geo: Optional[GeoMetadata]
    original_filename: str


def load_image_any(file_bytes: bytes, filename: str) -> LoadedImage:
    """Entry point used by the /api/depth/estimate router."""
    suffix = Path(filename).suffix.lower()
    if suffix in GEOTIFF_EXTENSIONS:
        result = _try_load_geotiff(file_bytes, filename)
        if result is not None:
            return result
        # Openable as TIFF but no CRS embedded -> fall through and treat
        # as a plain image.
    return _load_plain_image(file_bytes, filename)


def _try_load_geotiff(file_bytes: bytes, filename: str) -> Optional[LoadedImage]:
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.warp import transform_bounds
    except ImportError:
        # rasterio/GDAL not installed in this environment — degrade to
        # treating the file as a plain (non-georeferenced) image rather
        # than hard-failing the whole request.
        return None

    try:
        with MemoryFile(file_bytes) as memfile, memfile.open() as src:
            if src.crs is None:
                return None

            if src.count >= 3:
                arr = src.read([1, 2, 3])  # (3, H, W)
            else:
                band = src.read(1)
                arr = np.stack([band, band, band], axis=0)

            rgb = _to_uint8(np.transpose(arr, (1, 2, 0)))

            bounds = src.bounds
            try:
                bounds_lonlat = transform_bounds(src.crs, "EPSG:4326", *bounds)
            except Exception:
                bounds_lonlat = None

            geo = GeoMetadata(
                crs=str(src.crs),
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                bounds_lonlat=bounds_lonlat,
                transform=list(src.transform)[:6],
                width=src.width,
                height=src.height,
            )
            return LoadedImage(
                rgb=rgb, is_georeferenced=True, geo=geo, original_filename=filename
            )
    except rasterio.errors.RasterioIOError:
        return None


def _load_plain_image(file_bytes: bytes, filename: str) -> LoadedImage:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    rgb = np.array(img)
    return LoadedImage(rgb=rgb, is_georeferenced=False, geo=None, original_filename=filename)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Rescale non-uint8 raster bands (e.g. 16-bit satellite imagery) into
    a displayable/inferrable 8-bit range using a robust 1st/99th
    percentile stretch, so outlier pixels don't wash out the contrast."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)
