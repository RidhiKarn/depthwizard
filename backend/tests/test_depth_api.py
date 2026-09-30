"""
Smoke test for the Stage 1 API. Runs actual Depth Anything inference
(downloads weights on first run if not cached), so it's slow (CPU:
~5-15s) — kept minimal on purpose. Run with:

    pytest tests/test_depth_api.py -v

from inside backend/.
"""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _assert_openable_tif(data_uri: str, *, expects_crs: bool):
    """Decodes an inline data: URI raster export and verifies it's a
    real, openable GeoTIFF — not just a string in the response — with or
    without a CRS as expected for that case. Outputs are embedded as
    data: URIs rather than served from a saved file (see
    app/routers/depth.py's _save_plain_tif/_save_geotiff docstrings) so
    there's no separate URL to fetch here."""
    import base64

    import rasterio
    from rasterio.io import MemoryFile

    assert data_uri.startswith("data:image/tiff;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    with MemoryFile(raw) as memfile, memfile.open() as src:
        assert src.count == 1
        assert (src.crs is not None) == expects_crs
        data = src.read(1)
        assert data.shape[0] > 0 and data.shape[1] > 0


def _fake_png_bytes(w: int = 64, h: int = 64) -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _fake_geotiff_bytes(
    bounds: tuple[float, float, float, float] = (13.40, 52.50, 13.41, 52.51),
    size: int = 48,
) -> bytes:
    """A tiny synthetic GeoTIFF (EPSG:4326) over Berlin — chosen because
    we've confirmed the matching Copernicus DEM GLO-30 tile
    (Copernicus_DSM_COG_10_N52_00_E013_00_DEM) is publicly reachable."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    left, bottom, right, top = bounds
    transform = from_bounds(left, bottom, right, top, size, size)
    rng = np.random.default_rng(1)
    data = rng.integers(0, 255, size=(3, size, size), dtype=np.uint8)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=size,
            width=size,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)
        return memfile.read()


def test_health():
    resp = client.get("/api/depth/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_estimate_requires_auth():
    """The core endpoint is gated behind a real account, not just a
    frontend-only decoration (see app/routers/auth.py::get_current_user)."""
    resp = client.post(
        "/api/depth/estimate",
        files={"file": ("sample.png", _fake_png_bytes(), "image/png")},
    )
    assert resp.status_code == 401


def test_estimate_rejects_bad_extension(auth_headers):
    resp = client.post(
        "/api/depth/estimate",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_estimate_png_end_to_end(auth_headers):
    resp = client.post(
        "/api/depth/estimate",
        files={"file": ("sample.png", _fake_png_bytes(), "image/png")},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["is_georeferenced"] is False
    assert body["calibration_stage"] == "relative_uncalibrated"
    assert body["height_grid_resolution"] == len(body["height_grid"])
    assert body["image_url"].startswith("data:image/png;base64,")
    assert body["depth_heatmap_url"].startswith("data:image/png;base64,")
    assert body["confidence_heatmap_url"] is None

    # rDSM raster export (no CRS — this upload has no location data).
    assert body["dsm_geotiff_url"] is None
    assert body["relative_dsm_tif_url"] is not None
    _assert_openable_tif(body["relative_dsm_tif_url"], expects_crs=False)


def test_estimate_with_confidence_ensemble(auth_headers):
    """Stage 2e: also runs MiDaS and returns a disagreement heatmap.
    Slower (downloads MiDaS weights on first run) — that's why it's a
    separate opt-in test rather than the default path above."""
    resp = client.post(
        "/api/depth/estimate",
        files={"file": ("sample.png", _fake_png_bytes(), "image/png")},
        data={"include_confidence": "true"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["confidence_heatmap_url"] is not None
    assert body["confidence_heatmap_url"].startswith("data:image/png;base64,")


def test_estimate_geotiff_dem_calibration(auth_headers):
    """Stage 2a: a georeferenced upload over Berlin should get real
    elevation values from the public Copernicus DEM GLO-30 bucket.
    Requires network access to copernicus-dem-30m.s3.amazonaws.com."""
    resp = client.post(
        "/api/depth/estimate",
        files={"file": ("berlin.tif", _fake_geotiff_bytes(), "image/tiff")},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["is_georeferenced"] is True
    assert body["geo"]["crs"] == "EPSG:4326"
    assert body["calibration_stage"] == "dem_calibrated", body.get("dem_calibration_error")
    assert body["dem_calibration_error"] is None
    assert body["absolute_height_grid"] is not None
    assert len(body["absolute_height_grid"]) == body["height_grid_resolution"]
    report = body["dem_calibration_report"]
    assert report["source"] == "Copernicus DEM GLO-30"
    assert report["sample_count"] >= 10
    # Berlin is flat/low-lying — sanity-bound the fitted elevation range
    # rather than pinning exact values (regression coefficients depend
    # on the random synthetic relative-depth values in this test).
    assert -50 <= report["elevation_min_m"] <= 200
    assert -50 <= report["elevation_max_m"] <= 200

    # Downloadable GeoTIFF DSM (fix 3) — verify it's a real, openable,
    # correctly-georeferenced raster, not just a URL string.
    assert body["dsm_geotiff_url"] is not None
    _assert_openable_tif(body["dsm_geotiff_url"], expects_crs=True)
    # Georeferenced + DEM-calibrated already has a real GeoTIFF — the
    # plain no-CRS rDSM export is for the cases that DON'T, so it
    # shouldn't also appear here.
    assert body["relative_dsm_tif_url"] is None


def test_estimate_png_skips_dem_calibration(auth_headers):
    """Non-georeferenced uploads must never attempt DEM lookup, even
    though calibrate_dem defaults to true."""
    resp = client.post(
        "/api/depth/estimate",
        files={"file": ("sample.png", _fake_png_bytes(), "image/png")},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["calibration_stage"] == "relative_uncalibrated"
    assert body["absolute_height_grid"] is None
    assert body["dem_calibration_error"] is None
    assert body["relative_dsm_tif_url"] is not None


def test_estimate_uses_lora_when_available(monkeypatch, tmp_path, auth_headers):
    """Stage 2b: no adapter is trained in this repo (see
    training/README.md), so this proves the wiring end-to-end with a
    real (structurally valid, untrained) adapter — same one
    test_lora_calibration.py verifies in isolation — rather than
    leaving the router-level activation path unexercised."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForDepthEstimation

    from app.config import settings
    from app.services.calibration import lora_calibration

    base = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["neck", "head"],
    )
    get_peft_model(base, config).save_pretrained(str(tmp_path))

    monkeypatch.setattr(settings, "lora_adapter_dir", tmp_path)
    lora_calibration._peft_model = None
    lora_calibration._training_resolution = None

    resp = client.post(
        "/api/depth/estimate",
        files={"file": ("sample.png", _fake_png_bytes(), "image/png")},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["calibration_stage"] == "lora_calibrated", body.get("lora_calibration_error")
    assert body["lora_calibration_error"] is None
    assert body["absolute_height_grid"] is not None
    assert len(body["absolute_height_grid"]) == body["height_grid_resolution"]

    # LoRA-calibrated has real meters but no CRS to embed — still gets
    # the plain rDSM raster export, at full resolution (real meters).
    assert body["relative_dsm_tif_url"] is not None
    _assert_openable_tif(body["relative_dsm_tif_url"], expects_crs=False)
