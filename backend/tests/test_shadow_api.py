"""
Smoke test for Stage 2c (POST /api/shadow/estimate). No ML model
involved — fast, no network required.
"""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _image_with_shadow_bytes() -> bytes:
    """200x200 white image with a black rectangle standing in for a cast
    shadow, extending downward (image +y) from roughly y=100 to y=180 —
    matches a sun coming from due north (azimuth 0), whose shadow falls
    due south (+y in image space)."""
    arr = np.full((200, 200, 3), 255, dtype=np.uint8)
    arr[100:180, 80:120] = 0
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_shadow_estimate_requires_auth():
    resp = client.post(
        "/api/shadow/estimate",
        files={"file": ("scene.png", _image_with_shadow_bytes(), "image/png")},
        data={"sun_elevation_deg": "45", "sun_azimuth_deg": "0", "gsd_meters_per_px": "1.0"},
    )
    assert resp.status_code == 401


def test_shadow_estimate_requires_gsd_for_non_georeferenced(auth_headers):
    resp = client.post(
        "/api/shadow/estimate",
        files={"file": ("scene.png", _image_with_shadow_bytes(), "image/png")},
        data={"sun_elevation_deg": "45", "sun_azimuth_deg": "0"},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert "gsd_meters_per_px" in resp.json()["detail"]


def test_shadow_estimate_end_to_end(auth_headers):
    resp = client.post(
        "/api/shadow/estimate",
        files={"file": ("scene.png", _image_with_shadow_bytes(), "image/png")},
        data={
            "sun_elevation_deg": "45",
            "sun_azimuth_deg": "0",
            "gsd_meters_per_px": "1.0",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["candidate_count"] >= 1
    assert body["gsd_source"] == "user_supplied"

    # Image dimensions (fix: numbered markers on the photo) — the
    # frontend needs these to place each estimate's marker correctly.
    assert body["image_width"] == 200
    assert body["image_height"] == 200

    top = body["estimates"][0]
    # ~80px shadow at 1 m/px and 45 degree sun elevation -> ~80m (tan(45)=1).
    assert 50 <= top["shadow_length_px"] <= 100
    assert 50 <= top["estimated_height_m"] <= 100
    # Marker position should fall within the known shadow rectangle
    # (arr[100:180, 80:120] = 0), not just anywhere in the image.
    assert 80 <= top["x"] <= 120
    assert 100 <= top["y"] <= 180


def test_shadow_estimate_rejects_sun_below_horizon(auth_headers):
    resp = client.post(
        "/api/shadow/estimate",
        files={"file": ("scene.png", _image_with_shadow_bytes(), "image/png")},
        data={
            "sun_elevation_deg": "0",
            "sun_azimuth_deg": "0",
            "gsd_meters_per_px": "1.0",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 400
