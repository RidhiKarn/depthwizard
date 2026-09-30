"""Shared pytest fixtures across the whole backend test suite."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def isolated_users_db(tmp_path, monkeypatch):
    """Every test gets its own throwaway SQLite file for auth data, so
    test runs never read/write the real dev database."""
    monkeypatch.setattr(settings, "users_db_path", tmp_path / "test_users.db")


@pytest.fixture
def auth_headers():
    """Signs up a unique test user and returns an Authorization header,
    for hitting endpoints that require a signed-in session
    (see app/routers/auth.py::get_current_user)."""
    client = TestClient(app)
    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post("/api/auth/signup", json={"email": email, "password": "testpassword123"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
