"""
Email/password accounts (app/services/auth.py + app/routers/auth.py).
No mocks — real SQLite writes (to an isolated per-test DB, see
conftest.py::isolated_users_db), real PBKDF2 hashing, real JWT
encode/decode.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_signup_then_me():
    resp = client.post(
        "/api/auth/signup", json={"email": "new.user@example.com", "password": "correcthorse123"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["email"] == "new.user@example.com"
    assert body["access_token"]

    me_resp = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}
    )
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "new.user@example.com"


def test_signup_rejects_short_password():
    resp = client.post("/api/auth/signup", json={"email": "a@example.com", "password": "short"})
    assert resp.status_code == 400


def test_signup_rejects_invalid_email():
    resp = client.post(
        "/api/auth/signup", json={"email": "not-an-email", "password": "correcthorse123"}
    )
    assert resp.status_code == 400


def test_signup_rejects_duplicate_email():
    payload = {"email": "dupe@example.com", "password": "correcthorse123"}
    first = client.post("/api/auth/signup", json=payload)
    assert first.status_code == 200
    second = client.post("/api/auth/signup", json=payload)
    assert second.status_code == 409


def test_login_success():
    client.post("/api/auth/signup", json={"email": "login@example.com", "password": "correcthorse123"})
    resp = client.post(
        "/api/auth/login", json={"email": "login@example.com", "password": "correcthorse123"}
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == "login@example.com"


def test_login_wrong_password():
    client.post("/api/auth/signup", json={"email": "wrongpw@example.com", "password": "correcthorse123"})
    resp = client.post(
        "/api/auth/login", json={"email": "wrongpw@example.com", "password": "wrongpassword"}
    )
    assert resp.status_code == 401


def test_login_unknown_email():
    resp = client.post(
        "/api/auth/login", json={"email": "nobody@example.com", "password": "correcthorse123"}
    )
    assert resp.status_code == 401


def test_me_rejects_missing_token():
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_me_rejects_garbage_token():
    resp = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401
