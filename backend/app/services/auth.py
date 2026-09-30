"""
Email/password accounts, so the app behaves like a real product rather
than an anonymous demo. Deliberately built on the Python standard
library only (sqlite3, hashlib, secrets) plus PyJWT (pure Python, no
compiled/native dependency — chosen after this project already hit one
painful native-dependency wall with mmcv, see README > "Why Metric3D v2
isn't integrated") rather than pulling in SQLAlchemy/passlib/etc.

Storage: a single-file SQLite database at settings.users_db_path. Fine
for a hackathon-scale product; swapping to a real database later only
touches this module.

Passwords: PBKDF2-HMAC-SHA256, 260,000 iterations (OWASP's current
minimum recommendation for PBKDF2-SHA256), random 16-byte salt per user,
via hashlib.pbkdf2_hmac — no third-party crypto dependency needed.

Sessions: a signed JWT (HS256) carrying the user id + email, returned to
the client on signup/login and sent back as `Authorization: Bearer
<token>` on subsequent requests. Verified by the get_current_user
dependency (see app/routers/auth.py), which is applied to the
depth/shadow endpoints too — this isn't just a frontend-only gate.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import jwt

from app.config import settings

PBKDF2_ITERATIONS = 260_000


class AuthError(Exception):
    pass


class EmailAlreadyRegistered(AuthError):
    pass


class InvalidCredentials(AuthError):
    pass


class InvalidToken(AuthError):
    pass


@dataclass
class User:
    id: int
    email: str


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.users_db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return conn


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    ).hex()


def signup(email: str, password: str) -> User:
    email = email.strip().lower()
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)

    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (email, password_hash, salt.hex(), time.time()),
        )
        conn.commit()
        return User(id=cursor.lastrowid, email=email)
    except sqlite3.IntegrityError as exc:
        raise EmailAlreadyRegistered(f"An account with email '{email}' already exists.") from exc
    finally:
        conn.close()


def login(email: str, password: str) -> User:
    email = email.strip().lower()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, email, password_hash, salt FROM users WHERE email = ?", (email,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise InvalidCredentials("Incorrect email or password.")

    user_id, stored_email, stored_hash, salt_hex = row
    candidate_hash = _hash_password(password, bytes.fromhex(salt_hex))
    if not secrets.compare_digest(candidate_hash, stored_hash):
        raise InvalidCredentials("Incorrect email or password.")

    return User(id=user_id, email=stored_email)


def create_access_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "exp": int(time.time()) + settings.jwt_expire_days * 86400,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> User:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise InvalidToken(f"Invalid or expired session token: {exc}") from exc
    return User(id=int(payload["sub"]), email=payload["email"])


def get_user_by_id(user_id: int) -> Optional[User]:
    conn = _connect()
    try:
        row = conn.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return User(id=row[0], email=row[1]) if row else None
