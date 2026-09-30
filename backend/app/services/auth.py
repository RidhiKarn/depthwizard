"""
Email/password accounts, so the app behaves like a real product rather
than an anonymous demo. Deliberately built on the Python standard
library only (sqlite3, hashlib, secrets) plus PyJWT (pure Python, no
compiled/native dependency — chosen after this project already hit one
painful native-dependency wall with mmcv, see README > "Why Metric3D v2
isn't integrated") rather than pulling in SQLAlchemy/passlib/etc.

Storage: SQLite by default (a single file at settings.users_db_path) —
fine for local dev and for any host with a persistent disk. When
DEPTHWIZARD_DATABASE_URL is set (a postgres://... URL), this switches to
Postgres instead, via psycopg. That's required on Vercel specifically:
serverless functions have no shared/persistent filesystem between
invocations, so a SQLite file would not reliably remember accounts from
one request to the next. Vercel's own Storage tab can provision a free
Postgres database (Neon-backed) and injects the connection string
automatically — no separate account/service needed.

Both backends share the same public functions (signup/login/etc.) and
behavior; only _connect() and the handful of places that need
dialect-specific SQL (placeholder style, autoincrement syntax,
duplicate-key detection) branch on which one is active.

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
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import jwt

from app.config import settings

PBKDF2_ITERATIONS = 260_000

# DEPTHWIZARD_DATABASE_URL is the canonical name to set. DATABASE_URL /
# POSTGRES_URL are accepted too since that's what most Postgres
# marketplace integrations (e.g. Neon, via Vercel's Storage tab) inject
# automatically — one less manual rename step when wiring a new deploy.
_DATABASE_URL = (
    os.environ.get("DEPTHWIZARD_DATABASE_URL", "").strip()
    or os.environ.get("DATABASE_URL", "").strip()
    or os.environ.get("POSTGRES_URL", "").strip()
)
_USE_POSTGRES = _DATABASE_URL.startswith("postgres://") or _DATABASE_URL.startswith(
    "postgresql://"
)


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


def _connect_sqlite() -> sqlite3.Connection:
    # Only reached when DEPTHWIZARD_DATABASE_URL is unset (local dev, or
    # any host with a real persistent disk) — never on Vercel, whose
    # filesystem is read-only outside /tmp, so this mkdir is safe here
    # but would NOT be safe done unconditionally at config.py import time.
    settings.users_db_path.parent.mkdir(parents=True, exist_ok=True)
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


def _connect_postgres():
    import psycopg

    conn = psycopg.connect(_DATABASE_URL)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    ).hex()


def signup(email: str, password: str) -> User:
    email = email.strip().lower()
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
    created_at = time.time()

    if _USE_POSTGRES:
        import psycopg

        conn = _connect_postgres()
        try:
            row = conn.execute(
                "INSERT INTO users (email, password_hash, salt, created_at) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (email, password_hash, salt.hex(), created_at),
            ).fetchone()
            conn.commit()
            return User(id=row[0], email=email)
        except psycopg.errors.UniqueViolation as exc:
            raise EmailAlreadyRegistered(f"An account with email '{email}' already exists.") from exc
        finally:
            conn.close()

    conn = _connect_sqlite()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (email, password_hash, salt.hex(), created_at),
        )
        conn.commit()
        return User(id=cursor.lastrowid, email=email)
    except sqlite3.IntegrityError as exc:
        raise EmailAlreadyRegistered(f"An account with email '{email}' already exists.") from exc
    finally:
        conn.close()


def login(email: str, password: str) -> User:
    email = email.strip().lower()

    if _USE_POSTGRES:
        conn = _connect_postgres()
        try:
            row = conn.execute(
                "SELECT id, email, password_hash, salt FROM users WHERE email = %s", (email,)
            ).fetchone()
        finally:
            conn.close()
    else:
        conn = _connect_sqlite()
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
    if _USE_POSTGRES:
        conn = _connect_postgres()
        try:
            row = conn.execute("SELECT id, email FROM users WHERE id = %s", (user_id,)).fetchone()
        finally:
            conn.close()
    else:
        conn = _connect_sqlite()
        try:
            row = conn.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
        finally:
            conn.close()
    return User(id=row[0], email=row[1]) if row else None
