"""
Email/password accounts (see app/services/auth.py for the storage/
hashing/token details). `get_current_user` is imported by
routers/depth.py and routers/shadow.py to actually require a signed-in
session on the product's core endpoints — this is a real gate, not just
a frontend-only decoration.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from app.schemas import AuthResponse, LoginRequest, MeResponse, SignupRequest
from app.services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


async def get_current_user(authorization: str = Header(None)) -> auth_service.User:
    """FastAPI dependency: require `Authorization: Bearer <token>`,
    used on any endpoint that should only work for signed-in users."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated. Please sign in.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        return auth_service.decode_access_token(token)
    except auth_service.InvalidToken as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/signup", response_model=AuthResponse)
async def signup(body: SignupRequest) -> AuthResponse:
    if "@" not in body.email or "." not in body.email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    try:
        user = auth_service.signup(body.email, body.password)
    except auth_service.EmailAlreadyRegistered as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    token = auth_service.create_access_token(user)
    return AuthResponse(access_token=token, email=user.email)


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest) -> AuthResponse:
    try:
        user = auth_service.login(body.email, body.password)
    except auth_service.InvalidCredentials as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    token = auth_service.create_access_token(user)
    return AuthResponse(access_token=token, email=user.email)


@router.get("/me", response_model=MeResponse)
async def me(current_user: auth_service.User = Depends(get_current_user)) -> MeResponse:
    return MeResponse(email=current_user.email)
