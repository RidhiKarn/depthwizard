"""FastAPI entrypoint. Run locally with:

    uvicorn app.main:app --reload --port 8000

from inside backend/ (with the venv from requirements.txt activated).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import auth, depth, shadow

app = FastAPI(
    title=settings.app_name,
    description=(
        "DepthWizard — single-view satellite/aerial image to elevation map. "
        "SIH26175. Stage 1 (Depth Anything relative depth), Stage 2a (DEM "
        "calibration), and Stage 2c (shadow cross-check) are live. LoRA "
        "(2b) and Metric3D v2 (2d) are staged in app/services/ but not yet "
        "wired in — see README > Roadmap."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(depth.router, prefix=settings.api_prefix)
app.include_router(shadow.router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {"name": settings.app_name, "status": "ok", "docs": "/docs"}
