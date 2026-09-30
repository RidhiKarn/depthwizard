"""
Central configuration for the DepthWizard backend.

All tunables live here so the rest of the codebase never hardcodes paths,
model names, or magic numbers.
"""
import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- General ---
    app_name: str = "DepthWizard API"
    api_prefix: str = "/api"

    # --- CORS ---
    # Stored as a plain string, not list[str]: pydantic-settings tries to
    # strict-JSON-decode any list-typed env var at the settings-SOURCE
    # level, before any field_validator runs — so a validator can't fix
    # up a plain comma-separated value, it errors out first. Using
    # validation_alias (bypassing the automatic DEPTHWIZARD_ prefix, so
    # the public env var name stays the clean DEPTHWIZARD_CORS_ORIGINS)
    # keeps the field a plain str at the source level; cors_origins below
    # is the list[str] the rest of the app actually uses.
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        validation_alias="DEPTHWIZARD_CORS_ORIGINS",
    )

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_origins_raw.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(origin) for origin in parsed]
            except json.JSONDecodeError:
                pass
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    # --- Storage ---
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    upload_dir: Path = data_dir / "uploads"
    output_dir: Path = data_dir / "outputs"

    # --- Depth model (Stage 1, primary backbone) ---
    # Depth Anything (Hugging Face `LiheYoung/depth-anything-*-hf`)
    # replaced MiDaS as Stage 1's default backbone: it's trained on 1.5M
    # labeled + 62M unlabeled images and outperforms MiDaS on every
    # published benchmark cited in the project spec (e.g. DDAD AbsRel
    # 0.230 vs 0.251, KITTI delta1 0.947 vs 0.850).
    #   - "vits" (ViT-S, 24.8M params): live web demo / real-time user
    #     uploads, where inference latency matters.
    #   - "vitl" (ViT-L, 335.3M params): offline benchmark runs / the
    #     best-accuracy figures for the report — not used on the live
    #     request path by default.
    depth_anything_live_variant: str = "vits"
    depth_anything_benchmark_variant: str = "vitl"

    # MiDaS is NOT removed — see app/services/midas_model.py. It's kept
    # as the second model in the Stage 2e confidence/uncertainty
    # ensemble (MiDaS vs Depth Anything disagreement), not yet wired
    # into the live request path.
    midas_model_type: str = "MiDaS_small"

    depth_map_max_dim: int = 512  # resize longest edge before inference
    height_grid_resolution: int = 128  # downsampled grid sent to the 3D viewer

    # --- Stage 2b: LoRA adapter (see training/train_lora.py) ---
    # If a trained adapter (adapter_config.json + weights) exists here,
    # app.services.calibration.lora_calibration.is_lora_available()
    # returns true and the router can use it. Empty/missing directory
    # is the current, honest default — no adapter has been trained yet.
    lora_adapter_dir: Path = data_dir / "lora_adapter"

    # --- Auth (email/password accounts) ---
    users_db_path: Path = data_dir / "users.db"
    # DEV DEFAULT ONLY — override via DEPTHWIZARD_JWT_SECRET in any real
    # deployment. A fixed default is fine for local/demo use (this repo
    # isn't handling anything sensitive), but must never ship as-is to a
    # publicly reachable deployment.
    jwt_secret: str = "depthwizard-dev-secret-change-me"
    jwt_expire_days: int = 7

    model_config = SettingsConfigDict(env_prefix="DEPTHWIZARD_")


settings = Settings()

# Ensure runtime directories exist at import time.
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
