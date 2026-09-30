# DepthWizard — backend

FastAPI inference service for DepthWizard (SIH26175). See the root
project README for full architecture, datasets, and calibration
details.

Deployed as a Python Vercel Function (zero-config: Vercel auto-detects
`app/main.py`'s `app` object as the entrypoint from `requirements.txt` +
the file layout — see `vercel.json` for the only extra config, which
just trims dev-only files out of the deployed bundle). This is a
separate Vercel project from the frontend, with its own URL — the
frontend's `NEXT_PUBLIC_API_BASE_URL` env var points at it.

Previously tried Render's and Hugging Face Spaces' free tiers first —
see git history / `Dockerfile` (still present, unused by the current
deploy, kept in case a container host is ever needed again) — but
Render's 512MB free-tier RAM cap couldn't hold PyTorch's baseline
footprint even after switching the live inference path to ONNX Runtime,
and Hugging Face now requires a paid plan for Docker Spaces. Vercel's
Python Functions give 2GB RAM on the free Hobby tier, which is a
comfortable margin.

Two consequences of being a serverless function specifically (no
filesystem shared between requests):
  - Auth (`app/services/auth.py`) uses Postgres, not a local SQLite
    file, when `DEPTHWIZARD_DATABASE_URL` (or `DATABASE_URL`/
    `POSTGRES_URL`) is set — see that module's docstring. Falls back to
    SQLite when unset, which is what local dev/tests use.
  - Generated outputs (depth heatmap PNGs, GeoTIFF/.tif DSM exports) are
    returned as inline base64 `data:` URIs embedded directly in the JSON
    response, not saved to disk + served via a separate download URL —
    see `app/routers/depth.py`'s `_save_png`/`_save_plain_tif`/
    `_save_geotiff`. This also keeps responses under Vercel's 4.5MB
    function response-body limit (GeoTIFF exports use DEFLATE
    compression specifically to help with that).

`requirements.txt` is the lean set that actually deploys (no
torch/transformers/peft — the Stage 2e confidence ensemble and Stage 2b
LoRA calibration are opt-in/not-yet-trained anyway, and lazy-imported so
a normal request never touches them regardless). `requirements-dev.txt`
layers those on top for local development and the full pytest suite.

## Required environment variables

Set these in the Vercel project's Settings → Environment Variables:

| Name | Value |
|---|---|
| `DEPTHWIZARD_JWT_SECRET` | any long random string — must be set, the code default is dev-only |
| `DEPTHWIZARD_CORS_ORIGINS` | your deployed frontend URL, e.g. `https://your-app.vercel.app` (comma-separate multiple) |
| `DEPTHWIZARD_DATABASE_URL` | a Postgres connection string (e.g. from a Neon integration via Vercel's Storage tab) — required in production; local dev/tests work fine without it |

## Local development

See the main repo's `backend/` folder and its own setup instructions —
this file is specifically about the deployed service.
