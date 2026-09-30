# Deployment target: Render (free web service, Docker runtime) — see
# backend/README.md and root README > Deployment. Hugging Face Spaces
# was the original plan, but Docker Spaces now require a paid plan
# there; this Dockerfile is provider-agnostic and runs unmodified on
# Render/Railway/Fly/HF Spaces (all set $PORT; falls back to 7860 when
# unset).
FROM python:3.12-slim

WORKDIR /app

# rasterio's manylinux wheel bundles GDAL itself, so no apt-get libgdal
# needed — but it does NOT bundle libexpat, since the manylinux wheel
# policy assumes that specific library is always present on the host.
# python:3.12-slim strips it out (that's what "slim" means), which
# crashed every request that touched rasterio with "ImportError:
# libexpat.so.1: cannot open shared object file" — confirmed via actual
# deploy logs, not a guess.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces (and most PaaS runtimes) run containers as a non-root user
# by convention; matches their expectations and is good practice anyway.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Model weights (Depth Anything, MiDaS) download into these caches on
# first request — not baked into the image, so first request after a
# cold start is slow (see root README > Deployment for the "precompute
# a sample result" mitigation this implies for a smooth demo).
ENV HF_HOME=/app/data/.hf_cache \
    TORCH_HOME=/app/data/.torch_cache \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
