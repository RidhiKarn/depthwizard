# Deployment target: Hugging Face Spaces (Docker SDK, free CPU tier) —
# see backend/README.md for the Space metadata this needs, and the root
# README > Deployment section for why (PyTorch + transformers need more
# RAM than most free-tier "serverless" platforms allow; HF Spaces' free
# CPU Basic tier gives 16GB RAM / 2 vCPU, plenty of headroom).
# Also runs unmodified on Render/Railway (both set $PORT; falls back to
# 7860, HF Spaces' default, when unset).
FROM python:3.12-slim

WORKDIR /app

# rasterio's manylinux wheel bundles GDAL, so no apt-get libgdal needed.
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
    TORCH_HOME=/app/data/.torch_cache

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
