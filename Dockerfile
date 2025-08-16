# ===== Backend Space Dockerfile =====
# Works on CPU or GPU Spaces.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system deps (git helps with HF/LFS redirects)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# -------- Hugging Face cache/offload defaults (overridable via Space Variables) --------
# Use /data (Spaces persistent volume) by default
ENV HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    HUGGINGFACE_HUB_CACHE=/data/hf/hub \
    TRANSFORMERS_CACHE=/data/hf/transformers \
    OFFLOAD_DIR=/data/offload

# Pre-create dirs (safe if already exist)
RUN mkdir -p /data/hf/hub /data/hf/transformers /data/offload || true

# Install Python deps first for better layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . /workspace

# -------- App config defaults (overridable) --------
ENV BACKEND_TOKEN=dev-token \
    ENABLE_DVP=1 \
    ENABLE_TM=1 \
    MAX_REQS=5 \
    QUICK_LIMIT=0

# Uvicorn entrypoint (Spaces provides $PORT)
EXPOSE 7860
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
