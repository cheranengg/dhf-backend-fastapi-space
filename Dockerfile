# ---- Base: PyTorch + CUDA 12.1 (works with A10G on HF Spaces) ----
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Caches → /data (persisted on HF Spaces)
ENV HF_HOME=/data/hf_home \
    TRANSFORMERS_CACHE=/data/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app ./app

# HF Spaces sets $PORT; bind to all interfaces
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
