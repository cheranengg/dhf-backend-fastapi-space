# ===== Backend Space Dockerfile =====
# Works on CPU or GPU Spaces. Uses Python 3.11 slim base.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (git is handy for HF downloads that redirect to git-lfs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# -------- Hugging Face cache (writable) --------
# Using a container-local cache avoids permission errors under /home/user
ENV HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf
RUN mkdir -p /cache/hf

WORKDIR /workspace

# Install Python deps first (better layer caching)
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . /workspace

# -------- Useful defaults (can be overridden in Space “Variables”) --------
# Bearer token the Streamlit front-end uses to call this backend
ENV BACKEND_TOKEN=dev-token

# Model toggles (you can override in Space “Variables” UI)
# 0 = use rule-based/heuristics; 1 = load your fine-tuned models
ENV USE_HA_MODEL=0 \
    USE_DVP_MODEL=1 \
    USE_TM_MODEL=1

# Where to load your merged models from (Hugging Face Hub repos or local paths)
# These match your uploaded repos shown in screenshots
ENV DVP_MODEL_DIR=cheranengg/dhf-dvp-merged \
    TM_MODEL_DIR=cheranengg/dhf-tm-merged \
    HA_MODEL_MERGED_DIR=cheranengg/dhf-ha-merged

# Uvicorn entrypoint (Spaces passes $PORT)
EXPOSE 7860
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
