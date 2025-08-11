# CUDA + Python base
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python deps first (cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY app ./app

# If you host merged checkpoints on HF Hub, they’ll be pulled at runtime
# via HF_TOKEN. Otherwise bake them into the image at /workspace/app/models/.

# Spaces provides $PORT; bind to 0.0.0.0
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
