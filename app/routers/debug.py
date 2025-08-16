# app/routers/debug.py
from __future__ import annotations

import os
from fastapi import APIRouter

from app.models.ha_infer import (
    USE_HA_ADAPTER,
    HA_ADAPTER_REPO,
    BASE_MODEL_ID,
    HA_RAG_PATH,
    MAUDE_LOCAL_PATH,
    _RAG_DB,
    _MAUDE_ROWS,
    CACHE_DIR,
    OFFLOAD_DIR,
)

router = APIRouter()

@router.get("/debug/ha_status")
def ha_status():
    rag_exists = os.path.exists(HA_RAG_PATH)
    rag_size = os.path.getsize(HA_RAG_PATH) if rag_exists else 0

    maude_exists = os.path.exists(MAUDE_LOCAL_PATH)
    maude_size = os.path.getsize(MAUDE_LOCAL_PATH) if maude_exists else 0

    # Mirror the main.py status fields you were reading in Colab
    return {
        "adapter_enabled": bool(USE_HA_ADAPTER),
        "adapter_repo": HA_ADAPTER_REPO,
        "base_model": BASE_MODEL_ID,
        "rag_path": HA_RAG_PATH,
        "rag_file_exists": rag_exists,
        "rag_file_size": rag_size,
        "rag_rows_loaded": len(_RAG_DB),
        "maude_local_path": MAUDE_LOCAL_PATH,
        "maude_file_exists": maude_exists,
        "maude_file_size": maude_size,
        "maude_local_rows": len(_MAUDE_ROWS),
        "hf_cache_dir": CACHE_DIR,
        "offload_dir": OFFLOAD_DIR,
        "env_sample": {
            "HF_HOME": os.getenv("HF_HOME"),
            "HF_HUB_CACHE": os.getenv("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
            "OFFLOAD_DIR": os.getenv("OFFLOAD_DIR"),
            "HA_INPUT_MAX_TOKENS": os.getenv("HA_INPUT_MAX_TOKENS"),
            "HA_MAX_NEW_TOKENS": os.getenv("HA_MAX_NEW_TOKENS"),
        },
    }
