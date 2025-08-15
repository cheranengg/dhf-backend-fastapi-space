# app/routers/debug.py
from fastapi import APIRouter
import os, json

from app.models.ha_infer import (
    USE_HA_ADAPTER, USE_HA_MODEL, HA_ADAPTER_REPO, HA_MODEL_DIR,
    RAG_SYNTHETIC_PATH, _RAG_DB, _token_cache_kwargs, CACHE_DIR, BASE_MODEL_ID
)

router = APIRouter()

@router.get("/debug/ha_status")
def ha_status():
    exists = os.path.exists(RAG_SYNTHETIC_PATH)
    size = 0
    try:
        if exists:
            size = os.path.getsize(RAG_SYNTHETIC_PATH)
    except Exception:
        pass

    return {
        "adapter_enabled": USE_HA_ADAPTER,
        "adapter_repo": HA_ADAPTER_REPO,
        "merged_enabled": USE_HA_MODEL,
        "merged_dir": HA_MODEL_DIR,
        "base_model": BASE_MODEL_ID,
        "rag_path": RAG_SYNTHETIC_PATH,
        "rag_file_exists": exists,
        "rag_file_size": size,
        "rag_rows_loaded": len(_RAG_DB),
        "hf_cache_dir": CACHE_DIR,
        "hf_token_provided": bool(_token_cache_kwargs().get("token")),
        "env_sample": {
            "USE_HA_ADAPTER": os.getenv("USE_HA_ADAPTER"),
            "USE_HA_MODEL": os.getenv("USE_HA_MODEL"),
            "HA_RAG_PATH": os.getenv("HA_RAG_PATH"),
        },
    }
