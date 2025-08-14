# app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, List

# Clean HF cache on cold start (prints a line in logs)
try:
    import startup_cleanup  # noqa: F401
except Exception:
    pass

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Model modules (HA, DVP)
from app.models import ha_infer, dvp_infer  # add tm_infer when you enable TM


# -------------------------------------------------------------------
# Auth / App setup
# -------------------------------------------------------------------
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")

app = FastAPI(title="DHF Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _auth(authorization: str = Header(default="")):
    """Simple bearer-token auth shared by all endpoints."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != BACKEND_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root():
    # Keep it terse to avoid leaking info publicly
    return {"detail": "Not Found"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug/flags")
def debug_flags(_: None = Header(default=None)):
    """Quick snapshot of important flags/env used by the backend."""
    flags = {
        "USE_HA_MODEL": os.getenv("USE_HA_MODEL", "1"),
        "USE_DVP_MODEL": os.getenv("USE_DVP_MODEL", "1"),
        "USE_TM_MODEL": os.getenv("USE_TM_MODEL", "0"),
        "HA_MODEL_MERGED_DIR": os.getenv("HA_MODEL_MERGED_DIR", ""),
        "DVP_MODEL_DIR": os.getenv("DVP_MODEL_DIR", ""),
        "TM_MODEL_DIR": os.getenv("TM_MODEL_DIR", ""),
        "HF_HOME": os.getenv("HF_HOME", ""),
        "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE", ""),
        "HF_HUB_ENABLE_HF_TRANSFER": os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0"),
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "8"),
        # don’t echo tokens
    }
    return flags


@app.post("/hazard-analysis")
def hazard_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str = Header(default=""),
):
    """
    Request body:
    {
      "requirements": [
        {"Requirement ID": "...", "Requirements": "..."},
        ...
      ]
    }
    """
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        ha_rows = ha_infer.ha_predict(reqs)
        return {"ok": True, "ha": ha_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HA failed: {e}")


@app.post("/dvp")
def dvp_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str = Header(default=""),
):
    """
    Request body:
    {
      "requirements": [...],   # same shape as hazard-analysis
      "ha": [...]              # output from hazard-analysis (optional but helpful)
    }
    """
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        dvp_rows = dvp_infer.dvp_predict(reqs, ha_rows)
        return {"ok": True, "dvp": dvp_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVP failed: {e}")


@app.post("/debug/smoke")
def debug_smoke(
    authorization: str = Header(default=""),
):
    """
    Quick smoke test that runs both HA and DVP on a few synthetic requirements.
    """
    _auth(authorization)
    reqs = [
        {
            "Requirement ID": "REQ-001",
            "Requirements": "The pump shall maintain flow accuracy within ±5% from set value across 0.1–999 ml/hr.",
        },
        {
            "Requirement ID": "REQ-002",
            "Requirements": "Device labeling shall be legible at 30 cm and use ISO 15223-1 symbols.",
        },
        {
            "Requirement ID": "REQ-003",
            "Requirements": "Electrical insulation shall comply with IEC 60601-1 dielectric strength requirements.",
        },
    ]
    try:
        ha_rows = ha_infer.ha_predict(reqs)
        dvp_rows = dvp_infer.dvp_predict(reqs, ha_rows)
        return {"ok": True, "ha": ha_rows, "dvp": dvp_rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}
