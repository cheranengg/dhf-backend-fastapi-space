# app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Optional startup hook to clear HF cache (/tmp/hf) each boot.
try:
    import startup_cleanup  # noqa: F401
except Exception:
    pass

# Import model modules
from app.models import ha_infer  # we'll add dvp_infer later when you flip it on

# -------------------------------------------------------------------
# Auth / App setup
# -------------------------------------------------------------------
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")

app = FastAPI(title="DHF Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _auth(authorization: str | None):
    """Simple bearer-token auth shared by all endpoints."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != BACKEND_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    # Quick peek at HA env for debugging
    return {
        "ok": True,
        "USE_HA_MODEL": os.getenv("USE_HA_MODEL"),
        "BASE_MODEL_ID": os.getenv("BASE_MODEL_ID"),
        "HA_ADAPTER_REPO": os.getenv("HA_ADAPTER_REPO"),
        "LOAD_4BIT": os.getenv("LOAD_4BIT"),
        "FORCE_CPU": os.getenv("FORCE_CPU"),
    }


@app.post("/hazard-analysis")
def hazard_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
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
        # optional truncation for speed: keep first N requirements
        max_req = int(os.getenv("MAX_REQS", "10"))
        reqs = reqs[:max_req]
        ha_rows = ha_infer.ha_predict(reqs)
        return {"ok": True, "ha": ha_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HA failed: {e}")


@app.post("/debug/smoke")
def debug_smoke(
    authorization: str | None = Header(default=None),
):
    """
    Quick smoke test that runs HA on a few synthetic requirements.
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
    ]
    try:
        ha_rows = ha_infer.ha_predict(reqs)
        return {"ok": True, "ha": ha_rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}
