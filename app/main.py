# app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Optional startup hook to clear HF cache (/tmp/hf) each boot.
try:  # noqa: SIM105
    import startup_cleanup  # type: ignore  # noqa: F401
except Exception:
    pass

# Model modules (from app/models/*_infer.py)
from app.models import ha_infer, dvp_infer, tm_infer  # noqa: E402

# -------------------------------------------------------------------
# Auth / App setup
# -------------------------------------------------------------------
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")

# Soft row caps (modules already cap, but we keep here for consistency)
HA_MAX_ROWS = int(os.getenv("HA_MAX_ROWS", "10"))
DVP_MAX_ROWS = int(os.getenv("DVP_MAX_ROWS", "10"))
TM_MAX_ROWS = int(os.getenv("TM_MAX_ROWS", "10"))

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
    return {"ok": True}


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
        ha_rows = ha_infer.ha_predict(reqs)
        return {"ok": True, "ha": ha_rows[:HA_MAX_ROWS]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HA failed: {e}")


@app.post("/dvp")
def dvp_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
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
        return {"ok": True, "dvp": dvp_rows[:DVP_MAX_ROWS]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVP failed: {e}")


@app.post("/tm")
def tm_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    """
    Request body:
    {
      "requirements": [...],
      "ha": [...],
      "dvp": [...]
    }
    """
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        dvp_rows: List[Dict[str, Any]] = payload.get("dvp") or []
        tm_rows = tm_infer.tm_predict(reqs, ha_rows, dvp_rows)
        return {"ok": True, "tm": tm_rows[:TM_MAX_ROWS]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TM failed: {e}")


@app.post("/debug/smoke")
def debug_smoke(
    authorization: str | None = Header(default=None),
):
    """
    Quick smoke test that runs HA, DVP, and TM on a tiny synthetic requirement set.
    """
    _auth(authorization)
    reqs = [
        {
            "Requirement ID": "REQ-001",
            "Requirements": "Pump shall maintain flow accuracy within ±5%.",
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
        ha_rows = ha_infer.ha_predict(reqs)[:HA_MAX_ROWS]
        dvp_rows = dvp_infer.dvp_predict(reqs, ha_rows)[:DVP_MAX_ROWS]
        tm_rows = tm_infer.tm_predict(reqs, ha_rows, dvp_rows)[:TM_MAX_ROWS]
        return {"ok": True, "ha": ha_rows, "dvp": dvp_rows, "tm": tm_rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}
