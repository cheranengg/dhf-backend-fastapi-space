# app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Optional startup hook (cache clean)
try:
    import startup_cleanup  # noqa: F401
except Exception:
    pass

# Models
from app.models import ha_infer

# DVP is optional; import guarded
_HAS_DVP = False
try:
    from app.models import dvp_infer  # type: ignore
    _HAS_DVP = True
except Exception:
    _HAS_DVP = False

# Flags
ENABLE_DVP = os.getenv("ENABLE_DVP", "0") == "1"

# Auth / App setup
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
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != BACKEND_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")

@app.get("/health")
def health():
    return {
        "ok": True,
        # HA
        "USE_HA_MODEL": os.getenv("USE_HA_MODEL"),
        "BASE_MODEL_ID": os.getenv("BASE_MODEL_ID"),
        "HA_ADAPTER_REPO": os.getenv("HA_ADAPTER_REPO"),
        # DVP
        "ENABLE_DVP": ENABLE_DVP,
        "HAS_DVP_MODULE": _HAS_DVP,
    }

# -------------------- HA --------------------
@app.post("/hazard-analysis")
def hazard_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    """
    Body: {"requirements":[{"Requirement ID":"...","Requirements":"..."}, ...]}
    """
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        max_req = int(os.getenv("MAX_REQS", "10"))
        reqs = reqs[:max_req]
        ha_rows = ha_infer.ha_predict(reqs)
        return {"ok": True, "ha": ha_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HA failed: {e}")

# -------------------- DVP --------------------
def _dvp_stub(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Safe fallback so the UI doesn't 404 while DVP model is off."""
    TECH = ["electrical","mechanical","flow","pressure","occlusion","accuracy","alarm"]
    VIS  = ["label","marking","display","visual","color","contrast","font"]
    out: List[Dict[str, Any]] = []
    for r in requirements or []:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "")
        vid = str(r.get("Verification ID") or r.get("verification_id") or "")
        txt = str(r.get("Requirements") or r.get("requirements") or "")
        low = txt.lower()
        if not txt or txt.lower().endswith("requirements"):
            out.append({
                "verification_id": vid, "requirement_id": rid, "requirements": txt,
                "verification_method": "NA", "sample_size": "NA",
                "test_procedure": "NA", "acceptance_criteria": "NA",
            })
            continue
        method = "Physical Testing" if any(k in low for k in TECH) else ("Visual Inspection" if any(k in low for k in VIS) else "Physical Inspection")
        sample = "30"
        ac = "TBD"
        if "flow" in low: ac = "±5% from set value (IEC 60601-2-24)"
        elif "occlusion" in low: ac = "Alarm ≤ 30 s at 100 kPa back pressure (IEC 60601-2-24)"
        bullets = (
            "- Verify at three setpoints; record measured vs setpoint (n=3).\n"
            "- Repeatability across 5 cycles; compute deviation and std dev.\n"
            "- Boundary test at min/max; record alarms and pass/fail.\n"
            "- Capture equipment IDs and calibration; attach raw data."
        )
        out.append({
            "verification_id": vid, "requirement_id": rid, "requirements": txt,
            "verification_method": method, "sample_size": sample,
            "test_procedure": bullets, "acceptance_criteria": ac,
        })
    return out

@app.post("/dvp")
def dvp_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    """
    Body:
    {
      "requirements": [...],   # same shape as HA input
      "ha": [...]              # optional HA rows
    }
    """
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        max_req = int(os.getenv("MAX_REQS", "10"))
        reqs = reqs[:max_req]

        # If model is enabled and module exists, use it; else stub.
        if ENABLE_DVP and _HAS_DVP:
            rows = dvp_infer.dvp_predict(reqs, ha_rows)  # type: ignore
        else:
            rows = _dvp_stub(reqs, ha_rows)
        return {"ok": True, "dvp": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVP failed: {e}")

# -------------------- Smoke --------------------
@app.post("/debug/smoke")
def debug_smoke(authorization: str | None = Header(default=None)):
    _auth(authorization)
    reqs = [
        {"Requirement ID": "REQ-001", "Requirements": "Pump shall maintain flow accuracy within ±5%."},
        {"Requirement ID": "REQ-002", "Requirements": "Device labeling shall be legible at 30 cm and use ISO 15223-1 symbols."},
    ]
    try:
        ha_rows = ha_infer.ha_predict(reqs)
        if ENABLE_DVP and _HAS_DVP:
            dvp_rows = dvp_infer.dvp_predict(reqs, ha_rows)  # type: ignore
        else:
            dvp_rows = _dvp_stub(reqs, ha_rows)
        return {"ok": True, "ha": ha_rows, "dvp": dvp_rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}
