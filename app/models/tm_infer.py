# app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Optional startup hook (e.g., clear HF cache)
try:
    import startup_cleanup  # noqa: F401
except Exception:
    pass

# ---- Models (HA required, DVP/TM optional) ----
from app.models import ha_infer

_HAS_DVP = False
try:
    from app.models import dvp_infer  # type: ignore
    _HAS_DVP = True
except Exception:
    _HAS_DVP = False

_HAS_TM = False
try:
    from app.models import tm_infer  # type: ignore
    _HAS_TM = True
except Exception:
    _HAS_TM = False

# ---- Flags / Auth ----
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")
ENABLE_DVP = os.getenv("ENABLE_DVP", "0") == "1"
ENABLE_TM  = os.getenv("ENABLE_TM",  "0") == "1"
MAX_REQS   = int(os.getenv("MAX_REQS", "10"))

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

# ---------------- Health ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "USE_HA_MODEL": os.getenv("USE_HA_MODEL"),
        "BASE_MODEL_ID": os.getenv("BASE_MODEL_ID"),
        "HA_ADAPTER_REPO": os.getenv("HA_ADAPTER_REPO"),
        "ENABLE_DVP": ENABLE_DVP,
        "HAS_DVP_MODULE": _HAS_DVP,
        "ENABLE_TM": ENABLE_TM,
        "HAS_TM_MODULE": _HAS_TM,
        "MAX_REQS": MAX_REQS,
    }

# ---------------- HA ----------------
@app.post("/hazard-analysis")
def hazard_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = (payload.get("requirements") or [])[:MAX_REQS]
        ha_rows = ha_infer.ha_predict(reqs)
        return {"ok": True, "ha": ha_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HA failed: {e}")

# ---------------- DVP (model or stub) ----------------
def _dvp_stub(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    TECH = ["electrical","mechanical","flow","pressure","occlusion","accuracy","alarm"]
    VIS  = ["label","marking","display","visual","color","contrast","font"]
    out: List[Dict[str, Any]] = []
    for r in requirements or []:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "")
        vid = str(r.get("Verification ID") or r.get("verification_id") or "")
        txt = str(r.get("Requirements") or r.get("requirements") or "")
        low = txt.lower()
        if not txt or low.endswith("requirements"):
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
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = (payload.get("requirements") or [])[:MAX_REQS]
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        if ENABLE_DVP and _HAS_DVP:
            rows = dvp_infer.dvp_predict(reqs, ha_rows)  # type: ignore
        else:
            rows = _dvp_stub(reqs, ha_rows)
        return {"ok": True, "dvp": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVP failed: {e}")

# ---------------- TM (model or stub) ----------------
def _join_unique(values: List[str]) -> str:
    vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().upper() != "NA"]
    seen: List[str] = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    return ", ".join(seen) if seen else "NA"

def _tm_stub(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]], dvp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    ha_by_req: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in ha or []:
        rid = str(h.get("requirement_id") or h.get("Requirement ID") or "")
        if rid:
            ha_by_req[rid].append(h)

    dvp_by_vid: Dict[str, Dict[str, Any]] = {}
    for d in dvp or []:
        vid = str(d.get("verification_id") or d.get("Verification ID") or "")
        if vid and vid not in dvp_by_vid:
            dvp_by_vid[vid] = d

    rows: List[Dict[str, Any]] = []
    for r in (requirements or []):
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "")
        vid = str(r.get("Verification ID") or r.get("verification_id") or "")
        rtxt = str(r.get("Requirements") or r.get("requirements") or "")

        if not rtxt or rtxt.lower().endswith("requirements"):
            rows.append({
                "verification_id": vid or "NA",
                "requirement_id": rid,
                "requirements": rtxt,
                "risk_ids": "NA",
                "risks_to_health": "NA",
                "ha_risk_controls": "NA",
                "verification_method": "NA",
                "acceptance_criteria": "NA",
            })
            continue

        ha_slice = ha_by_req.get(rid, [])
        drow = dvp_by_vid.get(vid, {})

        risk_ids = _join_unique([h.get("risk_id") or h.get("Risk ID") or "" for h in ha_slice])
        risks_to_health = _join_unique([h.get("risk_to_health") or h.get("Risk to Health") or "" for h in ha_slice])
        risk_controls = _join_unique([h.get("risk_control") or h.get("HA Risk Control") or "" for h in ha_slice])
        method = drow.get("verification_method") or drow.get("Verification Method") or "TBD - Human / SME input"
        criteria = drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "TBD - Human / SME input"

        rows.append({
            "verification_id": vid,
            "requirement_id": rid,
            "requirements": rtxt,
            "risk_ids": risk_ids if risk_ids != "NA" else "TBD - Human / SME input",
            "risks_to_health": risks_to_health if risks_to_health != "NA" else "TBD - Human / SME input",
            "ha_risk_controls": risk_controls if risk_controls != "NA" else "TBD - Human / SME input",
            "verification_method": method,
            "acceptance_criteria": criteria,
        })
    return rows

def _tm_predict(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]], dvp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if ENABLE_TM and _HAS_TM:
        return tm_infer.tm_predict(requirements, ha, dvp)  # type: ignore
    return _tm_stub(requirements, ha, dvp)

# Primary TM route (what older UI expects)
@app.post("/trace-matrix")
def trace_matrix_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = (payload.get("requirements") or [])[:MAX_REQS]
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        dvp_rows: List[Dict[str, Any]] = payload.get("dvp") or []
        rows = _tm_predict(reqs, ha_rows, dvp_rows)
        return {"ok": True, "trace_matrix": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TM failed: {e}")

# Short alias (what your Streamlit is calling now)
@app.post("/tm")
def tm_alias(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    _auth(authorization)
    try:
        reqs: List[Dict[str, Any]] = (payload.get("requirements") or [])[:MAX_REQS]
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        dvp_rows: List[Dict[str, Any]] = payload.get("dvp") or []
        rows = _tm_predict(reqs, ha_rows, dvp_rows)
        return {"ok": True, "tm": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TM failed: {e}")

# ---------------- Smoke ----------------
@app.post("/debug/smoke")
def debug_smoke(authorization: str | None = Header(default=None)):
    _auth(authorization)
    reqs = [
        {"Requirement ID": "REQ-001", "Requirements": "Pump shall maintain flow accuracy within ±5%."},
        {"Requirement ID": "REQ-002", "Requirements": "Device labeling shall be legible at 30 cm and use ISO 15223-1 symbols."},
    ]
    try:
        ha_rows = ha_infer.ha_predict(reqs)
        dvp_rows = (dvp_infer.dvp_predict(reqs, ha_rows) if ENABLE_DVP and _HAS_DVP else _dvp_stub(reqs, ha_rows))  # type: ignore
        tm_rows  = _tm_predict(reqs, ha_rows, dvp_rows)
        return {"ok": True, "ha": ha_rows, "dvp": dvp_rows, "tm": tm_rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}
