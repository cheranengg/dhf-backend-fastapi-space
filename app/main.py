# app/main.py
from __future__ import annotations
import os, traceback, json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.utils.io_schemas import (
    RequirementInput, HazardAnalysisRow, HazardAnalysisOutput,
    DvpRow, DvpOutput, TraceMatrixRow, TraceMatrixOutput
)
from app.utils.guardrails import (
    DEFAULT_ALLOWED_METHODS, sanitize_text, ensure_tbd,
    normalize_tm_row, guard_tm_rows
)
from app.models import ha_infer, dvp_infer, tm_infer

# -------- Security --------
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "dev-token")

def require_auth(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != BACKEND_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

# -------- CORS --------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS_LIST = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]

# -------- App --------
app = FastAPI(title="DHF Backend (HA/DVP/TM)", version=os.getenv("APP_VERSION", "1.0.0"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Helpers --------
def _reqs_to_dicts(reqs: List[RequirementInput | Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize incoming requirements to canonical dicts the models expect.
    - If incoming objects already have 'Requirements', pass through (handled by caller).
    - If Pydantic RequirementInput objects, preserve Verification ID if present in dict form.
    """
    out: List[Dict[str, Any]] = []
    for r in reqs:
        if isinstance(r, dict):
            rid = sanitize_text(str(r.get("Requirement ID") or r.get("requirement_id") or ""))
            vid = sanitize_text(str(r.get("Verification ID") or r.get("verification_id") or ""))
            rtxt = sanitize_text(str(r.get("Requirements") or r.get("requirements") or ""))
        else:
            # RequirementInput dataclass-like
            rid = sanitize_text(r.requirement_id or "")
            vid = ""  # RequirementInput schema usually doesn't include VID
            rtxt = sanitize_text(r.requirements or "")
        out.append({"Requirement ID": rid, "Verification ID": vid, "Requirements": rtxt})
    return out

def _normalize_ha_row(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "requirement_id": sanitize_text(r.get("requirement_id") or r.get("Requirement ID") or ""),
        "risk_id": ensure_tbd(r.get("risk_id") or r.get("Risk ID")),
        "hazard": ensure_tbd(r.get("hazard") or r.get("Hazard")),
        "hazardous_situation": ensure_tbd(r.get("hazardous_situation") or r.get("Hazardous situation") or r.get("Hazardous Situation")),
        "risk_to_health": ensure_tbd(r.get("risk_to_health") or r.get("Risk to Health")),
        "harm": ensure_tbd(r.get("harm") or r.get("Harm")),
        "sequence_of_events": ensure_tbd(r.get("sequence_of_events") or r.get("Sequence of Events")),
        "severity_of_harm": ensure_tbd(r.get("severity_of_harm") or r.get("Severity of Harm")),
        "p0": ensure_tbd(r.get("p0") or r.get("P0")),
        "p1": ensure_tbd(r.get("p1") or r.get("P1")),
        "poh": ensure_tbd(r.get("poh") or r.get("PoH")),
        "risk_index": ensure_tbd(r.get("risk_index") or r.get("Risk Index")),
        "risk_control": ensure_tbd(r.get("risk_control") or r.get("Risk Control") or r.get("HA Risk Control")),
    }

def _normalize_dvp_row(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verification_id": sanitize_text(r.get("verification_id") or r.get("Verification ID") or ""),
        "requirement_id": sanitize_text(r.get("requirement_id") or r.get("Requirement ID") or ""),
        "requirements": ensure_tbd(r.get("requirements") or r.get("Requirements")),
        "verification_method": ensure_tbd(r.get("verification_method") or r.get("Verification Method")),
        "sample_size": sanitize_text(r.get("sample_size") or r.get("Sample Size") or ""),
        "acceptance_criteria": ensure_tbd(r.get("acceptance_criteria") or r.get("Acceptance Criteria")),
        "test_procedure": ensure_tbd(r.get("test_procedure") or r.get("Test Procedure")),
    }

def _normalize_tm_row_api(r: Dict[str, Any]) -> Dict[str, Any]:
    base = normalize_tm_row({
        "verification_id": r.get("verification_id") or r.get("Verification ID"),
        "requirement_id": r.get("requirement_id") or r.get("Requirement ID"),
        "requirements": r.get("requirements") or r.get("Requirements"),
        "risk_ids": r.get("risk_ids") or r.get("Risk ID(s)") or r.get("risk_id"),
        "risks_to_health": r.get("risks_to_health") or r.get("Risk to Health"),
        "ha_risk_controls": r.get("ha_risk_controls") or r.get("HA Risk Control(s)") or r.get("risk_control"),
        "verification_method": r.get("verification_method") or r.get("Verification Method"),
        "acceptance_criteria": r.get("acceptance_criteria") or r.get("Acceptance Criteria"),
    })
    base["risk_ids"] = base["risk_ids"].replace(" ,", ",").replace(", ,", ",").strip()
    base["ha_risk_controls"] = base["ha_risk_controls"].replace(" ,", ",").replace(", ,", ",").strip()
    return base

# -------- Routes --------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ready")
def ready():
    """
    Lightweight readiness check:
    - Ensures flags/environment are readable
    - Does NOT run heavy generation
    """
    return {
        "ok": True,
        "device": ("cuda" if os.getenv("USE_HA_MODEL") == "1" or os.getenv("USE_DVP_MODEL") == "1" or os.getenv("USE_TM_MODEL") == "1" else "cpu"),
        "flags": {
            "USE_HA_MODEL": os.getenv("USE_HA_MODEL", "0"),
            "USE_DVP_MODEL": os.getenv("USE_DVP_MODEL", "0"),
            "USE_TM_MODEL": os.getenv("USE_TM_MODEL", "0"),
        }
    }

@app.get("/version")
def version():
    return {"version": app.version}

@app.get("/debug/flags", dependencies=[Depends(require_auth)])
def debug_flags():
    return {
        "USE_HA_MODEL": os.getenv("USE_HA_MODEL", "0"),
        "USE_DVP_MODEL": os.getenv("USE_DVP_MODEL", "0"),
        "USE_TM_MODEL": os.getenv("USE_TM_MODEL", "0"),
        "HA_MODEL_MERGED_DIR": os.getenv("HA_MODEL_MERGED_DIR", ""),
        "DVP_MODEL_DIR": os.getenv("DVP_MODEL_DIR", ""),
        "TM_MODEL_DIR": os.getenv("TM_MODEL_DIR", ""),
        "LORA_HA_DIR": os.getenv("LORA_HA_DIR", ""),
        "BASE_MODEL_ID": os.getenv("BASE_MODEL_ID", ""),
        "HF_CACHE_DIR": os.getenv("HF_CACHE_DIR", ""),
    }

@app.post("/debug/smoke", dependencies=[Depends(require_auth)])
def debug_smoke():
    try:
        reqs = [
            {"Requirement ID": "REQ-001", "Verification ID": "VER-001", "Requirements": "Detect air-in-line within 1 second"},
            {"Requirement ID": "REQ-002", "Verification ID": "VER-002", "Requirements": "Stop infusion on occlusion"},
        ]
        ha_rows = ha_infer.ha_predict(reqs)
        dvp_rows = dvp_infer.dvp_predict(reqs, ha_rows)
        tm_rows  = tm_infer.tm_predict(reqs, ha_rows, dvp_rows)
        return {"ok": True, "sizes": {"ha": len(ha_rows), "dvp": len(dvp_rows), "tm": len(tm_rows)}}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

@app.post("/hazard-analysis", response_model=HazardAnalysisOutput, dependencies=[Depends(require_auth)])
def hazard_analysis(payload: Dict[str, Any]):
    try:
        raw_reqs = payload.get("requirements", [])
        if not isinstance(raw_reqs, list) or not raw_reqs:
            raise HTTPException(400, "`requirements` must be a non-empty list")
        # pass-through if canonical; else normalize
        reqs_dict = raw_reqs if isinstance(raw_reqs[0], dict) and "Requirements" in (raw_reqs[0] or {}) else _reqs_to_dicts(raw_reqs)
        ha_rows_raw: List[Dict[str, Any]] = ha_infer.ha_predict(reqs_dict)
        ha_rows_norm = [_normalize_ha_row(r) for r in ha_rows_raw]
        return {"ha": [HazardAnalysisRow(**row) for row in ha_rows_norm]}  # type: ignore
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"HA failed: {e}")

@app.post("/dvp", response_model=DvpOutput, dependencies=[Depends(require_auth)])
def dvp(payload: Dict[str, Any]):
    try:
        raw_reqs = payload.get("requirements", [])
        ha_rows = payload.get("ha", [])
        if not isinstance(raw_reqs, list) or not raw_reqs:
            raise HTTPException(400, "`requirements` must be a non-empty list")
        reqs_dict = raw_reqs if isinstance(raw_reqs[0], dict) and "Requirements" in (raw_reqs[0] or {}) else _reqs_to_dicts(raw_reqs)
        dvp_rows_raw: List[Dict[str, Any]] = dvp_infer.dvp_predict(reqs_dict, ha_rows)
        dvp_rows_norm = [_normalize_dvp_row(r) for r in dvp_rows_raw]
        return {
            "dvp": [DvpRow(
                verification_id=row["verification_id"],
                requirement_id=row.get("requirement_id"),
                requirements=row.get("requirements"),
                verification_method=row["verification_method"],
                sample_size=int(row["sample_size"]) if row.get("sample_size", "").isdigit() else None,
                acceptance_criteria=row["acceptance_criteria"],
            ) for row in dvp_rows_norm]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"DVP failed: {e}")

@app.post("/trace-matrix", response_model=TraceMatrixOutput, dependencies=[Depends(require_auth)])
def trace_matrix(payload: Dict[str, Any]):
    try:
        raw_reqs = payload.get("requirements", [])
        ha_rows = payload.get("ha", [])
        dvp_rows = payload.get("dvp", [])
        if not isinstance(raw_reqs, list) or not raw_reqs:
            raise HTTPException(400, "`requirements` must be a non-empty list")
        reqs_dict = raw_reqs if isinstance(raw_reqs[0], dict) and "Requirements" in (raw_reqs[0] or {}) else _reqs_to_dicts(raw_reqs)
        tm_rows_raw: List[Dict[str, Any]] = tm_infer.tm_predict(reqs_dict, ha_rows, dvp_rows)
        tm_rows_norm = [_normalize_tm_row_api(r) for r in tm_rows_raw]
        result = guard_tm_rows(tm_rows_norm, allowed_methods=DEFAULT_ALLOWED_METHODS)
        # You can log result.issues here for telemetry if needed
        return {"trace_matrix": [TraceMatrixRow(**row) for row in tm_rows_norm]}  # type: ignore
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TM failed: {e}")

# -------- Optional: warmup on startup (no heavy gen) --------
@app.on_event("startup")
def _maybe_warmup():
    """
    If WARMUP=1, try to lightly touch model loaders to pay the import cost early.
    This won't run full generation or block the service for long.
    """
    if os.getenv("WARMUP", "0") != "1":
        return
    try:
        # Touch loaders safely (no prompt generation)
        if os.getenv("USE_HA_MODEL", "0") == "1":
            ha_infer._load_model()  # type: ignore[attr-defined]
        if os.getenv("USE_DVP_MODEL", "0") == "1":
            dvp_infer._load_model()  # type: ignore[attr-defined]
        if os.getenv("USE_TM_MODEL", "0") == "1":
            tm_infer._load_tm_model()  # type: ignore[attr-defined]
    except Exception:
        # Warmup is best-effort; never block startup
        pass
