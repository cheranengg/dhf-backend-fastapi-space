# app/main.py
from __future__ import annotations

# --- ensure HF cache dir is writable and exists ---
import os
cache_dir = (
    os.environ.get("HF_HOME")
    or os.environ.get("HF_HUB_CACHE")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or os.environ.get("TRANSFORMERS_CACHE")
    or "/data/.cache/hf"        # on HF Spaces, /data is writable
)
os.environ.setdefault("HF_HOME", cache_dir)
os.environ.setdefault("HF_HUB_CACHE", cache_dir)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
os.makedirs(cache_dir, exist_ok=True)

import traceback
from typing import Any, Dict, List
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

# -------- App --------
app = FastAPI(title="DHF Backend (HA/DVP/TM)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------- Helpers --------
def _reqs_to_dicts(reqs: List[RequirementInput]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in reqs:
        out.append({
            "Requirement ID": sanitize_text(r.requirement_id or ""),
            "Verification ID": "",
            "Requirements": sanitize_text(r.requirements),
        })
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

@app.get("/debug/flags", dependencies=[Depends(require_auth)])
def debug_flags():
    return {
        "USE_HA_MODEL": os.getenv("USE_HA_MODEL", "0"),
        "USE_DVP_MODEL": os.getenv("USE_DVP_MODEL", "0"),
        "USE_TM_MODEL": os.getenv("USE_TM_MODEL", "0"),
        "DVP_MODEL_DIR": os.getenv("DVP_MODEL_DIR", ""),
        "TM_MODEL_DIR": os.getenv("TM_MODEL_DIR", ""),
        "HA_MODEL_MERGED_DIR": os.getenv("HA_MODEL_MERGED_DIR", ""),
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
        reqs_dict = raw_reqs if "Requirements" in (raw_reqs[0] or {}) else _reqs_to_dicts(raw_reqs)
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
        reqs_dict = raw_reqs if "Requirements" in (raw_reqs[0] or {}) else _reqs_to_dicts(raw_reqs)
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
        reqs_dict = raw_reqs if "Requirements" in (raw_reqs[0] or {}) else _reqs_to_dicts(raw_reqs)
        tm_rows_raw: List[Dict[str, Any]] = tm_infer.tm_predict(reqs_dict, ha_rows, dvp_rows)
        tm_rows_norm = [_normalize_tm_row_api(r) for r in tm_rows_raw]
        _ = guard_tm_rows(tm_rows_norm, allowed_methods=DEFAULT_ALLOWED_METHODS)  # optional
        return {"trace_matrix": [TraceMatrixRow(**row) for row in tm_rows_norm]}  # type: ignore
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TM failed: {e}")
