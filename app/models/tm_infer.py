# app/main.py
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

# Optional startup hook (no-op if file not present)
try:
    import startup_cleanup  # noqa: F401
except Exception:
    pass

# -------------------------------------------------------------------
# Feature flags & limits
# -------------------------------------------------------------------
ENABLE_DVP    = os.getenv("ENABLE_DVP", "1") == "1"
ENABLE_TM     = os.getenv("ENABLE_TM",  "1") == "1"   # <- default ON now
MAX_REQS      = int(os.getenv("MAX_REQS", "5"))
QUICK_LIMIT   = int(os.getenv("QUICK_LIMIT", "0"))    # 0 = off; >0 caps rows for quick tests
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")

# -------------------------------------------------------------------
# Model modules (import DVP/TM defensively)
# -------------------------------------------------------------------
from app.models import ha_infer  # uses infer_ha()

_dvp_available = False
try:
    from app.models import dvp_infer  # type: ignore
    _dvp_available = True
except Exception:
    _dvp_available = False

_tm_available = False
try:
    from app.models import tm_infer  # type: ignore
    _tm_available = True
except Exception:
    _tm_available = False

# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------
app = FastAPI(title="DHF Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _auth(authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != BACKEND_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")

def _limit_requirements(reqs: List[Dict[str, Any]], n: Optional[int] = None) -> List[Dict[str, Any]]:
    if not isinstance(reqs, list):
        return []
    hard = n or QUICK_LIMIT or MAX_REQS
    if hard and len(reqs) > hard:
        return reqs[:hard]
    return reqs

def _cap_rows(rows: List[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    if n and len(rows) > n:
        return rows[:n]
    if QUICK_LIMIT and len(rows) > QUICK_LIMIT:
        return rows[:QUICK_LIMIT]
    return rows

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "enable_dvp": ENABLE_DVP and _dvp_available,
        "enable_tm": ENABLE_TM and _tm_available,
        "tm_available": _tm_available,
        "max_reqs": MAX_REQS,
        "quick_limit": QUICK_LIMIT,
        # helpful to see adapter settings quickly
        "tm_adapter_enabled": os.getenv("USE_TM_ADAPTER", "1") == "1",
        "tm_adapter_repo": os.getenv("TM_ADAPTER_REPO", "cheranengg/dhf-tm-adapter"),
        "base_model": os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
    }

@app.get("/")
def root():
    return {"ok": True, "service": "DHF Backend"}

@app.get("/debug/ha_status")
def debug_ha_status():
    rag_rows = len(getattr(ha_infer, "_RAG_DB", []) or [])
    maude_local_rows = len(getattr(ha_infer, "_MAUDE_ROWS", []) or [])
    return {
        "adapter_enabled": bool(os.getenv("USE_HA_ADAPTER", "1") == "1"),
        "adapter_repo": os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter"),
        "base_model": os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
        "rag_path": os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl"),
        "rag_rows_loaded": rag_rows,
        "maude_local_only": os.getenv("MAUDE_LOCAL_ONLY", "1") == "1",
        "maude_local_path": os.getenv("MAUDE_LOCAL_JSONL", "app/rag_sources/maude_sigma_spectrum.jsonl"),
        "maude_fraction": float(os.getenv("MAUDE_FRACTION", "0.70")),
        "maude_local_rows": maude_local_rows,
        "hf_cache_dir": getattr(ha_infer, "CACHE_DIR", os.getenv("HF_HOME", "/tmp/hf")),
        "offload_dir": getattr(ha_infer, "OFFLOAD_DIR", os.getenv("OFFLOAD_DIR", "/tmp/offload")),
        "ha_row_limit": int(os.getenv("HA_ROW_LIMIT", "5")),
        "ha_input_max_tokens": int(os.getenv("HA_INPUT_MAX_TOKENS", "512")),
        "ha_max_new_tokens": int(os.getenv("HA_MAX_NEW_TOKENS", "192")),
    }

@app.post("/hazard-analysis")
def hazard_endpoint(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(default=None),
    debug: int = Query(0, description="Include diagnostics in response"),
    limit: Optional[int] = Query(default=None, description="Cap rows for quick tests"),
):
    _auth(authorization)
    diag: Dict[str, Any] = {}
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        reqs = _limit_requirements(reqs, n=limit)
        ha_rows = ha_infer.infer_ha(reqs)
        ha_rows = _cap_rows(ha_rows, limit)
        if debug:
            diag = {
                "adapter": os.getenv("USE_HA_ADAPTER", "1") == "1",
                "adapter_repo": os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter"),
                "base_model": os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
                "rag_path": os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl"),
                "rag_rows": len(getattr(ha_infer, "_RAG_DB", []) or []),
                "maude_fetch": os.getenv("MAUDE_FETCH", "0") == "1",
                "maude_device": os.getenv("MAUDE_DEVICE", "SIGMA SPECTRUM"),
                "maude_limit": int(os.getenv("MAUDE_LIMIT", "20")),
                "maude_ttl": int(os.getenv("MAUDE_TTL", "86400")),
                "n_rows": len(ha_rows),
                "quick_limit": QUICK_LIMIT,
                "limit_param": limit,
            }
        return {"ok": True, "ha": ha_rows, **({"diag": diag} if debug else {})}
    except Exception as e:
        tb = traceback.format_exc()
        print({"hazard_endpoint_error": str(e), "traceback": tb})
        if debug:
            diag.update({"error": str(e), "traceback": tb})
            return {"ok": False, "ha": [], "diag": diag}
        raise HTTPException(status_code=500, detail=f"HA failed: {e}")

@app.post("/dvp")
def dvp_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(default=None),
    limit: Optional[int] = Query(default=None, description="Cap rows for quick tests"),
):
    _auth(authorization)
    if not (ENABLE_DVP and _dvp_available):
        raise HTTPException(status_code=503, detail="DVP unavailable.")
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        reqs    = _limit_requirements(reqs, n=limit)
        ha_rows = _cap_rows(ha_rows, limit)
        dvp_rows = dvp_infer.dvp_predict(reqs, ha_rows)  # type: ignore
        dvp_rows = _cap_rows(dvp_rows, limit)
        return {"ok": True, "dvp": dvp_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVP failed: {e}")

# ---- TM core handler (shared by /tm and /trace-matrix) ----
def _tm_handler(
    payload: Dict[str, Any],
    authorization: Optional[str],
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    _auth(authorization)
    if not (ENABLE_TM and _tm_available):
        raise HTTPException(status_code=503, detail="TM unavailable.")
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        dvp_rows: List[Dict[str, Any]] = payload.get("dvp") or []
        reqs     = _limit_requirements(reqs, n=limit)
        ha_rows  = _cap_rows(ha_rows, limit)
        dvp_rows = _cap_rows(dvp_rows, limit)

        # Use model-backed joiner (deterministic; model presence is optional)
        tm_rows = tm_infer.tm_predict(reqs, ha_rows, dvp_rows)  # type: ignore
        tm_rows = _cap_rows(tm_rows, limit)
        return {"ok": True, "tm": tm_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TM failed: {e}")


@app.post("/tm")
def tm_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(default=None),
    limit: Optional[int] = Query(default=None, description="Cap rows for quick tests"),
):
    return _tm_handler(payload, authorization, limit)

@app.post("/trace-matrix")
def trace_matrix_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(default=None),
    limit: Optional[int] = Query(default=None, description="Cap rows for quick tests"),
):
    return _tm_handler(payload, authorization, limit)

# ------------------------- Debug -----------------------------------
@app.post("/debug/smoke")
def debug_smoke(
    authorization: Optional[str] = Header(default=None),
    limit: Optional[int] = Query(default=None, description="Cap rows for quick tests"),
):
    _auth(authorization)
    reqs = [
        {"Requirement ID": "REQ-001", "Verification ID": "VER-001",
         "Requirements": "The pump shall maintain flow accuracy within ±5% from set value."},
        {"Requirement ID": "REQ-002", "Verification ID": "VER-002",
         "Requirements": "Occlusion alarm shall trigger within 30 seconds at 100 kPa back pressure."},
        {"Requirement ID": "REQ-003", "Verification ID": "VER-003",
         "Requirements": "Labeling shall be legible at 30 cm and use ISO 15223-1 symbols."},
        {"Requirement ID": "REQ-004", "Verification ID": "VER-004",
         "Requirements": "Materials in patient-contact shall pass ISO 10993-1 cytotoxicity test."},
        {"Requirement ID": "REQ-005", "Verification ID": "VER-005",
         "Requirements": "Leakage current at rated voltage shall not exceed 100 µA."},
    ]
    reqs = _limit_requirements(reqs, n=limit)
    try:
        ha_rows = ha_infer.infer_ha(reqs)
        ha_rows = _cap_rows(ha_rows, limit)
        out: Dict[str, Any] = {"ok": True, "ha": ha_rows}

        if ENABLE_DVP and _dvp_available:
            dvp_rows = dvp_infer.dvp_predict(reqs, ha_rows)  # type: ignore
            out["dvp"] = _cap_rows(dvp_rows, limit)

        if ENABLE_TM and _tm_available:
            out["tm"] = _cap_rows(
                _tm_handler({"requirements": reqs, "ha": ha_rows, "dvp": out.get("dvp", [])},
                            authorization, limit)["tm"],  # type: ignore
                limit
            )
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}
