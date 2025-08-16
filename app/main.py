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
ENABLE_DVP   = os.getenv("ENABLE_DVP", "1") == "1"
ENABLE_TM    = os.getenv("ENABLE_TM",  "0") == "1"   # start disabled until ready
MAX_REQS     = int(os.getenv("MAX_REQS", "5"))
QUICK_LIMIT  = int(os.getenv("QUICK_LIMIT", "0"))    # 0 = off; >0 caps rows for quick tests
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
    allow_origins=["*"],  # lock down for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _auth(authorization: Optional[str]) -> None:
    """Simple bearer-token auth shared by all endpoints."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != BACKEND_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")


def _limit_requirements(
    reqs: List[Dict[str, Any]],
    n: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Cap the number of requirement rows we *process* to speed up test runs.
    Precedence: per-call n > QUICK_LIMIT > MAX_REQS.
    """
    if not isinstance(reqs, list):
        return []
    hard = n or QUICK_LIMIT or MAX_REQS
    if hard and len(reqs) > hard:
        return reqs[:hard]
    return reqs


def _cap_rows(rows: List[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    """Cap the number of rows we *return* for quick previews."""
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
    }

@app.get("/")
def root():
    return {"ok": True, "service": "DHF Backend"}
    
@app.get("/debug/ha_status")
def debug_ha_status():
    """Adapter-only status + RAG + MAUDE visibility."""
    rag_rows = len(getattr(ha_infer, "_RAG_DB", []) or [])
    maude_local_rows = len(getattr(ha_infer, "_MAUDE_ROWS", []) or [])
    return {
        "adapter_enabled": bool(os.getenv("USE_HA_ADAPTER", "1") == "1"),
        "adapter_repo": os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter"),
        "base_model": os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
        "rag_path": os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl"),
        "rag_rows_loaded": rag_rows,
        # MAUDE local enrichment
        "maude_local_only": os.getenv("MAUDE_LOCAL_ONLY", "1") == "1",
        "maude_local_path": os.getenv("MAUDE_LOCAL_JSONL", "app/rag_sources/sigma_spectrum_maude.jsonl"),
        "maude_fraction": float(os.getenv("MAUDE_FRACTION", "0.70")),
        "maude_local_rows": maude_local_rows,
        # effective caches/offload (what the model really uses)
        "hf_cache_dir": getattr(ha_infer, "CACHE_DIR", os.getenv("HF_HOME", "/tmp/hf")),
        "offload_dir": getattr(ha_infer, "OFFLOAD_DIR", os.getenv("OFFLOAD_DIR", "/tmp/offload")),
        # token/row caps
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

    diag: Dict[str, Any] = {}
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        reqs = _limit_requirements(reqs, n=limit)

        # === HA inference (adapter + synthetic RAG + MAUDE enrichment) ===
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
            diag.update({
                "error": str(e),
                "traceback": tb,
            })
            return {"ok": False, "ha": [], "diag": diag}
        raise HTTPException(status_code=500, detail=f"HA failed: {e}")


@app.post("/dvp")
def dvp_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(default=None),
    limit: Optional[int] = Query(default=None, description="Cap rows for quick tests"),
):
    """
    Request body:
    {
      "requirements": [...],   # shape like hazard-analysis
      "ha": [...]              # output from hazard-analysis (optional but helpful)
    }
    """
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


def _tm_handler(
    payload: Dict[str, Any],
    authorization: Optional[str],
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Accepts:
    {
      "requirements": [...],
      "ha": [...],
      "dvp": [...]
    }
    Returns: {"ok": true, "tm": [...]}
    """
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

        if hasattr(tm_infer, "tm_predict"):
            tm_rows = tm_infer.tm_predict(reqs, ha_rows, dvp_rows)  # type: ignore
        else:
            # very light fallback (no model): join key fields so UI doesn’t 404
            from collections import defaultdict

            def j(values):
                vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().upper() != "NA"]
                seen = []
                for v in vals:
                    if v not in seen:
                        seen.append(v)
                return ", ".join(seen) if seen else "TBD - Human / SME input"

            ha_by_req = defaultdict(list)
            for h in ha_rows:
                rid = str(h.get("requirement_id") or h.get("Requirement ID") or "")
                if rid:
                    ha_by_req[rid].append(h)

            dvp_by_vid = {}
            for d in dvp_rows:
                vid = str(d.get("verification_id") or d.get("Verification ID") or "")
                if vid and vid not in dvp_by_vid:
                    dvp_by_vid[vid] = d

            tm_rows: List[Dict[str, Any]] = []
            for r in reqs:
                rid = str(r.get("Requirement ID") or "")
                vid = str(r.get("Verification ID") or "")
                rtxt = str(r.get("Requirements") or "")

                ha_slice = ha_by_req.get(rid, [])
                drow = dvp_by_vid.get(vid, {})

                risk_ids = j([h.get("risk_id") or h.get("Risk ID") or "" for h in ha_slice])
                risks    = j([h.get("risk_to_health") or h.get("Risk to Health") or "" for h in ha_slice])
                controls = j([h.get("risk_control") or h.get("HA Risk Control") or "" for h in ha_slice])

                method = drow.get("verification_method") or drow.get("Verification Method") or "TBD - Human / SME input"
                crit   = drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "TBD - Human / SME input"

                tm_rows.append({
                    "verification_id": vid or "NA",
                    "requirement_id": rid,
                    "requirements": rtxt,
                    "risk_ids": risk_ids,
                    "risks_to_health": risks,
                    "ha_risk_controls": controls,
                    "verification_method": method,
                    "acceptance_criteria": crit,
                })

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
    """
    Quick smoke test that runs HA (always) and DVP/TM if enabled.
    """
    _auth(authorization)
    reqs = [
        {"Requirement ID": "REQ-001", "Verification ID": "VER-001",
         "Requirements": "The pump shall maintain flow accuracy within ±5% from set value."},
        {"Requirement ID": "REQ-002", "Verification ID": "VER-002",
         "Requirements": "Labeling shall be legible at 30 cm and use ISO 15223-1 symbols."},
        {"Requirement ID": "REQ-003", "Verification ID": "VER-003",
         "Requirements": "Occlusion alarm shall trigger within 30 seconds at 100 kPa back pressure."},
        {"Requirement ID": "REQ-004", "Verification ID": "VER-004",
         "Requirements": "Leakage current at rated voltage shall not exceed 100 µA."},
        {"Requirement ID": "REQ-005", "Verification ID": "VER-005",
         "Requirements": "User interface shall prevent decimal-point mis-entry during setup."},
        {"Requirement ID": "REQ-006", "Verification ID": "VER-006",
         "Requirements": "Materials in patient-contact shall pass ISO 10993-1 cytotoxicity test."},
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
