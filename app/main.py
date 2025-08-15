# app/main.py
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

# Optional startup hook (no-op if file not present)
try:
    import startup_cleanup  # noqa: F401
except Exception:
    pass

# Feature flags
ENABLE_DVP = os.getenv("ENABLE_DVP", "1") == "1"
ENABLE_TM  = os.getenv("ENABLE_TM",  "0") == "1"  # start disabled until ready
MAX_REQS   = int(os.getenv("MAX_REQS", "10"))

# Import model modules
from app.models import ha_infer  # HA adapter-only infer_ha()

# DVP/TM are optional; import defensively
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


def _limit_requirements(reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cap volume to keep inference stable/responsive."""
    if not isinstance(reqs, list):
        return []
    if MAX_REQS and len(reqs) > MAX_REQS:
        return reqs[:MAX_REQS]
    return reqs


# -------------------------------------------------------------------
# Health / Debug
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "enable_dvp": ENABLE_DVP and _dvp_available,
        "enable_tm": ENABLE_TM and _tm_available,
        "tm_available": _tm_available,
        "max_reqs": MAX_REQS,
    }


@app.get("/debug/ha_status")
def debug_ha_status():
    """Adapter-only HA status + RAG + MAUDE visibility."""
    rag_rows = len(getattr(ha_infer, "_RAG_DB", []) or [])
    return {
        "adapter_enabled": bool(os.getenv("USE_HA_ADAPTER", "1") == "1"),
        "adapter_repo": os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter"),
        "base_model": os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
        "rag_path": os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl"),
        "rag_rows_loaded": rag_rows,
        # MAUDE internet enrichment
        "maude_fetch": os.getenv("MAUDE_FETCH", "0") == "1",
        "maude_device": os.getenv("MAUDE_DEVICE", "SIGMA SPECTRUM"),
        "maude_limit": int(os.getenv("MAUDE_LIMIT", "20")),
        "maude_ttl": int(os.getenv("MAUDE_TTL", "86400")),
    }


@app.get("/debug/dvp_status")
def debug_dvp_status():
    """DVP adapter + retrieval/Serper flags for quick inspection."""
    return {
        "dvp_available": _dvp_available,
        "adapter_enabled": os.getenv("USE_DVP_ADAPTER", "1") == "1",
        "adapter_repo": os.getenv("DVP_ADAPTER_REPO", "cheranengg/dhf-dvp-adapter"),
        "base_model": os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
        "retrieval_enabled": os.getenv("ENABLE_DVP_RETRIEVAL", "1") == "1",
        "serper_key_present": bool(os.getenv("SERPER_API_KEY", "")),
        "gen": {
            "max_new_tokens": int(os.getenv("DVP_MAX_NEW_TOKENS", "320")),
            "temperature": float(os.getenv("DVP_TEMPERATURE", "0.3")),
            "top_p": float(os.getenv("DVP_TOP_P", "0.9")),
            "do_sample": os.getenv("DVP_DO_SAMPLE", "1") == "1",
            "num_beams": int(os.getenv("DVP_NUM_BEAMS", "1")),
            "repetition_penalty": float(os.getenv("DVP_REPETITION_PENALTY", "1.05")),
        },
    }


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.post("/hazard-analysis")
def hazard_endpoint(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
    debug: int = Query(0),
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
        reqs = _limit_requirements(reqs)

        # === HA inference (adapter + synthetic RAG + MAUDE enrichment) ===
        ha_rows = ha_infer.infer_ha(reqs)

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
            }

        return {"ok": True, "ha": ha_rows, **({"diag": diag} if debug else {})}

    except Exception as e:
        tb = traceback.format_exc()
        # log full details to the Space logs
        print({"hazard_endpoint_error": str(e), "traceback": tb})
        if debug:
            diag.update({
                "error": str(e),
                "traceback": tb,
                "adapter": os.getenv("USE_HA_ADAPTER", "1") == "1",
                "adapter_repo": os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter"),
                "base_model": os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
                "rag_path": os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl"),
                "rag_rows": len(getattr(ha_infer, "_RAG_DB", []) or []),
                "maude_fetch": os.getenv("MAUDE_FETCH", "0") == "1",
                "maude_device": os.getenv("MAUDE_DEVICE", "SIGMA SPECTRUM"),
                "maude_limit": int(os.getenv("MAUDE_LIMIT", "20")),
                "maude_ttl": int(os.getenv("MAUDE_TTL", "86400")),
            })
            return {"ok": False, "ha": [], "diag": diag}
        raise HTTPException(status_code=500, detail=f"HA failed: {e}")


@app.post("/dvp")
def dvp_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    """
    Request body:
    {
      "requirements": [...],
      "ha": [...]      # optional but helpful
    }
    """
    _auth(authorization)
    if not (ENABLE_DVP and _dvp_available):
        raise HTTPException(status_code=503, detail="DVP unavailable.")
    try:
        reqs: List[Dict[str, Any]] = payload.get("requirements") or []
        ha_rows: List[Dict[str, Any]] = payload.get("ha") or []
        reqs = _limit_requirements(reqs)
        dvp_rows = dvp_infer.dvp_predict(reqs, ha_rows)  # type: ignore
        return {"ok": True, "dvp": dvp_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVP failed: {e}")


# ---------------- TM handlers (two routes mapping to same function) -------------
def _tm_handler(
    payload: Dict[str, Any],
    authorization: str | None,
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
        reqs = _limit_requirements(reqs)

        if hasattr(tm_infer, "tm_predict"):
            tm_rows = tm_infer.tm_predict(reqs, ha_rows, dvp_rows)  # type: ignore
        else:
            # lightweight fallback row (no model): prevents 404s in UI
            from collections import defaultdict

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

            def j(values):
                vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().upper() != "NA"]
                seen = []
                for v in vals:
                    if v not in seen:
                        seen.append(v)
                return ", ".join(seen) if seen else "TBD - Human / SME input"

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

        return {"ok": True, "tm": tm_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TM failed: {e}")


@app.post("/tm")
def tm_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    return _tm_handler(payload, authorization)


@app.post("/trace-matrix")
def trace_matrix_endpoint(
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
):
    return _tm_handler(payload, authorization)


# ------------------------- Debug -----------------------------------
@app.post("/debug/smoke")
def debug_smoke(
    authorization: str | None = Header(default=None),
):
    """
    Quick smoke test that runs HA (always) and DVP/TM if enabled.
    """
    _auth(authorization)
    reqs = [
        {"Requirement ID": "REQ-001", "Verification ID": "VER-001",
         "Requirements": "The pump shall maintain flow accuracy within +/- 5% from set value."},
        {"Requirement ID": "REQ-002", "Verification ID": "VER-002",
         "Requirements": "Labeling shall be legible at 30 cm and use ISO 15223-1 symbols."},
    ]
    try:
        ha_rows = ha_infer.infer_ha(_limit_requirements(reqs))
        out: Dict[str, Any] = {"ok": True, "ha": ha_rows}

        if ENABLE_DVP and _dvp_available:
            dvp_rows = dvp_infer.dvp_predict(_limit_requirements(reqs), ha_rows)  # type: ignore
            out["dvp"] = dvp_rows

        if ENABLE_TM and _tm_available:
            out["tm"] = _tm_handler({"requirements": reqs, "ha": ha_rows, "dvp": out.get("dvp", [])}, authorization)["tm"]  # type: ignore

        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}
