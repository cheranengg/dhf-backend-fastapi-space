# app/routers/ha.py
from __future__ import annotations

from fastapi import APIRouter, Request
from typing import Dict, Any, List

from app.models.ha_infer import (
    infer_ha,             # <-- public API in current ha_infer.py
    _RAG_DB,              # for quick diag counts
    USE_HA_ADAPTER,
)

router = APIRouter()

@router.post("/hazard-analysis")
async def hazard_analysis(request: Request, body: Dict[str, Any]):
    """
    Expects: {"requirements": [{"Requirement ID": "...", "Requirements": "..."}]}
    Returns: {"ha": [...] } (+ optional {"diag": ...} if debug=1)
    """
    reqs: List[Dict[str, Any]] = body.get("requirements", []) or []
    rows = infer_ha(reqs)

    debug = request.query_params.get("debug", "0") == "1"
    resp: Dict[str, Any] = {"ha": rows}
    if debug:
        resp["diag"] = {
            "rag_rows": len(_RAG_DB),
            "adapter": USE_HA_ADAPTER,
            "n_rows": len(rows),
        }
    return resp
