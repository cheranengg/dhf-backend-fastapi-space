# app/models/tm_infer.py
from __future__ import annotations
import os
from typing import List, Dict, Any

USE_TM_MODEL = os.getenv("USE_TM_MODEL", "1") == "1"  # kept for symmetry
TM_MAX_ROWS  = int(os.getenv("TM_MAX_ROWS", "10"))

def _norm(s: Any) -> str:
    return str(s or "").strip()

def tm_predict(requirements: List[Dict[str, Any]],
               ha_rows: List[Dict[str, Any]],
               dvp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a simple trace matrix:
      Requirement ID -> (HA risk rows) -> (DVP verification rows)
    One TM row per (Requirement ID, Risk_to_health) linking to nearest DVP row by same Requirement ID.
    """
    if not USE_TM_MODEL:
        return []

    # Index DVP rows by requirement_id for quick lookup
    dvp_by_req: Dict[str, Dict[str, Any]] = {}
    for d in dvp_rows or []:
        rid = _norm(d.get("requirement_id") or d.get("Requirement ID"))
        if rid and rid not in dvp_by_req:
            dvp_by_req[rid] = d

    rows: List[Dict[str, Any]] = []
    for h in ha_rows or []:
        rid = _norm(h.get("requirement_id") or h.get("Requirement ID"))
        if not rid:
            continue

        dvp = dvp_by_req.get(rid, {})
        rows.append({
            "requirement_id": rid,
            "requirement": _norm(next((r.get("Requirements") for r in requirements if _norm(r.get("Requirement ID")) == rid), "")),
            "risk_id": _norm(h.get("risk_id")),
            "risk_to_health": _norm(h.get("risk_to_health")),
            "hazard": _norm(h.get("hazard")),
            "verification_id": _norm(dvp.get("verification_id")),
            "verification_method": _norm(dvp.get("verification_method")),
            "sample_size": _norm(dvp.get("sample_size")),
            "dvp_present": "Yes" if dvp else "No",
        })
        if len(rows) >= TM_MAX_ROWS:
            break

    return rows[:TM_MAX_ROWS]
