# app/routers/dvp.py
from fastapi import APIRouter, HTTPException
import json

router = APIRouter()

@router.post("/dvp")
async def dvp_endpoint(body: dict):
    try:
        ha = body.get("ha", [])
        if isinstance(ha, dict):
            ha = [ha]
        elif isinstance(ha, str):
            try:
                parsed = json.loads(ha)
                ha = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                ha = []
        elif not isinstance(ha, list):
            ha = []

        ha = [x for x in ha if isinstance(x, dict)]
        # ... your existing DVP logic that uses row.get(...) ...
        return {"ok": True, "ha_rows": len(ha)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVP failed: {e}")
