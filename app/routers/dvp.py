# app/routers/dvp.py  (example)
from fastapi import APIRouter, HTTPException
import json

router = APIRouter()

@router.post("/dvp")
async def dvp_endpoint(body: dict):
    try:
        # Normalize `ha` into a list[dict]
        ha = body.get("ha", [])
        if isinstance(ha, dict):
            ha = [ha]
        elif isinstance(ha, str):
            try:
                parsed = json.loads(ha)
                if isinstance(parsed, dict):
                    ha = [parsed]
                elif isinstance(parsed, list):
                    ha = parsed
                else:
                    ha = []
            except Exception:
                ha = []
        elif not isinstance(ha, list):
            ha = []

        # Keep only dict rows
        ha = [x for x in ha if isinstance(x, dict)]

        # Now safe to use row.get(...)
        # ... your existing DVP logic here ...

        return {"ok": True, "ha_rows": len(ha)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVP failed: {e}")
