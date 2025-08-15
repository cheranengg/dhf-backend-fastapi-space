# app/routers/ha.py  (where your /hazard-analysis endpoint lives)
from fastapi import APIRouter, Request
from app.models.ha_infer import ha_predict, _RAG_DB, USE_HA_ADAPTER, USE_HA_MODEL

router = APIRouter()

@router.post("/hazard-analysis")
async def hazard_analysis(request: Request, body: dict):
    reqs = body.get("requirements", [])
    rows = ha_predict(reqs)

    # if debug=1 in query string, attach quick diag
    debug = request.query_params.get("debug", "0") == "1"
    resp = {"ha": rows}
    if debug:
        resp["diag"] = {
            "rag_rows": len(_RAG_DB),
            "adapter": USE_HA_ADAPTER,
            "merged": USE_HA_MODEL,
            "n_rows": len(rows)
        }
    return resp
