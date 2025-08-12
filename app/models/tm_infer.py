# app/models/tm_infer.py
from __future__ import annotations
import os, json, re
from typing import List, Dict, Any, Optional

# ---------------- Env ----------------
USE_TM_MODEL = os.getenv("USE_TM_MODEL", "0") == "1"
TM_MODEL_DIR = os.getenv("TM_MODEL_DIR", "/models/mistral_finetuned_Trace_Matrix")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HOME = os.getenv("HF_HOME")
if USE_TM_MODEL:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
else:
    AutoTokenizer = AutoModelForCausalLM = None  # type: ignore
    torch = None  # type: ignore
    DEVICE = DTYPE = None  # type: ignore

_tokenizer: Optional["AutoTokenizer"] = None
_model: Optional["AutoModelForCausalLM"] = None

# ---------------- Helpers ----------------
_JSON_OBJ = re.compile(r"\{[\s\S]*?\}")

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_OBJ.findall(text)
    if not m:
        return None
    js = (m[-1]
          .replace("'", '"')
          .replace("\\n", " "))
    js = re.sub(r"\s+", " ", js)
    js = re.sub(r",\s*\}", "}", js)
    js = re.sub(r",\s*\]", "]", js)
    try:
        return json.loads(js)
    except Exception:
        return None

def _is_heading(requirement_text: str) -> bool:
    t = (requirement_text or "").strip().lower()
    return (not t) or t.endswith(" requirements") or t in {
        "functional requirements", "performance requirements", "safety requirements",
        "usability requirements", "environmental requirements", "design inputs", "general requirements"
    }

def _join_unique(values: List[str]) -> str:
    vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().upper() != "NA"]
    seen: List[str] = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    return ", ".join(seen) if seen else "NA"

def _compose_fallback(rid: str, vid: str, rtxt: str,
                      ha_slice: List[Dict[str, Any]], drow: Dict[str, Any]) -> Dict[str, Any]:
    risk_ids = _join_unique([h.get("risk_id") or h.get("Risk ID") or "" for h in ha_slice])
    risks_to_health = _join_unique([h.get("risk_to_health") or h.get("Risk to Health") or "" for h in ha_slice])
    risk_controls = _join_unique([h.get("risk_control") or h.get("HA Risk Control") or "" for h in ha_slice])
    method = drow.get("verification_method") or drow.get("Verification Method") or "TBD - Human / SME input"
    criteria = drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "TBD - Human / SME input"
    return {
        "verification_id": vid, "requirement_id": rid, "requirements": rtxt,
        "risk_ids": risk_ids if risk_ids != "NA" else "TBD - Human / SME input",
        "risks_to_health": risks_to_health if risks_to_health != "NA" else "TBD - Human / SME input",
        "ha_risk_controls": risk_controls if risk_controls != "NA" else "TBD - Human / SME input",
        "verification_method": method, "acceptance_criteria": criteria,
    }

def _load_tm_model():
    global _tokenizer, _model
    if not USE_TM_MODEL or _model is not None:
        return
    token_kw = {"token": HF_TOKEN} if HF_TOKEN else {}
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    _tokenizer = AutoTokenizer.from_pretrained(TM_MODEL_DIR, **token_kw, cache_dir=HF_HOME)  # type: ignore
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token  # type: ignore
    _model = AutoModelForCausalLM.from_pretrained(TM_MODEL_DIR, torch_dtype=DTYPE, **token_kw, cache_dir=HF_HOME)  # type: ignore
    _model.to(DEVICE)  # type: ignore

# ---------------- Public API ----------------
def tm_predict(requirements: List[Dict[str, Any]],
               ha: List[Dict[str, Any]],
               dvp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _load_tm_model()
    rows: List[Dict[str, Any]] = []

    # group HA by requirement_id
    from collections import defaultdict
    ha_by_req: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in ha or []:
        rid = str(h.get("requirement_id") or h.get("Requirement ID") or "")
        if rid:
            ha_by_req[rid].append(h)

    # map DVP by verification_id
    dvp_by_vid: Dict[str, Dict[str, Any]] = {}
    for d in dvp or []:
        vid = str(d.get("verification_id") or d.get("Verification ID") or "")
        if vid and vid not in dvp_by_vid:
            dvp_by_vid[vid] = d

    for r in requirements:
        rid = str(r.get("Requirement ID", ""))
        vid = str(r.get("Verification ID", ""))
        rtxt = r.get("Requirements", "") or ""

        if _is_heading(rtxt):
            rows.append({
                "verification_id": vid or "NA", "requirement_id": rid, "requirements": rtxt,
                "risk_ids": "NA", "risks_to_health": "NA", "ha_risk_controls": "NA",
                "verification_method": "NA", "acceptance_criteria": "NA",
            })
            continue

        ha_slice = ha_by_req.get(rid, [])
        drow = dvp_by_vid.get(vid, {})

        if not USE_TM_MODEL:
            rows.append(_compose_fallback(rid, vid, rtxt, ha_slice, drow))
            continue

        # build a compact instruction + context
        instruction = (
            "You are generating a Traceability Matrix row for an infusion pump.\n"
            "Return ONLY one JSON object with keys: "
            "{\"verification_id\",\"requirement_id\",\"requirements\","
            "\"risk_ids\",\"risks_to_health\",\"ha_risk_controls\","
            "\"verification_method\",\"acceptance_criteria\"}.\n"
            "For list-like fields, return a single comma-separated string."
        )
        context = {
            "requirement": {"Requirement ID": rid, "Verification ID": vid, "Requirements": rtxt},
            "ha": ha_slice,
            "dvp": drow,
        }
        prompt = instruction + "\nINPUT:\n" + json.dumps(context, ensure_ascii=False)

        parsed = None
        try:
            import torch
            inputs = _tokenizer(prompt, return_tensors="pt").to(DEVICE)  # type: ignore
            with torch.no_grad():
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.05,
                )  # type: ignore
            decoded = _tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore
            parsed = _extract_json(decoded)
        except Exception:
            parsed = None

        if not parsed:
            rows.append(_compose_fallback(rid, vid, rtxt, ha_slice, drow))
            continue

        # Ensure required fields and fill defaults from HA+DVP if blank
        risk_ids = _join_unique([h.get("risk_id") or h.get("Risk ID") or "" for h in ha_slice])
        risks_to_health = _join_unique([h.get("risk_to_health") or h.get("Risk to Health") or "" for h in ha_slice])
        risk_controls = _join_unique([h.get("risk_control") or h.get("HA Risk Control") or "" for h in ha_slice])
        method = drow.get("verification_method") or drow.get("Verification Method") or "TBD - Human / SME input"
        criteria = drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "TBD - Human / SME input"

        parsed.setdefault("verification_id", vid)
        parsed.setdefault("requirement_id", rid)
        parsed.setdefault("requirements", rtxt)
        parsed.setdefault("risk_ids", risk_ids or "TBD - Human / SME input")
        parsed.setdefault("risks_to_health", risks_to_health or "TBD - Human / SME input")
        parsed.setdefault("ha_risk_controls", risk_controls or "TBD - Human / SME input")
        parsed.setdefault("verification_method", method)
        parsed.setdefault("acceptance_criteria", criteria)

        for k in ("risk_ids","risks_to_health","ha_risk_controls","verification_method","acceptance_criteria"):
            if not str(parsed.get(k, "")).strip():
                parsed[k] = "TBD - Human / SME input"

        rows.append(parsed)

    return rows
