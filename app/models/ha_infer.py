# app/models/ha_infer.py

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
ADAPTER_REPO = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")
MERGED_REPO  = os.getenv("HA_MERGED_REPO", "cheranengg/dhf-ha-merged")
BASE_MODEL   = os.getenv("HA_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

RAG_PATH = Path(os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl"))
DEVICE   = 0 if torch.cuda.is_available() else -1

adapter_enabled = os.getenv("HA_ADAPTER", "1") == "1"
merged_enabled  = os.getenv("HA_MERGED", "1") == "1"

# -------------------------------------------------------------------
# Load Model
# -------------------------------------------------------------------
print(f"🔹 Loading Hazard Analysis model: adapter={adapter_enabled}, merged={merged_enabled}")

if merged_enabled:
    model_name = MERGED_REPO
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif adapter_enabled:
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
else:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=DEVICE)

# -------------------------------------------------------------------
# Load RAG Source
# -------------------------------------------------------------------
rag_rows = []
if RAG_PATH.exists():
    with open(RAG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rag_rows.append(json.loads(line))
            except Exception:
                pass

print(f"📂 RAG file exists: {RAG_PATH.exists()}, rows loaded: {len(rag_rows)}")

# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------
def _match_rag(requirement: str) -> Dict[str, Any]:
    """Very basic keyword match against RAG source."""
    req_low = requirement.lower()
    for row in rag_rows:
        if any(k in req_low for k in [str(v).lower() for v in row.values() if isinstance(v, str)]):
            return row
    return {}

def _gen_row_for_risk(req_id: str, requirement: str) -> Dict[str, Any]:
    rag_match = _match_rag(requirement)
    if rag_match:
        print(f"✅ RAG match found for {req_id}: {rag_match}")
    else:
        print(f"⚠️ No RAG match for {req_id}, sending to model")

    prompt = f"""
You are an expert in medical device hazard analysis.
Generate a structured hazard analysis row for the following requirement.

Requirement ID: {req_id}
Requirement: {requirement}

Fields to generate:
- Risk ID
- Risk to Health
- Hazard
- Hazardous Situation
- Harm
- Sequence of Events
- Severity of Harm
- P0
- P1
- PoH
- Risk Index
- Risk Control
If unknown, return "TBD".
"""
    response = gen_pipe(prompt, max_new_tokens=300, temperature=0.2, do_sample=False)[0]["generated_text"]

    # Parse (very naive parsing for demo — can replace with regex/json parse)
    out = {k.lower().replace(" ", "_"): "TBD" for k in [
        "Risk ID", "Risk to Health", "Hazard", "Hazardous Situation", "Harm", "Sequence of Events",
        "Severity of Harm", "P0", "P1", "PoH", "Risk Index", "Risk Control"
    ]}

    if rag_match:
        out.update({
            "risk_id": rag_match.get("Risk ID", "TBD"),
            "risk_to_health": rag_match.get("Risk to Health", "TBD"),
            "hazard": rag_match.get("Hazard", "TBD"),
            "hazardous_situation": rag_match.get("Hazardous situation", "TBD"),
            "harm": rag_match.get("Harm", "TBD"),
            "sequence_of_events": rag_match.get("Sequence of Events", "TBD"),
            "severity_of_harm": str(rag_match.get("Severity of Harm", "TBD")),
            "p0": rag_match.get("P0", "TBD"),
            "p1": rag_match.get("P1", "TBD"),
            "poh": rag_match.get("PoH", "TBD"),
            "risk_index": rag_match.get("Risk Index", "TBD"),
            "risk_control": rag_match.get("Risk Control", "TBD")
        })

    return {
        "requirement_id": req_id,
        **out
    }

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def ha_predict(reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in reqs:
        rid = str(r.get("Requirement ID") or "").strip()
        rtxt = str(r.get("Requirements") or "").strip()
        if not rid or not rtxt:
            continue
        rows.append(_gen_row_for_risk(rid, rtxt))
    return rows
