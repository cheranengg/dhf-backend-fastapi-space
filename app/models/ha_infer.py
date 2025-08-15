# app/models/ha_infer.py

import os
import json
import re
import time
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# =======================================
# Env Vars
# =======================================
USE_HA_ADAPTER = os.getenv("USE_HA_ADAPTER", "1") == "1"
HA_ADAPTER_REPO = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HA_RAG_PATH = os.getenv("HA_RAG_PATH", "/workspace/app/rag_sources/ha_synthetic.jsonl")

MAUDE_FETCH = os.getenv("MAUDE_FETCH", "0") == "1"
MAUDE_DEVICE = os.getenv("MAUDE_DEVICE", "SIGMA SPECTRUM")
MAUDE_LIMIT = int(os.getenv("MAUDE_LIMIT", "20"))
MAUDE_TTL = int(os.getenv("MAUDE_TTL", "86400"))
MAUDE_CACHE_DIR = os.getenv("MAUDE_CACHE_DIR", "/tmp/maude_cache")

# =======================================
# Load Adapter Model
# =======================================
print(f"✅ Loading HA adapter model: {HA_ADAPTER_REPO}")
tokenizer = AutoTokenizer.from_pretrained(HA_ADAPTER_REPO)
model = AutoModelForCausalLM.from_pretrained(
    HA_ADAPTER_REPO,
    device_map="auto",
    torch_dtype="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# =======================================
# Helpers
# =======================================
def _generate_text(prompt: str, max_new_tokens=512) -> str:
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception:
        return None

# =======================================
# Load Synthetic RAG
# =======================================
_RAG_DB = None
if Path(HA_RAG_PATH).exists():
    print(f"✅ Loading synthetic HA RAG from {HA_RAG_PATH}")
    rag_data = load_dataset("json", data_files=HA_RAG_PATH)["train"]
    rag_rows = [dict(r) for r in rag_data]
    docs = [Document(page_content=json.dumps(r), metadata={"risk_to_health": r.get("Risk to Health", "")}) for r in rag_rows]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _RAG_DB = FAISS.from_documents(docs, embeddings)
else:
    print(f"⚠️ Synthetic RAG file not found at {HA_RAG_PATH}")

def _lookup_exact(db, key: str, value: str) -> Optional[Dict[str, Any]]:
    if not db or not value:
        return None
    all_docs = db.similarity_search(value, k=3)
    for d in all_docs:
        js = json.loads(d.page_content)
        if str(js.get(key, "")).strip().lower() == value.strip().lower():
            return js
    return None

# =======================================
# MAUDE Fetch & Cache
# =======================================
def _cache_path(device_brand: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", device_brand.strip()) or "device"
    os.makedirs(MAUDE_CACHE_DIR, exist_ok=True)
    return os.path.join(MAUDE_CACHE_DIR, f"{safe}.json")

def _load_cached(device_brand: str, ttl_seconds: int) -> Optional[List[Dict[str, Any]]]:
    path = _cache_path(device_brand)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_cached(device_brand: str, rows: List[Dict[str, Any]]):
    with open(_cache_path(device_brand), "w", encoding="utf-8") as f:
        json.dump(rows, f)

def fetch_maude_events(device_brand: str, limit: int) -> List[Dict[str, Any]]:
    url = "https://api.fda.gov/device/event.json"
    params = {
        "search": f'device.brand_name:"{device_brand}" OR device.generic_name:"{device_brand}"',
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("results", [])

def _maude_llm_extract(text: str) -> Optional[Dict[str, str]]:
    prompt = f"""Extract these fields in JSON only:
{{
  "Hazard": "",
  "Hazardous Situation": "",
  "Harm": "",
  "Sequence of Events": ""
}}
Text:
{text}"""
    decoded = _generate_text(prompt)
    return _extract_json(decoded)

def get_maude_ha_rows(device_brand: str, risk_to_health: str) -> List[Dict[str, Any]]:
    cached = _load_cached(device_brand, MAUDE_TTL)
    if cached:
        return cached
    try:
        evs = fetch_maude_events(device_brand, MAUDE_LIMIT)
        rows = []
        for ev in evs:
            text = ev.get("event_description") or ev.get("description_of_event") or ""
            mapping = _maude_llm_extract(text) if USE_HA_ADAPTER else None
            if not mapping:
                mapping = {
                    "Hazard": "Device malfunction",
                    "Hazardous Situation": "Patient exposed due to device issue",
                    "Harm": "Potential patient injury",
                    "Sequence of Events": "Setup or usage error"
                }
            rows.append({
                "Risk to Health": risk_to_health,
                **mapping
            })
        _save_cached(device_brand, rows)
        return rows
    except Exception:
        return []

# =======================================
# Merge Functions
# =======================================
def _merge_with_rag(parsed: Dict[str, Any], risk: str) -> Dict[str, Any]:
    # synthetic RAG
    if _RAG_DB:
        rec = _lookup_exact(_RAG_DB, "Risk to Health", risk) or {}
        for k in ["Hazard", "Hazardous Situation", "Harm", "Sequence of Events", "Severity of Harm", "P0", "P1", "PoH", "Risk Index", "Risk Control"]:
            if not str(parsed.get(k, "")).strip() and rec.get(k):
                parsed[k] = rec[k]
    # MAUDE enrichment
    if MAUDE_FETCH:
        maude_rows = get_maude_ha_rows(MAUDE_DEVICE, risk)
        for k in ["Hazard", "Hazardous Situation", "Harm", "Sequence of Events"]:
            if not str(parsed.get(k, "")).strip():
                for r in maude_rows:
                    if r.get(k):
                        parsed[k] = r[k]
                        break
    return parsed

# =======================================
# Main Inference
# =======================================
def infer_ha(requirements: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results = []
    for req in requirements:
        req_id = req.get("Requirement ID", "")
        req_text = req.get("Requirements", "")

        prompt = f"""Given the requirement below, fill in a Hazard Analysis JSON with:
Risk ID, Risk to Health, Hazard, Hazardous Situation, Harm, Sequence of Events, Severity of Harm, P0, P1, PoH, Risk Index, Risk Control.
Requirement: {req_text}
"""
        decoded = _generate_text(prompt)
        parsed = _extract_json(decoded) or {}

        # Always keep req id
        parsed["Requirement ID"] = req_id

        # Merge with RAG + MAUDE
        parsed = _merge_with_rag(parsed, parsed.get("Risk to Health", ""))

        # Risk control adjustment — paraphrase requirement slightly
        if not parsed.get("Risk Control") and req_text:
            parsed["Risk Control"] = f"Design and operational guidance derived from: {req_text}"

        results.append(parsed)
    return results
