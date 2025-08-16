# app/models/ha_infer.py
from __future__ import annotations

import os
import re
import gc
import json
import time
import random
from typing import List, Dict, Any, Optional

import torch

# ---------------------------
# Environment / toggles
# ---------------------------
USE_HA_ADAPTER: bool = os.getenv("USE_HA_ADAPTER", "1") == "1"
BASE_MODEL_ID: str = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HA_ADAPTER_REPO: str = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")

# RAG (synthetic HA) – optional, used for seeds + paraphrase
HA_RAG_PATH: str = os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl")

# Local MAUDE (fast, no network)
MAUDE_LOCAL_PATH: str = os.getenv(
    "MAUDE_LOCAL_JSONL", "app/rag_sources/sigma_spectrum_maude.jsonl"
)
# When True, we never hit the internet.
MAUDE_LOCAL_ONLY: bool = os.getenv("MAUDE_LOCAL_ONLY", "1") == "1"

# Fraction of rows where we enrich from MAUDE (0.0–1.0)
MAUDE_FRACTION: float = float(os.getenv("MAUDE_FRACTION", "0.70"))

# Generation limits (keep small = faster, safer)
HA_MAX_NEW_TOKENS: int = int(os.getenv("HA_MAX_NEW_TOKENS", "192"))
NUM_BEAMS: int = int(os.getenv("NUM_BEAMS", "1"))
DO_SAMPLE: bool = os.getenv("do_sample", "1") == "1"
TOP_P: float = float(os.getenv("HA_TOP_P", "0.90"))
TEMPERATURE: float = float(os.getenv("HA_TEMPERATURE", "0.35"))
REPETITION_PENALTY: float = float(os.getenv("HA_REPETITION_PENALTY", "1.05"))

# Safety / speed
FORCE_CPU: bool = os.getenv("FORCE_CPU", "0") == "1"
OFFLOAD_DIR: str = os.getenv("OFFLOAD_DIR", "/tmp/offload")
CACHE_DIR: str = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"

# Optional paraphrase control to avoid verbatim copies from RAG
PARAPHRASE_FROM_RAG: bool = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
PARAPHRASE_MAX_WORDS: int = int(os.getenv("PARAPHRASE_MAX_WORDS", "22"))

# Cap rows defensively (backend already caps, this is an extra guard)
ROW_LIMIT: int = int(os.getenv("HA_ROW_LIMIT", "5"))

# ---------------------------
# Globals
# ---------------------------
_tokenizer = None
_model = None
_RAG_DB: List[Dict[str, Any]] = []
_MAUDE_ROWS: List[Dict[str, Any]] = []  # parsed local MAUDE jsonl (fast)


# ---------------------------
# Utilities
# ---------------------------
def _jsonl_load(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return rows


def _maybe_truncate_words(s: str, max_words: int) -> str:
    toks = re.split(r"\s+", s.strip())
    if len(toks) <= max_words:
        return s.strip()
    return " ".join(toks[:max_words]).strip()


def _paraphrase_sentence(s: str) -> str:
    """
    Extremely light "paraphrase" to avoid verbatim copies:
    - lowercases, tweaks a few synonyms, trims length
    (kept simple so it’s deterministic & fast)
    """
    out = s.strip()
    out = out.replace("shall", "must")
    out = out.replace("patient", "the patient")
    out = out.replace("device", "system")
    out = out.replace("ensure", "make sure")
    out = out.replace("maintain", "keep")
    out = _maybe_truncate_words(out, PARAPHRASE_MAX_WORDS)
    return out


def _gc_cuda():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------
# Model loader (Adapter, 4-bit if available)
# ---------------------------
def _load_model():
    global _tokenizer, _model

    if _model is not None and _tokenizer is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "auto" if (torch.cuda.is_available() and not FORCE_CPU) else None
    dtype = torch.float16 if (torch.cuda.is_available() and not FORCE_CPU) else torch.float32

    # Try 4-bit load (best memory footprint). Fall back gracefully.
    load_kwargs = dict(
        device_map=device_map,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
        offload_folder=OFFLOAD_DIR if device_map == "auto" else None,
    )
    try:
        from peft import PeftModel
        bnb_ok = False
        try:
            from transformers import BitsAndBytesConfig  # noqa
            bnb_ok = True
        except Exception:
            bnb_ok = False

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        if bnb_ok:
            # 4-bit quant
            load_kwargs.update(dict(load_in_4bit=True))
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
        if USE_HA_ADAPTER:
            _model = PeftModel.from_pretrained(base, HA_ADAPTER_REPO, cache_dir=CACHE_DIR)
        else:
            _model = base
        try:
            _model.config.pad_token_id = _tokenizer.pad_token_id
        except Exception:
            pass

        if device_map is None:
            _model.to("cpu" if (FORCE_CPU or not torch.cuda.is_available()) else "cuda")

    except Exception:
        # Fallback: CPU fp32 – slow but won't OOM.
        _gc_cuda()
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        try:
            _model.config.pad_token_id = _tokenizer.pad_token_id
        except Exception:
            pass
        _model.to("cpu")


# ---------------------------
# RAG + MAUDE loaders
# ---------------------------
def _load_rag_once():
    """Load synthetic HA jsonl once."""
    global _RAG_DB
    if _RAG_DB:
        return
    _RAG_DB = _jsonl_load(HA_RAG_PATH)


def _load_maude_once():
    """Load local MAUDE jsonl once (fast)."""
    global _MAUDE_ROWS
    if _MAUDE_ROWS:
        return
    _MAUDE_ROWS = _jsonl_load(MAUDE_LOCAL_PATH)


def _maude_snippets(k: int = 6) -> List[str]:
    """
    Pick k short narratives from local MAUDE rows.
    We prioritize description-like fields if present.
    """
    _load_maude_once()
    if not _MAUDE_ROWS:
        return []

    # Collect texts
    texts: List[str] = []
    for r in _MAUDE_ROWS:
        # FDA’s fields vary; we check several common ones
        for key in (
            "event_description",
            "device_problem_text",
            "event_text",
            "text",
            "event",
        ):
            val = r.get(key)
            if isinstance(val, str) and len(val.strip()) > 20:
                texts.append(val.strip())

        # nested MDR text (if pre-flattened)
        mdr = r.get("mdr_text", [])
        if isinstance(mdr, list):
            for t in mdr:
                txt = t.get("text") if isinstance(t, dict) else None
                if isinstance(txt, str) and len(txt.strip()) > 20:
                    texts.append(txt.strip())

    if not texts:
        return []

    random.shuffle(texts)
    keep = []
    for t in texts[: 4 * k]:
        # short-ish
        keep.append(_maybe_truncate_words(t, 40))
        if len(keep) >= k:
            break
    return keep


# ---------------------------
# Prompt + generation
# ---------------------------
_SCHEMA = """
Return a compact JSON object with keys:
- "risk_id": a unique id like "HA-####"
- "risk_to_health": short phrase (e.g., "Air Embolism")
- "hazard": short phrase
- "hazardous_situation": short phrase
- "harm": short phrase
- "sequence_of_events": short phrase
- "severity_of_harm": one of "1","2","3","4","5"
- "p0": one of "Very Low","Low","Medium","High","Very High"
- "p1": same scale
- "poh": same scale
- "risk_index": one of "Low","Medium","High","Very High"
- "risk_control": succinct preventive/mitigating statement; do NOT mention requirement IDs.
"""

def _build_prompt(req_text: str, rag_seed: Optional[str], maude_bits: List[str]) -> str:
    context = ""
    if rag_seed:
        rag_line = rag_seed
        if PARAPHRASE_FROM_RAG:
            rag_line = _paraphrase_sentence(rag_seed)
        context += f"\nRAG seed (paraphrased): {rag_line}"
    if maude_bits:
        context += "\nMAUDE snippets:\n- " + "\n- ".join(maude_bits)

    prompt = f"""You are a safety engineer performing Hazard Analysis for an infusion pump.

Requirement: {req_text}

Use professional language. Prefer concise phrases.
If some items truly cannot be inferred, pick reasonable, generic infusion-pump patterns.

{_SCHEMA}

{context}

Now output ONLY the JSON:"""
    return prompt


def _generate_json(prompt: str) -> Dict[str, Any]:
    _load_model()
    device = _model.device  # type: ignore

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)  # type: ignore
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = _model.generate(  # type: ignore
            **inputs,
            max_new_tokens=HA_MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
        )

    txt = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore

    # Extract JSON blob
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if m:
        blob = m.group(0)
    else:
        # Best effort: sometimes models echo the prompt – take tail braces
        braces = txt.find("{")
        blob = txt[braces:] if braces >= 0 else "{}"

    # Coerce common JSON mistakes
    blob = re.sub(r",\s*}", "}", blob)
    blob = re.sub(r",\s*\]", "]", blob)

    try:
        data = json.loads(blob)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


# ---------------------------
# Post processing / defaults
# ---------------------------
def _ensure_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    def s(k, default="TBD"):
        v = d.get(k)
        return str(v).strip() if v is not None and str(v).strip() else default

    out = {
        "risk_id": s("risk_id", f"HA-{random.randint(1000, 9999)}"),
        "risk_to_health": s("risk_to_health", "Air Embolism"),
        "hazard": s("hazard", "Device malfunction"),
        "hazardous_situation": s("hazardous_situation", "Patient exposed to device fault"),
        "harm": s("harm", "Adverse physiological effects"),
        "sequence_of_events": s("sequence_of_events", "Improper setup or device issue led to patient exposure"),
        "severity_of_harm": s("severity_of_harm", "3"),
        "p0": s("p0", "Medium"),
        "p1": s("p1", "Medium"),
        "poh": s("poh", "Low"),
        "risk_index": s("risk_index", "Medium"),
        "risk_control": s("risk_control", "System operating manual provides clear purging and setup cautions/warnings"),
    }
    # Normalize severity to "1".."5"
    sev = out["severity_of_harm"]
    if not re.fullmatch(r"[1-5]", sev):
        out["severity_of_harm"] = "3"
    for k in ("p0", "p1", "poh"):
        if out[k] not in {"Very Low", "Low", "Medium", "High", "Very High"}:
            out[k] = "Medium"
    if out["risk_index"] not in {"Low", "Medium", "High", "Very High"}:
        out["risk_index"] = "Medium"
    return out


# ---------------------------
# Public API
# ---------------------------
def infer_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: [{"Requirement ID": "...", "Requirements": "..."} ...]
    Output rows with keys used by the UI/export pipeline.
    """
    _load_rag_once()

    rows: List[Dict[str, Any]] = []
    if not isinstance(requirements, list):
        return rows

    # Extra hard cap here for snappy testing
    reqs = requirements[:ROW_LIMIT]

    for r in reqs:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "").strip() or "REQ-XXX"
        req_text = str(r.get("Requirements") or r.get("requirements") or "").strip()

        # Pick a small RAG seed line (optional), then lightly paraphrase later
        rag_seed = None
        if _RAG_DB:
            try:
                rag_seed = random.choice(_RAG_DB).get("Hazard", None)
            except Exception:
                rag_seed = None

        # Use MAUDE snippets for ~MAUDE_FRACTION of rows
        maude_bits: List[str] = []
        if MAUDE_LOCAL_ONLY and random.random() < MAUDE_FRACTION:
            maude_bits = _maude_snippets(k=4)

        prompt = _build_prompt(req_text, rag_seed, maude_bits)
        raw = _generate_json(prompt)
        data = _ensure_fields(raw)

        # Package in our canonical schema
        rows.append({
            "requirement_id": rid,
            "risk_id": data["risk_id"],
            "risk_to_health": data["risk_to_health"],
            "hazard": data["hazard"],
            "hazardous_situation": data["hazardous_situation"],
            "harm": data["harm"],
            "sequence_of_events": data["sequence_of_events"],
            "severity_of_harm": data["severity_of_harm"],
            "p0": data["p0"],
            "p1": data["p1"],
            "poh": data["poh"],
            "risk_index": data["risk_index"],
            # Keep risk control high-level, never mention IDs. If we had a RAG seed and paraphrasing is enabled,
            # it will already be concise and non-verbatim.
            "risk_control": data["risk_control"],
        })

    return rows
