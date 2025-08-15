# app/models/ha_infer.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Env switches & locations
# =========================

# Prefer adapter if available
USE_HA_ADAPTER = os.getenv("USE_HA_ADAPTER", "0") == "1"
HA_ADAPTER_REPO = os.getenv("HA_ADAPTER_REPO", "")  # e.g., "cheranengg/dhf-ha-adapter"

# Fallback to merged model if enabled
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"
HA_MODEL_DIR = os.getenv("HA_MODEL_MERGED_DIR", os.getenv("HA_MODEL_DIR", ""))

# Base model tokenizer (and for adapter base weights)
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

# Generation knobs
HA_MAX_NEW_TOKENS = int(os.getenv("HA_MAX_NEW_TOKENS", "256"))
DO_SAMPLE = os.getenv("DO_SAMPLE", os.getenv("do_sample", "1")) == "1"
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))

# Device/loading
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
LOAD_4BIT = os.getenv("LOAD_4BIT", "0") == "1"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")

# HF auth/cache
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)
CACHE_DIR = (
    os.getenv("HF_HOME")
    or os.getenv("TRANSFORMERS_CACHE")
    or os.getenv("HUGGINGFACE_HUB_CACHE")
    or os.getenv("HF_HUB_CACHE")
    or "/tmp/hf"
)


def _token_cache_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw


# =========================
# Optional embeddings (best-effort)
# =========================
_HAS_EMB = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_EMB = True
except Exception:
    _HAS_EMB = False


# =========================
# Transformers guarded
# =========================
_tokenizer = None  # type: ignore
_model = None  # type: ignore
_DEVICE = "cpu"
_emb: Optional["SentenceTransformer"] = None  # type: ignore


def _load_tokenizer():
    """
    Tokenizer rules:
      - If using LoRA adapter, always load tokenizer from BASE_MODEL_ID.
      - Else, use HA_MODEL_DIR (merged) if set; fallback to BASE_MODEL_ID.
      - Set pad_token=eos_token for Mistral.
    """
    global _tokenizer
    from transformers import AutoTokenizer

    if _tokenizer is not None:
        return _tokenizer

    src = BASE_MODEL_ID if USE_HA_ADAPTER else (HA_MODEL_DIR or BASE_MODEL_ID)
    last_exc: Optional[Exception] = None

    # slow tokenizer first
    try:
        tok = AutoTokenizer.from_pretrained(src, use_fast=False, **_token_cache_kwargs())
        try:
            tok.legacy = True  # type: ignore[attr-defined]
        except Exception:
            pass
        _tokenizer = tok
    except Exception as e:
        last_exc = e
        # fast tokenizer fallback
        try:
            tok = AutoTokenizer.from_pretrained(src, use_fast=True, **_token_cache_kwargs())
            _tokenizer = tok
        except Exception as e2:
            last_exc = e2
            raise RuntimeError(f"HA tokenizer load failed from {src}: {last_exc}")

    # pad token safety for decoder-only
    if getattr(_tokenizer, "pad_token", None) is None:
        _tokenizer.pad_token = _tokenizer.eos_token  # type: ignore[attr-defined]

    return _tokenizer


def _load_model():
    """
    Load HA model:
      - If USE_HA_ADAPTER=1: load BASE model and attach LoRA adapter (HA_ADAPTER_REPO).
      - elif USE_HA_MODEL=1: load merged model from HA_MODEL_DIR.
      - else: leave _model=None (template-only fallback path).
    Also initializes MiniLM embeddings (best-effort) for requirement proximity.
    """
    global _model, _DEVICE, _emb

    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM

    # If neither adapter nor merged path is enabled, bail early (we'll use fallback rows).
    if not (USE_HA_ADAPTER or USE_HA_MODEL):
        _init_embeddings()
        return

    tok = _load_tokenizer()
    want_cuda = torch.cuda.is_available() and not FORCE_CPU
    dtype = torch.float16 if want_cuda else torch.float32

    if USE_HA_ADAPTER:
        try:
            try:
                # Optional helper if present in your repo
                from ._peft_loader import load_peft_model  # type: ignore

                _model = load_peft_model(
                    base_model_id=BASE_MODEL_ID,
                    adapter_id=HA_ADAPTER_REPO,
                    load_4bit=LOAD_4BIT,
                    offload_dir=OFFLOAD_DIR,
                    torch_dtype=dtype,
                    token_kwargs=_token_cache_kwargs(),
                )
            except Exception:
                # Direct PEFT path
                from peft import PeftModel

                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    **_token_cache_kwargs(),
                )
                _model = PeftModel.from_pretrained(base, HA_ADAPTER_REPO, **_token_cache_kwargs())
        except Exception as e:
            raise RuntimeError(f"Failed to load HA adapter '{HA_ADAPTER_REPO}': {e}")

    elif USE_HA_MODEL:
        if not HA_MODEL_DIR:
            raise RuntimeError("HA model path is not configured (HA_MODEL_MERGED_DIR).")
        _model = AutoModelForCausalLM.from_pretrained(
            HA_MODEL_DIR,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **_token_cache_kwargs(),
        )

    # pad token safety
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token  # type: ignore[attr-defined]

    _DEVICE = "cuda" if want_cuda else "cpu"
    if _model is not None:
        _model.to(_DEVICE)
        _model.eval()

    _init_embeddings()


def _init_embeddings():
    """Optional MiniLM embeddings for 'nearest requirement' control text."""
    global _emb
    if not _HAS_EMB:
        _emb = None
        return
    if _emb is not None:
        return
    try:
        _emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=CACHE_DIR)  # type: ignore
    except Exception:
        # tolerate embedder failure; we’ll fallback to generic controls
        _emb = None


# =========================
# Utilities
# =========================

_default_risks = [
    "Air Embolism",
    "Allergic response",
    "Infection",
    "Overdose",
    "Underdose",
    "Delay of therapy",
    "Environmental Hazard",
    "Incorrect Therapy",
    "Trauma",
    "Particulate",
]

# Canonicalization maps
_SEV_CANON = {
    "S1": "S1", "LOW": "S1", "NEGLIGIBLE": "S1", "MINIMAL": "S1",
    "S2": "S2", "MINOR": "S2", "MEDIUM": "S2", "MODERATE": "S2",
    "S3": "S3", "MAJOR": "S3", "HIGH": "S3", "SERIOUS": "S3",
    "S4": "S4", "CRITICAL": "S4", "VERY HIGH": "S4",
    "S5": "S4",  # coerce out-of-range to S4
}
# your stored numeric scale 1..5 will be derived from S1..S4 below
_SEV_NUM = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}

_P_CANON = {
    "VERY LOW": "Very Low", "VL": "Very Low",
    "LOW": "Low", "L": "Low",
    "MEDIUM": "Medium", "MID": "Medium", "M": "Medium",
    "HIGH": "High", "H": "High",
    "VERY HIGH": "Very High", "VH": "Very High",
}

def _canon_sev(v: str, default="S3") -> str:
    key = (v or "").strip().upper()
    return _SEV_CANON.get(key, default)

def _canon_p(v: str, default="Medium") -> str:
    key = (v or "").strip().upper()
    return _P_CANON.get(key, default)

def _balanced_json_block(text: str) -> Optional[str]:
    """
    Extract the first balanced {...} block from text (brace-aware).
    """
    if not text:
        return None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : i + 1]
    return None

def _repair_json_str(js: str) -> Optional[Dict[str, Any]]:
    """
    Lightweight repairs: smart quotes, trailing commas, collapsed whitespace.
    Try json, then optional json5 if present.
    """
    if not js:
        return None
    s = js
    # normalize quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # unify newlines & spaces
    s = s.replace("\\n", " ")
    s = re.sub(r"\s+", " ", s)
    # trailing commas
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # single quotes → double quotes (only for keys/strings heuristically)
    s = re.sub(r'(?<!\\)\'', '"', s)
    try:
        return json.loads(s)
    except Exception:
        try:
            import json5  # type: ignore
            return json5.loads(s)
        except Exception:
            return None

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    js = _balanced_json_block(text)
    if not js:
        return None
    return _repair_json_str(js)


def _calculate_risk_fields(parsed: Dict[str, Any]) -> Tuple[int, str, str, str, str]:
    # Canonicalize enums from model (robust to typos/case)
    sev_raw = str(parsed.get("Severity of Harm", "S3"))
    p0_raw = str(parsed.get("P0", "Medium"))
    p1_raw = str(parsed.get("P1", "Medium"))

    sev_c = _canon_sev(sev_raw, "S3")
    severity_num = _SEV_NUM.get(sev_c, 3)
    p0 = _canon_p(p0_raw, "Medium")
    p1 = _canon_p(p1_raw, "Medium")

    # POH matrix (code-authoritative)
    poh_matrix = {
        ("Very Low", "Very Low"): "Very Low",
        ("Very Low", "Low"): "Very Low",
        ("Very Low", "Medium"): "Low",
        ("Low", "Very Low"): "Very Low",
        ("Low", "Low"): "Low",
        ("Low", "Medium"): "Medium",
        ("Medium", "Very Low"): "Low",
        ("Medium", "Low"): "Medium",
        ("Medium", "Medium"): "Medium",
        ("Medium", "High"): "High",
        ("High", "Medium"): "High",
        ("High", "High"): "High",
        ("Very High", "High"): "Very High",
        ("Very High", "Very High"): "Very High",
    }
    poh = poh_matrix.get((p0, p1), "Medium")

    if severity_num >= 4 and poh in ("High", "Very High"):
        risk_index = "Extreme"
    elif severity_num >= 3 and poh in ("High", "Very High"):
        risk_index = "High"
    else:
        risk_index = "Medium"

    # return numeric severity for your schema, plus canonical enums
    return severity_num, p0, p1, poh, risk_index


def _nearest_req_control(reqs: List[Dict[str, Any]], hint_text: str) -> Tuple[str, str]:
    """
    Find a requirement to use as Risk Control text.
    Returns (control_text, source_flag).
    Fallback order: nearest (cos>=0.55) → second best within Δ0.03 → generic.
    """
    generic = "Refer to IEC 60601 and ISO 14971 risk controls"
    if not reqs:
        return generic, "generic"
    if not _HAS_EMB or _emb is None:
        return generic, "generic"

    try:
        corpus = [str(r.get("Requirements") or "") for r in reqs]
        ids = [str(r.get("Requirement ID") or "") for r in reqs]
        import numpy as np

        # Encode corpus & query
        vecs = _emb.encode(corpus, convert_to_numpy=True)  # type: ignore
        q = _emb.encode([hint_text or "risk control"], convert_to_numpy=True)[0]  # type: ignore

        # Cosine similarity
        denom = (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(q) + 1e-9))
        sims = (vecs @ q) / np.maximum(denom, 1e-9)

        top = int(sims.argmax())
        top_sim = float(sims[top]) if sims.size else 0.0

        if sims.size > 1:
            second = int(np.argsort(-sims)[1])
            second_sim = float(sims[second])
        else:
            second, second_sim = top, top_sim

        TH = 0.55
        DELTA = 0.03
        if top_sim >= TH:
            return f"{corpus[top]} (Ref: {ids[top]})", "nearest"
        elif second_sim >= TH - DELTA:
            return f"{corpus[second]} (Ref: {ids[second]})", "second_best"
        else:
            return generic, "generic"
    except Exception:
        return generic, "generic"


# =========================
# Prompting
# =========================

# Strict JSON, single object, schema-locked
_PROMPT = """You are generating ONE Hazard Analysis record as STRICT JSON ONLY.
Return EXACTLY one JSON object and nothing else.

Schema:
{{
  "Hazard": "string",
  "Hazardous Situation": "string",
  "Harm": "string",
  "Sequence of Events": "string",
  "Severity of Harm": "S1|S2|S3|S4",
  "P0": "Very Low|Low|Medium|High|Very High",
  "P1": "Very Low|Low|Medium|High|Very High"
}}

Rules:
- Keep each field concise (<= 20 words).
- Do not include explanations, markdown, or extra text.
- Use S1..S4 for Severity of Harm.

Context:
Risk to Health: {risk}
"""


def _gen_json_for_risk(risk: str) -> Dict[str, Any]:
    _load_model()
    if _model is None or _tokenizer is None:
        return {}

    import torch

    prompt = _PROMPT.format(risk=risk)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_DEVICE)  # type: ignore

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=HA_MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
    )

    # Two stable profiles:
    if DO_SAMPLE:
        gen_kwargs.update(dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, num_beams=1))
    else:
        gen_kwargs.update(dict(do_sample=False, num_beams=NUM_BEAMS))

    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kwargs)  # type: ignore
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    parsed = _extract_json(decoded) or {}
    return parsed


# =========================
# Public API
# =========================

def _fallback_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in requirements:
        rid = str(r.get("Requirement ID") or "")
        for risk in _default_risks:
            rows.append(
                {
                    "requirement_id": rid,
                    "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
                    "risk_to_health": risk,
                    "hazard": "TBD",
                    "hazardous_situation": "TBD",
                    "harm": "TBD",
                    "sequence_of_events": "TBD",
                    "severity_of_harm": "3",          # numeric string (your schema)
                    "p0": "Medium",
                    "p1": "Medium",
                    "poh": "Medium",
                    "risk_index": "Medium",
                    "risk_control": "Refer to IEC 60601 and ISO 14971 risk controls",
                }
            )
    return rows


def _gen_row_for_risk(
    requirements: List[Dict[str, Any]], risk: str, rid: str
) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if USE_HA_ADAPTER or USE_HA_MODEL:
        try:
            parsed = _gen_json_for_risk(risk)
        except Exception:
            parsed = {}

    severity_num, p0, p1, poh, risk_index = _calculate_risk_fields(parsed or {})

    # Use HS+Harm as hint for nearest requirement mapping
    hint = (
        (parsed.get("Hazardous Situation", "") or "")
        + " "
        + (parsed.get("Harm", "") or "")
    ).strip()

    control, _rc_source = _nearest_req_control(requirements, hint)

    return {
        "requirement_id": rid,
        "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
        "risk_to_health": risk,
        "hazard": parsed.get("Hazard", "TBD"),
        "hazardous_situation": parsed.get("Hazardous Situation", "TBD"),
        "harm": parsed.get("Harm", "TBD"),
        "sequence_of_events": parsed.get("Sequence of Events", "TBD"),
        "severity_of_harm": str(severity_num),  # keep numeric string for downstream
        "p0": p0,
        "p1": p1,
        "poh": poh,
        "risk_index": risk_index,
        "risk_control": control,
        # Optional debug fields (comment-out if you must keep schema minimal):
        # "_rc_source": _rc_source,
        # "_repaired": parsed != {} and isinstance(parsed, dict),
    }


def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    requirements: list of dicts with keys "Requirement ID" and "Requirements".
    Returns list of HA rows (dicts) – using adapter/merged model if enabled,
    otherwise stable fallback rows.
    """
    # If both adapter and merged are disabled, do the quick fallback.
    if not (USE_HA_ADAPTER or USE_HA_MODEL):
        return _fallback_ha(requirements)

    # Try to load model once; if it fails, use fallback.
    try:
        _load_model()
    except Exception:
        return _fallback_ha(requirements)

    out: List[Dict[str, Any]] = []
    for r in requirements:
        rid = str(r.get("Requirement ID") or "")
        for risk in _default_risks:
            try:
                row = _gen_row_for_risk(requirements, risk, rid)
            except Exception:
                row = _gen_row_for_risk([], risk, rid)
            out.append(row)
    return out
