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
DO_SAMPLE = os.getenv("do_sample", "1") == "1"
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))

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
CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"


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
      - Prefer slow tokenizer to avoid protobuf dependency; set legacy=True when possible.
      - Fall back to fast tokenizer if slow fails.
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
        return _tokenizer
    except Exception as e:
        last_exc = e

    # fast tokenizer fallback
    try:
        tok = AutoTokenizer.from_pretrained(src, use_fast=True, **_token_cache_kwargs())
        _tokenizer = tok
        return _tokenizer
    except Exception as e:
        last_exc = e
        raise RuntimeError(f"HA tokenizer load failed from {src}: {last_exc}")


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

    # If neither adapter nor merged path is enabled, bail early.
    if not (USE_HA_ADAPTER or USE_HA_MODEL):
        _init_embeddings()
        return

    tok = _load_tokenizer()
    want_cuda = torch.cuda.is_available() and not FORCE_CPU
    dtype = torch.float16 if want_cuda else torch.float32

    if USE_HA_ADAPTER:
        # Try to use your helper if present; otherwise direct PEFT load.
        try:
            try:
                # Optional helper (your repo already has this)
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
        if want_cuda:
            _model.to("cuda")
        else:
            _model.to("cpu")
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

_severity_map = {
    "negligible": 1,
    "minor": 2,
    "moderate": 3,
    "serious": 4,
    "critical": 5,
}


def _balanced_json_block(text: str) -> Optional[str]:
    """
    Extract the last balanced {...} block from text without recursive regex.
    Works well for single-object generations.
    """
    if not text:
        return None
    opens: List[int] = []
    candidates: List[Tuple[int, int]] = []

    for i, ch in enumerate(text):
        if ch == "{":
            opens.append(i)
        elif ch == "}" and opens:
            start = opens.pop()
            candidates.append((start, i + 1))

    if not candidates:
        return None

    start, end = candidates[-1]
    return text[start:end]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    js = _balanced_json_block(text)
    if not js:
        return None
    # cleanup
    js = js.replace("'", '"').replace("\\n", " ")
    js = re.sub(r"\s+", " ", js)
    js = re.sub(r",\s*\}", "}", js)
    js = re.sub(r",\s*\]", "]", js)
    try:
        return json.loads(js)
    except Exception:
        return None


def _calculate_risk_fields(parsed: Dict[str, Any]) -> Tuple[int, str, str, str, str]:
    sev_txt = str(parsed.get("Severity of Harm", "Moderate")).lower()
    severity = _severity_map.get(sev_txt, 3)
    p0 = str(parsed.get("P0", "Medium"))
    p1 = str(parsed.get("P1", "Medium"))

    poh_matrix = {
        ("Very Low", "Very Low"): "Very Low",
        ("Very Low", "Low"): "Very Low",
        ("Very Low", "Medium"): "Low",
        ("Low", "Very Low"): "Very Low",
        ("Low", "Low"): "Low",
        ("Low", "Medium"): "Medium",
        ("Medium", "Medium"): "Medium",
        ("Medium", "High"): "High",
        ("High", "Medium"): "High",
        ("High", "High"): "High",
        ("Very High", "High"): "Very High",
        ("Very High", "Very High"): "Very High",
    }
    poh = poh_matrix.get((p0, p1), "Medium")

    if severity == 5 and poh in ("High", "Very High"):
        risk_index = "Extreme"
    elif severity >= 3 and poh in ("High", "Very High"):
        risk_index = "High"
    else:
        risk_index = "Medium"

    return severity, p0, p1, poh, risk_index


def _nearest_req_control(reqs: List[Dict[str, Any]], hint_text: str) -> str:
    """Find the nearest textual requirement (optional embeddings)."""
    if not _HAS_EMB or _emb is None:
        return "Refer to IEC 60601 and ISO 14971 risk controls"
    try:
        corpus = [str(r.get("Requirements") or "") for r in reqs]
        ids = [str(r.get("Requirement ID") or "") for r in reqs]
        if not corpus:
            return "Refer to IEC 60601 and ISO 14971 risk controls"
        vecs = _emb.encode(corpus, convert_to_numpy=True)  # type: ignore
        import numpy as np

        q = _emb.encode([hint_text or "risk control"], convert_to_numpy=True)  # type: ignore
        sims = vecs @ q[0] / (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(q[0]) + 1e-9))
        i = int(sims.argmax())
        return f"{corpus[i]} (Ref: {ids[i]})" if 0 <= i < len(corpus) else "Refer to IEC 60601 and ISO 14971 risk controls"
    except Exception:
        return "Refer to IEC 60601 and ISO 14971 risk controls"


# =========================
# Prompting
# =========================

_PROMPT = """You are generating Hazard Analysis content for an infusion pump.
Return ONLY one JSON object with the exact keys:
{{
  "Hazard": "...",
  "Hazardous Situation": "...",
  "Harm": "...",
  "Sequence of Events": "...",
  "Severity of Harm": "Negligible|Minor|Moderate|Serious|Critical",
  "P0": "Very Low|Low|Medium|High|Very High",
  "P1": "Very Low|Low|Medium|High|Very High"
}}

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
        temperature=0.3,
        top_p=0.9,
    )
    if NUM_BEAMS > 1 and not DO_SAMPLE:
        gen_kwargs.update(dict(num_beams=NUM_BEAMS, do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=True))

    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kwargs)  # type: ignore
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    return _extract_json(decoded) or {}


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
                    "severity_of_harm": "3",
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

    severity, p0, p1, poh, risk_index = _calculate_risk_fields(parsed or {})
    hint = (
        (parsed.get("Hazardous Situation", "") or "")
        + " "
        + (parsed.get("Harm", "") or "")
    ).strip()
    control = _nearest_req_control(requirements, hint)

    return {
        "requirement_id": rid,
        "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
        "risk_to_health": risk,
        "hazard": parsed.get("Hazard", "TBD"),
        "hazardous_situation": parsed.get("Hazardous Situation", "TBD"),
        "harm": parsed.get("Harm", "TBD"),
        "sequence_of_events": parsed.get("Sequence of Events", "TBD"),
        "severity_of_harm": str(severity),
        "p0": p0,
        "p1": p1,
        "poh": poh,
        "risk_index": risk_index,
        "risk_control": control,
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
