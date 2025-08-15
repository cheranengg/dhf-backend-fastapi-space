# app/models/ha_infer.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# ----------------- runtime switches & locations -----------------
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"

# PEFT (adapter) strategy:
#   - we always load the base model (Mistral) + your fine-tuned adapter from HF
#   - no merging; you run exactly what trained well in Colab
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HA_ADAPTER_REPO = os.getenv("HA_ADAPTER_REPO", "")  # e.g. "cheranengg/dhf-ha-adapter"

# performance / device knobs
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
LOAD_4BIT = os.getenv("LOAD_4BIT", "1") == "1"  # keep memory small on L4

# --- cache & auth helpers ---
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


# ----------------- optional embeddings (best-effort) ---------------
_HAS_EMB = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# ----------------- model state -----------------------------------
_tokenizer = None  # type: ignore
_model = None  # type: ignore
_device = "cpu"
_emb: Optional["SentenceTransformer"] = None  # type: ignore


def _ensure_model():
    """
    Load base Mistral + your LoRA adapter (no merge).
    Uses slow tokenizer (SentencePiece) so protobuf isn't needed.
    Uses 4-bit quant on GPU if LOAD_4BIT=1 (recommended on HF Spaces L4).
    """
    global _tokenizer, _model, _device, _emb

    if _model is not None:
        return

    if not USE_HA_MODEL:
        # generation is disabled – do not try to load a model
        return

    if not HA_ADAPTER_REPO:
        raise RuntimeError("HA_ADAPTER_REPO is not set. Example: cheranengg/dhf-ha-adapter")

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from peft import PeftModel
    import torch

    # slow tokenizer -> avoids protobuf dependency
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False, **_token_cache_kwargs())
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    want_cuda = torch.cuda.is_available() and (not FORCE_CPU)
    _device = "cuda" if want_cuda else "cpu"

    quant_cfg = None
    dtype = torch.float16 if (want_cuda and not LOAD_4BIT) else None

    if want_cuda and LOAD_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto" if want_cuda else None,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
        **_token_cache_kwargs(),
    )

    _model = PeftModel.from_pretrained(base, HA_ADAPTER_REPO, **_token_cache_kwargs())
    if not want_cuda:
        _model.to("cpu")

    # extra guard
    try:
        _model.config.pad_token_id = _tokenizer.pad_token_id  # type: ignore
    except Exception:
        pass

    # embeddings (best-effort, optional)
    if _HAS_EMB:
        try:
            _emb = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR)  # type: ignore
        except Exception:
            _emb = None


# ----------------- utilities --------------------------------------
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

    # choose the last (most recent) candidate
    start, end = candidates[-1]
    return text[start:end]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    js = _balanced_json_block(text)
    if not js:
        return None
    # light cleanup
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
    """Nearest textual requirement (optional embeddings)."""
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


# ----------------- prompting --------------------------------------
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
    _ensure_model()
    if _model is None or _tokenizer is None:
        # If the model couldn't load (disabled or error), return empty dict
        return {}

    import torch

    prompt = _PROMPT.format(risk=risk)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)  # type: ignore
    with torch.no_grad():
        out = _model.generate(  # type: ignore
            **inputs,
            max_new_tokens=192,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    return _extract_json(decoded) or {}


# ----------------- public API --------------------------------------
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
    if USE_HA_MODEL:
        try:
            parsed = _gen_json_for_risk(risk)
        except Exception:
            parsed = {}

    severity, p0, p1, poh, risk_index = _calculate_risk_fields(parsed or {})
    hint = ((parsed.get("Hazardous Situation", "") or "") + " " + (parsed.get("Harm", "") or "")).strip()
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
    Returns list of HA rows (dicts) – either generated (if USE_HA_MODEL=1)
    or stable fallback rows.
    """
    # If generation is disabled entirely, do the quick fallback.
    if not USE_HA_MODEL:
        return _fallback_ha(requirements)

    # Try to warm-load model once; if it fails, use fallback.
    try:
        _ensure_model()
    except Exception:
        return _fallback_ha(requirements)

    out: List[Dict[str, Any]] = []
    for r in requirements:
        rid = str(r.get("Requirement ID") or "")
        # Keep it at 10 risks for speed unless overridden by env
        risks = _default_risks[: int(os.getenv("HA_RISKS_LIMIT", "10"))]
        for risk in risks:
            try:
                row = _gen_row_for_risk(requirements, risk, rid)
            except Exception:
                # row-level safety: never abort the whole request
                row = _gen_row_for_risk([], risk, rid)
            out.append(row)
    return out
