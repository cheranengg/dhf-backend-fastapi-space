# app/models/ha_infer.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# ================================================================
# Environment switches & locations
# ================================================================
USE_HA_ADAPTER = os.getenv("USE_HA_ADAPTER", "0") == "1"
HA_ADAPTER_REPO = os.getenv("HA_ADAPTER_REPO", "")

USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"
HA_MODEL_DIR = os.getenv("HA_MODEL_MERGED_DIR", os.getenv("HA_MODEL_DIR", ""))

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

HA_MAX_NEW_TOKENS = int(os.getenv("HA_MAX_NEW_TOKENS", "256"))
DO_SAMPLE = os.getenv("DO_SAMPLE", os.getenv("do_sample", "1")) == "1"
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))

FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
LOAD_4BIT = os.getenv("LOAD_4BIT", "0") == "1"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")

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


# ================================================================
# Optional embeddings (best-effort)
# ================================================================
_HAS_EMB = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False


# ================================================================
# Global state
# ================================================================
_tokenizer = None  # type: ignore
_model = None      # type: ignore
_DEVICE = "cpu"
_emb: Optional["SentenceTransformer"] = None  # type: ignore

# ---------- lightweight RAG ----------
_RAG_DB: List[Dict[str, Any]] = []
_RAG_TEXTS: List[str] = []
_RAG_EMB = None
RAG_SYNTHETIC_PATH = ""  # resolved at import via _resolve_rag_path()


# ================================================================
# RAG path resolution & loader
# ================================================================
def _resolve_rag_path() -> str:
    """
    Find the RAG JSONL file. Priority:
      1) HA_RAG_PATH (absolute OR relative to project root/app/)
      2) app/rag_sources/ha_synthetic.jsonl (relative to this file's parent)
      3) rag_sources/ha_synthetic.jsonl (project root)
      4) /workspace/app/app/rag_sources/ha_synthetic.jsonl  (HF Spaces common)
      5) /workspace/app/rag_sources/ha_synthetic.jsonl      (HF Spaces alt)
      6) /app/rag_sources/ha_synthetic.jsonl                (Docker alt)
    Returns first existing path; otherwise returns the best-guess default.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))               # .../app/models
    app_dir  = os.path.normpath(os.path.join(base_dir, ".."))           # .../app
    root_dir = os.path.normpath(os.path.join(app_dir, ".."))            # project root

    env_path = (os.getenv("HA_RAG_PATH", "") or "").strip()
    candidates: List[str] = []
    if env_path:
        if os.path.isabs(env_path):
            candidates.append(env_path)
        else:
            # try relative to app/ and project root
            candidates.append(os.path.normpath(os.path.join(app_dir, env_path)))
            candidates.append(os.path.normpath(os.path.join(root_dir, env_path)))

    # defaults
    candidates.extend([
        os.path.normpath(os.path.join(app_dir, "rag_sources", "ha_synthetic.jsonl")),
        os.path.normpath(os.path.join(root_dir, "rag_sources", "ha_synthetic.jsonl")),
        "/workspace/app/app/rag_sources/ha_synthetic.jsonl",
        "/workspace/app/rag_sources/ha_synthetic.jsonl",
        "/app/rag_sources/ha_synthetic.jsonl",
    ])

    for p in candidates:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            pass
    # last candidate (best guess)
    return candidates[0]


# Map many possible key spellings → canonical keys used internally
_CANON_KEYS = {
    "risk id": "Risk ID",
    "risk to health": "Risk to Health",
    "hazard": "Hazard",
    "hazardous situation": "Hazardous Situation",  # accepts "Hazardous situation"
    "harm": "Harm",
    "sequence of events": "Sequence of Events",
    "severity of harm": "Severity of Harm",
    "p0": "P0",
    "p1": "P1",
    "poh": "PoH",  # accept PoH/POH
    "risk index": "Risk Index",
    "risk control": "Risk Control",
}

def _canon_record_keys(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a JSONL record to expected keys + tidy values."""
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        ck = _CANON_KEYS.get(str(k).strip().lower(), k)
        out[ck] = v
    # strip trailing spaces in strings
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = v.strip()

    # numeric severity → S1..S4
    sev = out.get("Severity of Harm")
    if isinstance(sev, (int, float)) or (isinstance(sev, str) and sev.strip().isdigit()):
        n = int(sev)
        n = max(1, min(4, n))
        out["Severity of Harm"] = f"S{n}"
    elif isinstance(sev, str):
        s = sev.strip().upper()
        if s in {"S1", "S2", "S3", "S4"}:
            out["Severity of Harm"] = s

    # normalize P0/P1 buckets
    def _norm_p(x: Any) -> str:
        if isinstance(x, (int, float)) and 0 <= int(x) <= 4:
            return ["Very Low", "Low", "Medium", "High", "Very High"][int(x)]
        s = str(x or "").strip().upper()
        if s in {"VL", "VERY LOW", "0"}: return "Very Low"
        if s in {"L", "LOW", "1"}:       return "Low"
        if s in {"M", "MID", "MEDIUM", "2"}: return "Medium"
        if s in {"H", "HIGH", "3"}:      return "High"
        if s in {"VH", "VERY HIGH", "4"}:return "Very High"
        return "Medium"
    out["P0"] = _norm_p(out.get("P0", "Medium"))
    out["P1"] = _norm_p(out.get("P1", "Medium"))
    return out


def _load_rag_db(path: Optional[str] = None):
    """Load synthetic HA JSONL into in-memory store + build embed corpus."""
    global _RAG_DB, _RAG_TEXTS, _RAG_EMB, RAG_SYNTHETIC_PATH
    if _RAG_DB:
        return
    RAG_SYNTHETIC_PATH = path or _resolve_rag_path()
    exists = os.path.exists(RAG_SYNTHETIC_PATH)
    if not exists:
        print({"ha_rag": "missing", "path": RAG_SYNTHETIC_PATH})
        return

    loaded = 0
    with open(RAG_SYNTHETIC_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                try:
                    import json5  # type: ignore
                    rec = json5.loads(line)
                except Exception:
                    continue
            rec = _canon_record_keys(rec)
            _RAG_DB.append(rec)
            _RAG_TEXTS.append(" ".join([
                str(rec.get("Risk to Health","")),
                str(rec.get("Hazard","")),
                str(rec.get("Hazardous Situation","")),
                str(rec.get("Harm","")),
                str(rec.get("Sequence of Events","")),
                str(rec.get("Risk Control","")),
            ]))
            loaded += 1

    if _HAS_EMB:
        try:
            _RAG_EMB = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=CACHE_DIR
            )  # type: ignore
        except Exception:
            _RAG_EMB = None

    print({"ha_rag": "loaded", "path": RAG_SYNTHETIC_PATH, "rows": loaded})


def _rag_lookup(risk: str, requirement_text: str = "") -> Optional[Dict[str, Any]]:
    """Return best-matching synthetic HA row for this risk (or None)."""
    if not _RAG_DB:
        return None
    # 1) exact risk match
    m = [r for r in _RAG_DB if str(r.get("Risk to Health","")).strip().lower() == risk.lower()]
    if m:
        return m[0]
    # 2) semantic
    if not (_HAS_EMB and _RAG_EMB is not None):
        return None
    try:
        import numpy as np  # type: ignore
        corpus_emb = _RAG_EMB.encode(_RAG_TEXTS, convert_to_numpy=True)  # type: ignore
        q = _RAG_EMB.encode([risk + " " + requirement_text], convert_to_numpy=True)[0]  # type: ignore
        sims = (corpus_emb @ q) / (np.linalg.norm(corpus_emb, axis=1) * (np.linalg.norm(q) + 1e-9))
        i = int(sims.argmax())
        if float(sims[i]) >= 0.55:
            return _RAG_DB[i]
    except Exception:
        pass
    return None


# ================================================================
# Tokenizer / Model loaders
# ================================================================
def _load_tokenizer():
    global _tokenizer
    from transformers import AutoTokenizer

    if _tokenizer is not None:
        return _tokenizer

    src = BASE_MODEL_ID if USE_HA_ADAPTER else (HA_MODEL_DIR or BASE_MODEL_ID)
    try:
        tok = AutoTokenizer.from_pretrained(src, use_fast=False, **_token_cache_kwargs())
        try:
            tok.legacy = True  # type: ignore[attr-defined]
        except Exception:
            pass
        _tokenizer = tok
    except Exception:
        tok = AutoTokenizer.from_pretrained(src, use_fast=True, **_token_cache_kwargs())
        _tokenizer = tok

    if getattr(_tokenizer, "pad_token", None) is None:
        _tokenizer.pad_token = _tokenizer.eos_token  # type: ignore[attr-defined]
    return _tokenizer


def _init_embeddings():
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


def _load_model():
    global _model, _DEVICE, _emb
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM

    if not (USE_HA_ADAPTER or USE_HA_MODEL):
        _init_embeddings()
        _load_rag_db()
        return

    tok = _load_tokenizer()
    want_cuda = torch.cuda.is_available() and not FORCE_CPU
    dtype = torch.float16 if want_cuda else torch.float32

    if USE_HA_ADAPTER:
        try:
            try:
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

    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token  # type: ignore[attr-defined]

    _DEVICE = "cuda" if want_cuda else "cpu"
    if _model is not None:
        _model.to(_DEVICE)
        _model.eval()

    _init_embeddings()
    _load_rag_db()


# ================================================================
# Utilities
# ================================================================
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

_SEV_NUM = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}

_P_CANON = {
    "VERY LOW": "Very Low", "VL": "Very Low", "0": "Very Low",
    "LOW": "Low", "L": "Low", "1": "Low",
    "MEDIUM": "Medium", "MID": "Medium", "M": "Medium", "2": "Medium",
    "HIGH": "High", "H": "High", "3": "High",
    "VERY HIGH": "Very High", "VH": "Very High", "4": "Very High",
}

def _canon_sev(v: str, default="S3") -> str:
    s = str(v or "").strip().upper()
    if s.isdigit():
        n = max(1, min(4, int(s)))
        return f"S{n}"
    if s in {"S1","S2","S3","S4"}:
        return s
    if s in {"NEGLIGIBLE","LOW"}: return "S1"
    if s in {"MINOR","MODERATE"}: return "S2"
    if s in {"SERIOUS","MAJOR","HIGH"}: return "S3"
    if s in {"CRITICAL","VERY HIGH"}: return "S4"
    return default

def _canon_p(v: str, default="Medium") -> str:
    key = (str(v or "")).strip().upper()
    return _P_CANON.get(key, default)

def _balanced_json_block(text: str) -> Optional[str]:
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
    if not js:
        return None
    s = js
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = s.replace("\\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
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
    sev_raw = str(parsed.get("Severity of Harm", "S3"))
    p0_raw = str(parsed.get("P0", "Medium"))
    p1_raw = str(parsed.get("P1", "Medium"))

    sev_c = _canon_sev(sev_raw, "S3")
    severity_num = _SEV_NUM.get(sev_c, 3)
    p0 = _canon_p(p0_raw, "Medium")
    p1 = _canon_p(p1_raw, "Medium")

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

    return severity_num, p0, p1, poh, risk_index


def _nearest_req_control(reqs: List[Dict[str, Any]], hint_text: str) -> Tuple[str, str]:
    generic = "Refer to IEC 60601 and ISO 14971 risk controls"
    if not reqs:
        return generic, "generic"
    if not _HAS_EMB or _emb is None:
        return generic, "generic"
    try:
        corpus = [str(r.get("Requirements") or "") for r in reqs]
        ids = [str(r.get("Requirement ID") or "") for r in reqs]
        import numpy as np  # type: ignore
        vecs = _emb.encode(corpus, convert_to_numpy=True)  # type: ignore
        q = _emb.encode([hint_text or "risk control"], convert_to_numpy=True)[0]  # type: ignore
        sims = (vecs @ q) / (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(q) + 1e-9))
        top = int(sims.argmax())
        top_sim = float(sims[top]) if sims.size else 0.0
        order = list(reversed(list(sims.argsort()))) if sims.size > 1 else [top]
        second = order[1] if len(order) > 1 else top
        second_sim = float(sims[second])
        TH, DELTA = 0.55, 0.03
        if top_sim >= TH:
            return f"{corpus[top]} (Ref: {ids[top]})", "nearest"
        elif second_sim >= TH - DELTA:
            return f"{corpus[second]} (Ref: {ids[second]})", "second_best"
        else:
            return generic, "generic"
    except Exception:
        return generic, "generic"


# ================================================================
# Prompting
# ================================================================
_PROMPT = """You are generating ONE Hazard Analysis record as STRICT JSON ONLY.
Return EXACTLY one JSON object and nothing else.

Schema:
{
  "Hazard": "string",
  "Hazardous Situation": "string",
  "Harm": "string",
  "Sequence of Events": "string",
  "Severity of Harm": "S1|S2|S3|S4",
  "P0": "Very Low|Low|Medium|High|Very High",
  "P1": "Very Low|Low|Medium|High|Very High"
}

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
    gen_kwargs: Dict[str, Any] = dict(max_new_tokens=HA_MAX_NEW_TOKENS, repetition_penalty=REPETITION_PENALTY)
    if DO_SAMPLE:
        gen_kwargs.update(dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, num_beams=1))
    else:
        gen_kwargs.update(dict(do_sample=False, num_beams=NUM_BEAMS))
    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kwargs)  # type: ignore
    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    return _extract_json(decoded) or {}


# ================================================================
# Public API
# ================================================================
def _fallback_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in requirements:
        rid = str(r.get("Requirement ID") or "")
        for risk in _default_risks:
            rows.append({
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
            })
    return rows


def _merge_with_rag_if_needed(parsed: Dict[str, Any], risk: str, requirement_text: str) -> Dict[str, Any]:
    need_rag = (
        not parsed
        or all(not str(parsed.get(k, "")).strip() for k in ["Hazard", "Hazardous Situation", "Harm", "Sequence of Events"])
    )
    rag = _rag_lookup(risk, requirement_text) if need_rag else None
    if not rag:
        return parsed
    for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events","Severity of Harm","P0","P1","Risk Control","PoH","Risk Index"]:
        if not str(parsed.get(k, "")).strip():
            parsed[k] = rag.get(k, parsed.get(k))
    return parsed


def _gen_row_for_risk(requirements: List[Dict[str, Any]], risk: str, rid: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if USE_HA_ADAPTER or USE_HA_MODEL:
        try:
            parsed = _gen_json_for_risk(risk)
        except Exception:
            parsed = {}

    req_text = (requirements[0].get("Requirements", "") if requirements else "")
    parsed = _merge_with_rag_if_needed(parsed, risk, req_text)

    # Debug print per row (Space logs)
    try:
        print({
            "ha_row_debug": True,
            "rag_rows": len(_RAG_DB),
            "adapter": USE_HA_ADAPTER,
            "merged": USE_HA_MODEL,
            "req_id": rid,
            "risk": risk,
            "filled": {k: bool(str(parsed.get(k,"")).strip())
                       for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events","Risk Control"]}
        })
    except Exception:
        pass

    severity_num, p0, p1, poh, risk_index = _calculate_risk_fields(parsed or {})
    rc_from_model = (parsed.get("Risk Control") or "").strip()
    hint = ((parsed.get("Hazardous Situation","") or "") + " " + (parsed.get("Harm","") or "")).strip()
    control, _ = _nearest_req_control(requirements, hint)
    if rc_from_model and len(rc_from_model) > 6:
        control = rc_from_model
    poh = parsed.get("PoH", poh)
    risk_index = parsed.get("Risk Index", risk_index)

    return {
        "requirement_id": rid,
        "risk_id": f"HA-{abs(hash(risk + rid)) % 10_000:04}",
        "risk_to_health": risk,
        "hazard": parsed.get("Hazard", "TBD"),
        "hazardous_situation": parsed.get("Hazardous Situation", "TBD"),
        "harm": parsed.get("Harm", "TBD"),
        "sequence_of_events": parsed.get("Sequence of Events", "TBD"),
        "severity_of_harm": str(severity_num),
        "p0": p0,
        "p1": p1,
        "poh": poh,
        "risk_index": risk_index,
        "risk_control": control,
    }


def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    requirements: list[{"Requirement ID","Requirements"}].
    Returns HA rows – uses adapter/merged model if enabled; otherwise fallback.
    """
    # Make sure RAG is loaded (so /debug/ha_status reflects real count)
    _load_rag_db()

    if not (USE_HA_ADAPTER or USE_HA_MODEL):
        return _fallback_ha(requirements)

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


# Load RAG at import so /debug/ha_status shows real count even before first call
try:
    _load_rag_db()
except Exception:
    pass
