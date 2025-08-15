# app/models/ha_infer.py
from __future__ import annotations

import os, re, json, time
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np

# =========================
# Adapter-only config
# =========================
USE_HA_ADAPTER   = os.getenv("USE_HA_ADAPTER", "1") == "1"
HA_ADAPTER_REPO  = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")
BASE_MODEL_ID    = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

# Generation knobs
HA_MAX_NEW_TOKENS   = int(os.getenv("HA_MAX_NEW_TOKENS", "256"))
DO_SAMPLE           = os.getenv("DO_SAMPLE", os.getenv("do_sample", "0")) == "1"
NUM_BEAMS           = int(os.getenv("NUM_BEAMS", "1"))
TEMPERATURE         = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P               = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY  = float(os.getenv("REPETITION_PENALTY", "1.05"))

# HF cache/auth
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
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
# Optional embeddings (MiniLM)
# =========================
_HAS_EMB = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False
_emb = None

# =========================
# Globals
# =========================
_tokenizer = None  # type: ignore
_model = None      # type: ignore

# =========================
# Synthetic RAG (used ONLY for enums/backfill)
# =========================
RAG_SYNTHETIC_PATH = os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl")
_RAG_DB: List[Dict[str, Any]] = []
_RAG_TEXTS: List[str] = []

# =========================
# MAUDE fetch
# =========================
MAUDE_FETCH     = os.getenv("MAUDE_FETCH", "1") == "1"      # <-- default ON per your request
MAUDE_DEVICE    = os.getenv("MAUDE_DEVICE", "SIGMA SPECTRUM")
MAUDE_LIMIT     = int(os.getenv("MAUDE_LIMIT", "30"))
MAUDE_TTL       = int(os.getenv("MAUDE_TTL", "86400"))
MAUDE_CACHE_DIR = os.getenv("MAUDE_CACHE_DIR", "/tmp/maude_cache")

# =========================
# Anti-copy (from RAG) controls
# =========================
PARAPHRASE_FROM_RAG   = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
SIM_THRESHOLD         = float(os.getenv("SIM_THRESHOLD", "0.9"))  # a bit stricter
PARAPHRASE_MAX_WORDS  = int(os.getenv("PARAPHRASE_MAX_WORDS", "18"))

# =========================
# Risks per requirement
# =========================
_DEFAULT_RISKS = [
    "Air Embolism","Allergic response","Infection","Overdose","Underdose",
    "Delay of therapy","Environmental Hazard","Incorrect Therapy","Trauma","Particulate",
]

# =========================
# Loaders
# =========================
def _load_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False, **_token_cache_kwargs())
        try:
            tok.legacy = True  # type: ignore[attr-defined]
        except Exception:
            pass
        _tokenizer = tok
    except Exception:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, **_token_cache_kwargs())
    if getattr(_tokenizer, "pad_token", None) is None:
        _tokenizer.pad_token = _tokenizer.eos_token  # type: ignore[attr-defined]
    return _tokenizer

def _init_embeddings():
    global _emb
    if not _HAS_EMB:
        _emb = None
        return
    if _emb is None:
        try:
            _emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=CACHE_DIR)  # type: ignore
        except Exception:
            _emb = None

def _load_model():
    global _model
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if (torch.cuda.is_available() and not FORCE_CPU) else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        **_token_cache_kwargs(),
    )
    _model = PeftModel.from_pretrained(base, HA_ADAPTER_REPO, **_token_cache_kwargs())
    _init_embeddings()
    _load_rag_db()

# =========================
# RAG loader (canonicalize; keep for enums only)
# =========================
_CANON_KEYS = {
    "risk id": "Risk ID",
    "risk to health": "Risk to Health",
    "hazard": "Hazard",
    "hazardous situation": "Hazardous Situation",
    "hazardous situation ": "Hazardous Situation",
    "harm": "Harm",
    "sequence of events": "Sequence of Events",
    "severity of harm": "Severity of Harm",
    "p0": "P0",
    "p1": "P1",
    "poh": "PoH",
    "risk index": "Risk Index",
    "risk control": "Risk Control",
}
def _canon_record_keys(rec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        ck = _CANON_KEYS.get(str(k).strip().lower(), k)
        out[ck] = v
    for k in list(out.keys()):
        if isinstance(out[k], str):
            out[k] = out[k].strip()
    sev = out.get("Severity of Harm")
    if isinstance(sev, (int, float)) or (isinstance(sev, str) and sev.isdigit()):
        out["Severity of Harm"] = f"S{max(1, min(4, int(sev)))}"
    return out

def _load_rag_db():
    global _RAG_DB, _RAG_TEXTS
    if _RAG_DB:
        return
    try:
        if os.path.exists(RAG_SYNTHETIC_PATH):
            with open(RAG_SYNTHETIC_PATH, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        try:
                            import json5  # optional
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
            print({"ha_rag": "loaded", "path": RAG_SYNTHETIC_PATH, "rows": len(_RAG_DB)})
        else:
            print({"ha_rag": "missing", "path": RAG_SYNTHETIC_PATH})
    except Exception as e:
        print({"ha_rag_error": str(e), "path": RAG_SYNTHETIC_PATH})

# =========================
# JSON extraction / cleanup
# =========================
def _balanced_json_block(text: str) -> Optional[str]:
    if not text: return None
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i+1]
    return None

def _repair_json_str(js: str) -> Optional[Dict[str, Any]]:
    s = js.replace("“", '"').replace("”", '"').replace("’", "'")
    s = s.replace("\\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r'(?<!\\)\'', '"', s)
    try:
        return json.loads(s)
    except Exception:
        try:
            import json5
            return json5.loads(s)
        except Exception:
            return None

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    js = _balanced_json_block(text)
    return _repair_json_str(js) if js else None

# =========================
# Enums & scoring
# =========================
_SEV_NUM = {"S1":1,"S2":2,"S3":3,"S4":4}
_P_CANON = {
    "VERY LOW":"Very Low","VL":"Very Low","0":"Very Low",
    "LOW":"Low","L":"Low","1":"Low",
    "MEDIUM":"Medium","MID":"Medium","M":"Medium","2":"Medium",
    "HIGH":"High","H":"High","3":"High",
    "VERY HIGH":"Very High","VH":"Very High","4":"Very High",
}
def _canon_sev(v: Any, default="S3") -> str:
    s = str(v or "").strip().upper()
    if s.isdigit(): return f"S{max(1,min(4,int(s)))}"
    if s in _SEV_NUM: return s
    return default
def _canon_p(v: Any, default="Medium") -> str:
    return _P_CANON.get((str(v or "")).strip().upper(), default)

def _calculate_risk_fields(parsed: Dict[str, Any]) -> Tuple[int,str,str,str,str]:
    sev_c = _canon_sev(parsed.get("Severity of Harm","S3"))
    s_num = _SEV_NUM.get(sev_c, 3)
    p0 = _canon_p(parsed.get("P0","Medium"))
    p1 = _canon_p(parsed.get("P1","Medium"))
    poh_matrix = {
        ("Very Low","Very Low"):"Very Low", ("Very Low","Low"):"Very Low", ("Very Low","Medium"):"Low",
        ("Low","Very Low"):"Very Low", ("Low","Low"):"Low", ("Low","Medium"):"Medium",
        ("Medium","Very Low"):"Low", ("Medium","Low"):"Medium", ("Medium","Medium"):"Medium",
        ("Medium","High"):"High", ("High","Medium"):"High", ("High","High"):"High",
        ("Very High","High"):"Very High", ("Very High","Very High"):"Very High",
    }
    poh = poh_matrix.get((p0,p1), "Medium")
    if s_num >= 4 and poh in ("High","Very High"): ri = "Extreme"
    elif s_num >= 3 and poh in ("High","Very High"): ri = "High"
    else: ri = "Medium"
    return s_num, p0, p1, poh, ri

# =========================
# LLM generation
# =========================
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
- Fill every field; avoid 'TBD' or blanks.
- <= 20 words per field.
- Use S1..S4 for Severity of Harm.
- Output JSON only with the exact keys above; no extra text.

Context:
Risk to Health: {risk}
"""

def _generate_text(prompt: str) -> str:
    import torch
    tok = _load_tokenizer()
    _load_model()
    inputs = tok(prompt, return_tensors="pt")
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=HA_MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
    )
    if DO_SAMPLE:
        gen_kwargs.update(dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, num_beams=1))
    else:
        gen_kwargs.update(dict(do_sample=False, num_beams=NUM_BEAMS))
    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kwargs)  # type: ignore
    return tok.decode(out[0], skip_special_tokens=True)

def _gen_json_for_risk(risk: str) -> Dict[str, Any]:
    if not USE_HA_ADAPTER:
        return {}
    decoded = _generate_text(_PROMPT.format(risk=risk))
    js = _extract_json(decoded) or {}
    # retry if blank core fields
    needs = [k for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events"] if not str(js.get(k,"")).strip()]
    if needs:
        decoded = _generate_text(_PROMPT.format(risk=risk) + "\nDo not leave any field blank.")
        js2 = _extract_json(decoded) or {}
        for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events","Severity of Harm","P0","P1"]:
            if not str(js.get(k,"")).strip() and str(js2.get(k,"")).strip():
                js[k] = js2[k]
    return js

# =========================
# Anti-copy helpers
# =========================
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(a @ b / denom)

def _llm_paraphrase(text: str) -> str:
    try:
        prompt = f"""Rephrase the following into one concise medical phrase (<= {PARAPHRASE_MAX_WORDS} words).
Keep meaning and terminology accurate. Output only the phrase.

Text:
{text}"""
        decoded = _generate_text(prompt)
        line = decoded.strip().splitlines()[-1].strip().strip('"').strip()
        return line if len(line) >= 3 else text
    except Exception:
        return text

def _maybe_paraphrase_if_similar(source: str, candidate: str) -> str:
    if not PARAPHRASE_FROM_RAG:
        return candidate
    s = (source or "").strip()
    c = (candidate or "").strip()
    if not s or not c:
        return c
    if _HAS_EMB and _emb is not None:
        try:
            v1 = _emb.encode([s], convert_to_numpy=True)[0]  # type: ignore
            v2 = _emb.encode([c], convert_to_numpy=True)[0]  # type: ignore
            if _cosine_sim(v1, v2) >= SIM_THRESHOLD:
                return _llm_paraphrase(c)
            return c
        except Exception:
            pass
    # token overlap fallback
    s_tok = set(re.findall(r"[A-Za-z]+", s.lower()))
    c_tok = set(re.findall(r"[A-Za-z]+", c.lower()))
    overlap = len(s_tok & c_tok) / (len(c_tok) + 1e-9)
    return _llm_paraphrase(c) if overlap > 0.85 else c

# =========================
# RAG helpers (only for enums/backfill)
# =========================
def _lookup_rag_record(risk: str) -> Optional[Dict[str, Any]]:
    if not _RAG_DB:
        return None
    for r in _RAG_DB:
        if str(r.get("Risk to Health","")).strip().lower() == risk.strip().lower():
            return r
    return None

# =========================
# MAUDE fetch + synthesis (preferred for wording)
# =========================
def _cache_path(device_brand: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", device_brand.strip()) or "device"
    os.makedirs(MAUDE_CACHE_DIR, exist_ok=True)
    return os.path.join(MAUDE_CACHE_DIR, f"{safe}.json")

def _load_cached(device_brand: str, ttl_seconds: int) -> Optional[List[Dict[str, Any]]]:
    path = _cache_path(device_brand)
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _save_cached(device_brand: str, rows: List[Dict[str, Any]]):
    try:
        with open(_cache_path(device_brand), "w", encoding="utf-8") as f:
            json.dump(rows, f)
    except Exception:
        pass

def _fetch_maude_events(device_brand: str, limit: int) -> List[Dict[str, Any]]:
    url = "https://api.fda.gov/device/event.json"
    params = {
        "search": f'device.brand_name:"{device_brand}" OR device.generic_name:"{device_brand}"',
        "limit": max(1, min(100, int(limit))),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("results", [])

def _synthesize_from_maude(events: List[Dict[str, Any]], risk: str) -> Dict[str, Any]:
    """
    Summarize multiple MAUDE events into 4 fields via the adapter.
    This produces original wording grounded in MAUDE text.
    """
    if not events:
        return {}
    # Build a compact bullet list to keep prompt size reasonable
    bullets = []
    for e in events[:10]:
        txt = e.get("event_description") or e.get("description_of_event") or ""
        if not txt: continue
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) > 400:  # clip long narratives
            txt = txt[:400] + "..."
        bullets.append(f"- {txt}")
    context = "\n".join(bullets)

    prompt = f"""Using ONLY the following adverse event narratives, produce STRICT JSON with:
{{
  "Hazard": "",
  "Hazardous Situation": "",
  "Harm": "",
  "Sequence of Events": ""
}}

Rules:
- Be specific but <= 18 words per field.
- Use clinical wording; do not copy sentences from the list.
- Output JSON only.

Risk to Health: {risk}

Event narratives:
{context}
"""
    decoded = _generate_text(prompt)
    js = _extract_json(decoded) or {}
    # tiny cleanups
    for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events"]:
        v = (js.get(k) or "").strip()
        v = re.sub(r"\s+", " ", v)
        js[k] = v
    return js

def _get_maude_rows(device_brand: str, risk_to_health: str) -> List[Dict[str, Any]]:
    cached = _load_cached(device_brand, MAUDE_TTL)
    if cached is not None:
        return cached
    try:
        evs = _fetch_maude_events(device_brand, MAUDE_LIMIT)
        _save_cached(device_brand, evs)
        return evs
    except Exception:
        return []

# =========================
# Risk control (no IDs; no refs)
# =========================
def _strip_ids_refs(text: str) -> str:
    if not text: return text
    s = re.sub(r"\(Ref:[^)]+\)\s*$", "", text).strip()
    s = re.sub(r"\bREQ-\d+\b", "", s).strip()
    return s

def _paraphrase_control(text: str) -> str:
    if not text: return text
    s = " " + text.strip() + " "
    replacements = [
        (" shall ", " must "), (" will ", " must "), (" ensure ", " make sure "),
        (" within ", " kept within "), (" accuracy ", " precision "),
        (" user ", " operator "), (" device ", " unit "),
    ]
    for a,b in replacements: s = s.replace(a,b)
    s = _strip_ids_refs(s.strip())
    if len(s) < 30:
        s = "The design must " + s[0].lower() + s[1:]
    return s

def _nearest_req_control(reqs: List[Dict[str, Any]], hint_text: str) -> str:
    if not reqs:
        return "Implement design and operational mitigations aligned with applicable standards and safe-use guidance."
    if not (_HAS_EMB and _emb is not None):
        return _paraphrase_control(reqs[0].get("Requirements","") or "")
    try:
        corpus = [str(r.get("Requirements") or "") for r in reqs]
        vecs = _emb.encode(corpus, convert_to_numpy=True)  # type: ignore
        q    = _emb.encode([hint_text or "risk control"], convert_to_numpy=True)[0]  # type: ignore
        sims = (vecs @ q) / (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(q) + 1e-9))
        top  = int(sims.argmax())
        chosen = corpus[top] if 0 <= top < len(corpus) else corpus[0]
        return _paraphrase_control(chosen)
    except Exception:
        return _paraphrase_control(reqs[0].get("Requirements","") or "")

# =========================
# Row generation
# =========================
def _gen_row_for_risk(requirements: List[Dict[str, Any]], risk: str, rid: str) -> Dict[str, Any]:
    # 1) Start with MAUDE narratives → synthesize 4 fields
    parsed: Dict[str, Any] = {}
    if MAUDE_FETCH:
        try:
            maude_evs = _get_maude_rows(MAUDE_DEVICE, risk)
            parsed = _synthesize_from_maude(maude_evs, risk) or {}
        except Exception:
            parsed = {}

    # 2) If any descriptive field still blank, use adapter prompt (no RAG)
    if not all((parsed.get(k) for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events"])):
        try:
            llm_js = _gen_json_for_risk(risk)
            for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events"]:
                if not str(parsed.get(k,"")).strip() and str(llm_js.get(k,"")).strip():
                    parsed[k] = llm_js[k]
            # pull enums from LLM if provided
            for k in ["Severity of Harm","P0","P1"]:
                if str(llm_js.get(k,"")).strip():
                    parsed[k] = llm_js[k]
        except Exception:
            pass

    # 3) Backfill ONLY enums from RAG (no wording)
    rec = _lookup_rag_record(risk) or {}
    for k in ["Severity of Harm","P0","P1","PoH","Risk Index"]:
        if not str(parsed.get(k,"")).strip() and str(rec.get(k,"")).strip():
            parsed[k] = rec[k]

    # 4) Anti-copy vs RAG (if somehow text matches a RAG sentence)
    if rec:
        for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events"]:
            src = rec.get(k, "")
            if src and parsed.get(k):
                parsed[k] = _maybe_paraphrase_if_similar(src, parsed[k])

    # 5) Compute enums and risk index if needed
    severity_num, p0, p1, poh, risk_index = _calculate_risk_fields(parsed or {})
    poh = parsed.get("PoH", poh)
    risk_index = parsed.get("Risk Index", risk_index)

    # 6) Risk control (paraphrase requirement, never include IDs)
    hint = ((parsed.get("Hazardous Situation","") or "") + " " + (parsed.get("Harm","") or "")).strip()
    control = _nearest_req_control(requirements, hint)
    control = _strip_ids_refs(control)

    # Diagnostics
    try:
        print({
            "ha_row_debug": True,
            "req_id": rid, "risk": risk,
            "maude_fetch": MAUDE_FETCH,
            "filled": {k: bool(str(parsed.get(k,"")).strip())
                       for k in ["Hazard","Hazardous Situation","Harm","Sequence of Events"]},
        })
    except Exception:
        pass

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

# =========================
# Public API
# =========================
def infer_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        _load_tokenizer()
        _load_model()
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    for r in (requirements or []):
        rid = str(r.get("Requirement ID") or "")
        for risk in _DEFAULT_RISKS:
            try:
                row = _gen_row_for_risk(requirements, risk, rid)
            except Exception:
                row = {
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
                    "risk_control": "Implement design and operational mitigations aligned with applicable standards and safe-use guidance.",
                }
            out.append(row)
    return out

# Load RAG for enums once so /debug endpoints can show rows
try:
    _load_rag_db()
except Exception:
    pass
