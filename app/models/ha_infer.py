from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Runtime flags & locations
# =========================
USE_HA_MODEL = os.getenv("USE_HA_MODEL", "0") == "1"

# Prefer your merged repo (repo id or local path)
HA_MODEL_DIR = os.getenv("HA_MODEL_MERGED_DIR", os.getenv("HA_MODEL_DIR", "")) or "cheranengg/dhf-ha-merged"

# Optional: if you later decide to allow a base model, set this env.
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "").strip()

# HF cache & token (propagated by main.py, but we also respect direct envs)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
CACHE_DIR = (
    os.getenv("HF_HOME")
    or os.getenv("TRANSFORMERS_CACHE")
    or "/tmp/hf"
)

def _tk_kwargs():
    kw = {"cache_dir": CACHE_DIR, "use_fast": False}  # <- force slow tokenizer
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    # make failures loud so we can catch and try the next source
    kw["local_files_only"] = False
    return kw

# =========================
# Optional embeddings (best-effort)
# =========================
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# =========================
# Torch / HF imports
# =========================
if USE_HA_MODEL:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
else:
    torch = None  # type: ignore
    AutoTokenizer = AutoModelForCausalLM = None  # type: ignore

# Device selection
def _device() -> str:
    if not USE_HA_MODEL:
        return "cpu"
    try:
        import torch  # type: ignore
        if os.getenv("FORCE_CPU", "0") == "1":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

_DEVICE = _device()

# =========================
# Globals (lazy)
# =========================
_tokenizer: Optional["AutoTokenizer"] = None
_model: Optional["AutoModelForCausalLM"] = None
_emb: Optional["SentenceTransformer"] = None

_default_risks = [
    "Air Embolism","Allergic response","Infection","Overdose","Underdose",
    "Delay of therapy","Environmental Hazard","Incorrect Therapy","Trauma","Particulate",
]

_severity_map = {"negligible": 1, "minor": 2, "moderate": 3, "serious": 4, "critical": 5}

# =========================
# JSON helpers (no recursive regex!)
# =========================
def _last_json_object(text: str) -> Optional[str]:
    """
    Return the string slice of the LAST top-level {...} object.
    Works even if the model outputs prose + JSON.
    """
    if not text:
        return None
    start = -1
    depth = 0
    last: Optional[Tuple[int, int]] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    last = (start, i + 1)
    if last:
        s, e = last
        return text[s:e]
    return None

def _parse_json_from_text(txt: str) -> Optional[Dict[str, Any]]:
    raw = _last_json_object(txt)
    if not raw:
        return None
    js = raw.replace("'", '"').replace("\\n", " ")
    js = re.sub(r"\s+", " ", js)
    js = re.sub(r",\s*\}", "}", js)
    js = re.sub(r",\s*\]", "]", js)
    try:
        return json.loads(js)
    except Exception:
        return None

# =========================
# Risk calculations / guardrails
# =========================
def _calculate_risk_fields(parsed: Dict[str, Any]) -> Tuple[int, str, str, str, str]:
    sev_txt = str(parsed.get("Severity of Harm", "Moderate")).strip().lower()
    severity = _severity_map.get(sev_txt, 3)

    p0 = str(parsed.get("P0", "Medium"))
    p1 = str(parsed.get("P1", "Medium"))

    poh_matrix = {
        ("Very Low","Very Low"):"Very Low", ("Very Low","Low"):"Very Low",
        ("Very Low","Medium"):"Low", ("Low","Very Low"):"Very Low",
        ("Low","Low"):"Low", ("Low","Medium"):"Medium", ("Medium","Medium"):"Medium",
        ("Medium","High"):"High", ("High","Medium"):"High", ("High","High"):"High",
        ("Very High","High"):"Very High", ("Very High","Very High"):"Very High",
    }
    poh = poh_matrix.get((p0, p1), "Medium")

    if severity == 5 and poh in ("High", "Very High"):
        risk_index = "Extreme"
    elif severity >= 3 and poh in ("High", "Very High"):
        risk_index = "High"
    else:
        risk_index = "Medium"

    return severity, p0, p1, poh, risk_index

# =========================
# Model loading
# =========================
def _load_tokenizer():
    """
    Load tokenizer for HA model using slow SentencePiece tokenizer.
    We set legacy=True to avoid the protobuf requirement path.
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    sources = []
    HA_MODEL_DIR = os.getenv("HA_MODEL_MERGED_DIR") or os.getenv("HA_MODEL_DIR") or ""
    if HA_MODEL_DIR:
        sources.append(HA_MODEL_DIR)

    last_exc = None
    from transformers import AutoTokenizer

    tk_kwargs = {
        "use_fast": False,   # slow SP tokenizer
        "legacy": True,      # avoid new behavior that requires protobuf
    }
    # include cache + token if you already have helpers for that
    tk_kwargs.update(_token_cache_kwargs())

    for src in sources:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(src, **tk_kwargs)  # type: ignore
            return _tokenizer
        except Exception as e:
            last_exc = e

    raise RuntimeError(
        f"HA tokenizer load failed. Tried: {sources}. Last error: {last_exc}"
    )


def _load_model():
    global _model, _emb
    if not USE_HA_MODEL or _model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    import torch  # type: ignore

    tok = _load_tokenizer()
    dtype = torch.float16 if (torch.cuda.is_available() and _DEVICE == "cuda") else torch.float32

    sources: List[str] = []
    if HA_MODEL_DIR:
        sources.append(HA_MODEL_DIR)
    if BASE_MODEL_ID:
        sources.append(BASE_MODEL_ID)

    last_exc: Optional[Exception] = None
    for src in sources:
        try:
            mdl = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, **_token_cache_kwargs())  # type: ignore
            mdl.to(_DEVICE)
            _model = mdl
            break
        except Exception as e:
            last_exc = e
            _model = None
            continue

    if _model is None:
        raise RuntimeError(f"HA model load failed: {last_exc}")

    # Optional embeddings for nearest requirement hint
    if _HAS_EMB:
        try:
            _emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=CACHE_DIR)  # type: ignore
        except Exception:
            _emb = None

# =========================
# Prompt
# =========================
_PROMPT = (
    "You are creating a hazard analysis row for an infusion pump.\n"
    "Return ONLY a single JSON object with the following keys:\n"
    '{\n'
    '  "Hazard": "...",\n'
    '  "Hazardous Situation": "...",\n'
    '  "Harm": "...",\n'
    '  "Sequence of Events": "...",\n'
    '  "Severity of Harm": "Negligible|Minor|Moderate|Serious|Critical",\n'
    '  "P0": "Very Low|Low|Medium|High|Very High",\n'
    '  "P1": "Very Low|Low|Medium|High|Very High"\n'
    "}\n"
    "Risk to Health: {risk}\n"
)

# =========================
# Utility
# =========================
def _nearest_req_control(reqs: List[Dict[str, Any]], hint_text: str) -> str:
    """
    Try to pick the nearest requirement text as a risk control hint.
    Falls back to standard references if embeddings aren't available.
    """
    if not reqs:
        return "Refer to IEC 60601 / ISO 14971"
    if not _HAS_EMB or _emb is None:
        # simple heuristic: longest requirement text
        r = max(reqs, key=lambda r: len(str(r.get("Requirements") or "")))
        rid = r.get("Requirement ID") or ""
        return f"{r.get('Requirements','')} (Ref: {rid})" if r.get("Requirements") else "Refer to IEC 60601 / ISO 14971"

    try:
        corpus = [str(r.get("Requirements") or "") for r in reqs]
        ids    = [str(r.get("Requirement ID") or "") for r in reqs]
        if not corpus:
            return "Refer to IEC 60601 / ISO 14971"
        import numpy as np  # local import to avoid hard dep if not needed
        vecs = _emb.encode(corpus, convert_to_numpy=True)  # type: ignore
        index = faiss.IndexFlatL2(vecs.shape[1])  # type: ignore
        index.add(vecs)
        q = _emb.encode([hint_text or "risk control"], convert_to_numpy=True)  # type: ignore
        D, I = index.search(q, 1)  # type: ignore
        i = int(I[0][0])
        if 0 <= i < len(corpus) and corpus[i]:
            return f"{corpus[i]} (Ref: {ids[i]})"
    except Exception:
        pass
    return "Refer to IEC 60601 / ISO 14971"

def _gen_row_for_risk(reqs: List[Dict[str, Any]], risk: str) -> Dict[str, Any]:
    """
    Generate one HA row for a single Risk to Health.
    """
    if not USE_HA_MODEL:
        # Fallback stub (deterministic)
        return {
            "Hazard": "TBD",
            "Hazardous Situation": "TBD",
            "Harm": "TBD",
            "Sequence of Events": "TBD",
            "Severity of Harm": "3",
            "P0": "Medium",
            "P1": "Medium",
        }

    _load_model()
    import torch  # type: ignore

    prompt = _PROMPT.format(risk=risk)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_DEVICE)  # type: ignore

    with torch.inference_mode():  # safer than no_grad for text-gen
        out = _model.generate(
            **inputs,
            max_new_tokens=320,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            eos_token_id=_tokenizer.eos_token_id,  # type: ignore
        )  # type: ignore

    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore
    parsed = _parse_json_from_text(decoded) or {}

    # Guardrails
    severity, p0, p1, poh, risk_index = _calculate_risk_fields(parsed)
    hint = (parsed.get("Hazardous Situation", "") + " " + parsed.get("Harm", "")).strip()
    control = _nearest_req_control(reqs, hint)

    return {
        "Hazard": parsed.get("Hazard", "TBD"),
        "Hazardous Situation": parsed.get("Hazardous Situation", "TBD"),
        "Harm": parsed.get("Harm", "TBD"),
        "Sequence of Events": parsed.get("Sequence of Events", "TBD"),
        "Severity of Harm": str(severity),
        "P0": p0,
        "P1": p1,
        "PoH": poh,
        "Risk Index": risk_index,
        "Risk Control": control,
    }

# =========================
# Public API
# =========================
def ha_predict(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    requirements: list of dicts with keys
      - 'Requirement ID'
      - 'Verification ID' (optional)
      - 'Requirements' (text)

    Returns a list of rows (dicts) with keys expected by main.py normalizer.
    """
    rows: List[Dict[str, Any]] = []
    for r in requirements:
        rid = str(r.get("Requirement ID") or "")
        for risk in _default_risks:
            core = _gen_row_for_risk(requirements, risk)
            rows.append({
                "requirement_id": rid,
                "risk_id": f"HA-{abs(hash((risk, rid))) % 10000:04}",
                "risk_to_health": risk,
                "hazard": core["Hazard"],
                "hazardous_situation": core["Hazardous Situation"],
                "harm": core["Harm"],
                "sequence_of_events": core["Sequence of Events"],
                "severity_of_harm": core["Severity of Harm"],
                "p0": core["P0"],
                "p1": core["P1"],
                "poh": core["PoH"],
                "risk_index": core["Risk Index"],
                "risk_control": core["Risk Control"],
            })
    return rows
