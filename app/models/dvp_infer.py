# app/models/dvp_infer.py
from __future__ import annotations
import os, re, json, time
from typing import List, Dict, Any, Optional, Tuple

# ================================
# Environment / feature toggles
# ================================
USE_DVP_ADAPTER  = os.getenv("USE_DVP_ADAPTER", "1") == "1"
DVP_ADAPTER_REPO = os.getenv("DVP_ADAPTER_REPO", "cheranengg/dhf-dvp-adapter")
BASE_MODEL_ID    = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)
CACHE_DIR = (
    os.getenv("HF_HOME")
    or os.getenv("TRANSFORMERS_CACHE")
    or "/tmp/hf"
)
FORCE_CPU   = os.getenv("FORCE_CPU", "0") == "1"
MAX_NEW     = int(os.getenv("DVP_MAX_NEW_TOKENS", "320"))
TEMPERATURE = float(os.getenv("DVP_TEMPERATURE", "0.3"))
TOP_P       = float(os.getenv("DVP_TOP_P", "0.9"))
DO_SAMPLE   = os.getenv("DVP_DO_SAMPLE", "1") == "1"
NUM_BEAMS   = int(os.getenv("DVP_NUM_BEAMS", "1"))
REPETITION_PENALTY = float(os.getenv("DVP_REPETITION_PENALTY", "1.05"))

# Retrieval / Serper (optional online enrichment)
ENABLE_RETRIEVAL  = os.getenv("ENABLE_DVP_RETRIEVAL", "1") == "1"
SERPER_API_KEY    = os.getenv("SERPER_API_KEY", "")
SERPER_TOP_K      = int(os.getenv("SERPER_TOP_K", "3"))

# Diagnostics
DEBUG_DVP = os.getenv("DEBUG_DVP", "0") == "1"

def _token_cache_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {"cache_dir": CACHE_DIR}
    if HF_TOKEN: kw["token"] = HF_TOKEN
    return kw

# ================================
# Optional Guardrails (soft)
# ================================
_HAS_GUARDRAILS = False
try:
    from guardrails import Guard  # type: ignore
    _HAS_GUARDRAILS = True
except Exception:
    _HAS_GUARDRAILS = False

def _guard_validate(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Lightweight schema checks; uses guardrails if present, else manual checks."""
    errs: List[str] = []
    out = {
        "Verification Method": str(obj.get("Verification Method","")).strip() or "NA",
        "Sample Size": str(obj.get("Sample Size","")).strip() or "NA",
        "Test Procedure": str(obj.get("Test Procedure","")).strip() or "TBD",
        "Acceptance Criteria": str(obj.get("Acceptance Criteria","")).strip() or "TBD",
    }
    # Manual checks (works even if guardrails not installed)
    if not re.match(r"^[A-Za-z][A-Za-z \-/()]+$", out["Verification Method"]):
        errs.append("Verification Method format invalid")
    if not out["Sample Size"].isdigit():
        errs.append("Sample Size must be a number")
    if out["Test Procedure"] in ("", "TBD", "NA"):
        errs.append("Test Procedure missing")
    if out["Acceptance Criteria"] in ("", "TBD", "NA"):
        errs.append("Acceptance Criteria missing")
    return out, errs

# ================================
# Retrieval corpus (standards)
# ================================
_TEST_SPEC_LOOKUP = {
    # IEC 60601-1 (Electrical Safety)
    "insulation": "Insulation ≥ 50 MΩ at 500 V DC (IEC 60601-1:2012+A1:2020)",
    "leakage": "Leakage current ≤ 100 µA at rated voltage (IEC 60601-1:2012+A1:2020)",
    "dielectric": "Dielectric 1500 V AC for 1 min; no breakdown (IEC 60601-1 Ed 3.2)",
    "earth": "Earth continuity ≤ 0.1 Ω at 25 A for 1 min (IEC 60601-1 Ed 3.2)",
    # 60601-1-8 alarms
    "alarm": "Audible alarm ≥ 45 dB(A) at 1 m (IEC 60601-1-8)",
    # 60601-2-24 (pumps)
    "flow": "Flow accuracy within ±5% across 0.1–999 mL/h (IEC 60601-2-24)",
    "occlusion": "Occlusion alarm triggers ≤ 30 s at 100 kPa back pressure (IEC 60601-2-24)",
    # ISO 80369-7 connectors
    "luer": "No leakage under 300 kPa for 30 s (ISO 80369-7:2021)",
    # EMC
    "esd": "±6 kV contact, ±8 kV air discharge (IEC 61000-4-2)",
    "ri": "Radiated immunity 3 V/m, 80 MHz–1 GHz (IEC 61000-4-3)",
    # Environmental
    "temp cycle": "−25°C to +70°C, 10 cycles (IEC 60068-2-14)",
    "vibration": "10–500 Hz sweep, 0.5 g, 2 h per axis (IEC 60068-2-6)",
    "drop": "Drop 1.2 m × 10; no functional damage (IEC 60601-1)",
    # Labeling
    "label": "Symbols per ISO 15223-1; legible at 30 cm",
    # Biocomp / sterilization
    "biocompatibility": "Cytotoxicity, sensitization, irritation per ISO 10993-1",
    "sterility": "SAL ≤ 10⁻⁶ (ISO 11135:2014, EO sterilization)",
    # Packaging
    "bubble": "Bubble leak test per ASTM F2096; no leakage",
    "d4169": "Drop per ASTM D4169; 1.2 m across faces"
}

# Optional retrieval stack
_HAS_RETRIEVAL = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    import faiss  # type: ignore
    _HAS_RETRIEVAL = True
except Exception:
    _HAS_RETRIEVAL = False

_emb = None
_vs = None
_corpus: List[str] = []
_cmeta: List[Dict[str, str]] = []

def _init_retriever():
    global _emb, _vs, _corpus, _cmeta
    if not (ENABLE_RETRIEVAL and _HAS_RETRIEVAL):
        return
    if _vs is not None:
        return
    # Base corpus from map
    _corpus = []
    _cmeta = []
    for k, spec in _TEST_SPEC_LOOKUP.items():
        txt = f"{k}: {spec}"
        _corpus.append(txt)
        _cmeta.append({"key": k, "spec": spec})

    # Optional: Serper snippets (only if API key present)
    if SERPER_API_KEY:
        try:
            import requests
            def serper(query: str) -> List[str]:
                url = "https://google.serper.dev/search"
                headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
                payload = {"q": query, "num": SERPER_TOP_K}
                r = requests.post(url, headers=headers, json=payload, timeout=12)
                r.raise_for_status()
                data = r.json()
                snips = [o.get("snippet","") for o in data.get("organic", []) if o.get("snippet")]
                return [re.sub(r"\s+", " ", s).strip() for s in snips if s.strip()]
            # Seed some pump-centric queries
            for q in [
                "IEC 60601-2-24 flow rate accuracy infusion pump",
                "IEC 60601-1 dielectric strength medical device",
                "ISO 80369-7 luer leakage test",
                "IEC 61000-4-2 ESD levels medical",
            ]:
                for s in serper(q):
                    _corpus.append(s)
                    _cmeta.append({"key": "web", "spec": s})
        except Exception:
            pass

    # Build FAISS
    _emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=CACHE_DIR)
    vecs = _emb.encode(_corpus, convert_to_numpy=True)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    _vs = (index, vecs)

def _retrieve_specs(query: string, k: int = 4) -> List[str]:  # type: ignore[name-defined]
    if not (ENABLE_RETRIEVAL and _HAS_RETRIEVAL and _vs and _emb):
        # heuristic fallback when retrieval is off/unavailable
        q = query.lower()
        hits = []
        for kw, spec in _TEST_SPEC_LOOKUP.items():
            if kw in q:
                hits.append(spec)
        return hits[:k] if hits else list(_TEST_SPEC_LOOKUP.values())[:k]
    index, vecs = _vs
    qv = _emb.encode([query], convert_to_numpy=True)
    import numpy as np
    D, I = index.search(qv, min(k, len(_corpus)))
    specs = []
    for i in I[0]:
        specs.append(_cmeta[i]["spec"])
    return specs

# ================================
# Verification method / sample size
# ================================
USABILITY = ["usability", "human factors", "user", "ui", "use error"]
VISUAL    = ["label", "marking", "display", "color", "symbol"]
TECH      = ["electrical", "mechanical", "flow", "pressure", "occlusion", "accuracy", "alarm", "battery", "software"]

SEVERITY_TO_N = {5: 50, 4: 40, 3: 30, 2: 20, 1: 10}

def _verification_method(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in USABILITY): return "NA"
    if any(w in t for w in VISUAL):    return "Visual Inspection"
    if any(w in t for w in TECH):      return "Physical Testing"
    return "Physical Inspection"

def _sample_size(requirement_id: str, ha: List[Dict[str, Any]]) -> str:
    sev = None
    for h in ha or []:
        rid = str(h.get("requirement_id") or h.get("Requirement ID") or "")
        if rid == requirement_id:
            s = h.get("severity_of_harm") or h.get("Severity of Harm")
            try:
                s_int = int(str(s))
                sev = s_int if (sev is None or s_int > sev) else sev
            except Exception:
                pass
    if sev is not None:
        return str(SEVERITY_TO_N.get(int(sev), 30))
    # deterministic fallback
    try:
        digit = int(requirement_id.split("-")[-1]) % 5
        return str(20 + digit * 5)
    except Exception:
        return "30"

# ================================
# Model: adapter-only (PEFT)
# ================================
_tokenizer = None
_model     = None

def _load_model():
    global _tokenizer, _model
    if _model is not None or not USE_DVP_ADAPTER:
        _init_retriever()
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **_token_cache_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _tokenizer = tok

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if (torch.cuda.is_available() and not FORCE_CPU) else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        **_token_cache_kwargs(),
    )
    _model = PeftModel.from_pretrained(base, DVP_ADAPTER_REPO, **_token_cache_kwargs())
    try:
        _model.config.pad_token_id = _tokenizer.pad_token_id  # type: ignore
    except Exception:
        pass

    _init_retriever()

# ================================
# Prompt and generation
# ================================
_PROMPT = """You are a compliance test engineer.

Write ONLY a Design Verification section (JSON) for the requirement below.

Requirement: {req}

Context (standards/snippets):
{ctx}

Return EXACTLY this JSON object and nothing else:
{{
  "Verification Method": "NA|Visual Inspection|Physical Testing|Physical Inspection",
  "Sample Size": "<integer>",
  "Test Procedure": "3-4 bullet points, each with measurable values (units/thresholds/cycles)",
  "Acceptance Criteria": "Short line with concrete spec(s)"
}}
"""

def _gen_test_block(req: str, ctx: str) -> Dict[str, str]:
    import torch
    _load_model()
    if _model is None or _tokenizer is None:
        # fallback when model not available
        bullets = [
            f"- Verify {req} at three setpoints; record measured vs setpoint (n=3)",
            "- Repeatability across 5 cycles; compute max deviation and SD",
            "- Boundary test at min/max; confirm alarms and error handling",
            "- Attach equipment IDs, calibration records and raw data",
        ]
        return {
            "Verification Method": _verification_method(req),
            "Sample Size": "30",
            "Test Procedure": "\n".join(bullets),
            "Acceptance Criteria": "Meets stated specification across all trials",
        }

    prompt = _PROMPT.format(req=req, ctx=ctx or "(no external context)")
    inputs = _tokenizer(prompt, return_tensors="pt")
    # put tensors on model device
    try:
        dev = next(_model.parameters()).device  # type: ignore
        inputs = {k: v.to(dev) for k, v in inputs.items()}
    except Exception:
        pass

    with torch.no_grad():
        out = _model.generate(  # type: ignore
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=NUM_BEAMS if not DO_SAMPLE else 1,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
        )

    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract last JSON block
    def _balanced_json(text: str) -> Optional[str]:
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

    js = _balanced_json(decoded)
    try:
        obj = json.loads(js) if js else {}
    except Exception:
        obj = {}

    # Clean up bullets & fields
    def clean_bullets(s: str) -> str:
        lines = []
        for line in (s or "").splitlines():
            ln = line.strip().lstrip("-• ").strip()
            if len(ln.split()) >= 4:
                lines.append(f"- {ln}")
        return "\n".join(lines[:4]) if lines else "TBD"

    return {
        "Verification Method": str(obj.get("Verification Method","")).strip() or _verification_method(req),
        "Sample Size": str(obj.get("Sample Size","")).strip() or "30",
        "Test Procedure": clean_bullets(obj.get("Test Procedure","")),
        "Acceptance Criteria": str(obj.get("Acceptance Criteria","")).strip() or "TBD",
    }

# ================================
# Public API
# ================================
def dvp_predict(requirements: List[Dict[str, Any]], ha_rows: List[Dict[str, Any]]):
    """
    Inputs (each requirement row):
      - "Requirement ID" (or "requirement_id")
      - "Verification ID" (or "verification_id") optional
      - "Requirements"   (text)

    Returns rows with:
      - verification_id, requirement_id, requirements
      - verification_method, sample_size, test_procedure, acceptance_criteria
    """
    _load_model()

    rows: List[Dict[str, Any]] = []
    for r in requirements or []:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "")
        vid = str(r.get("Verification ID") or r.get("verification_id") or "")
        rtxt = str(r.get("Requirements") or r.get("requirements") or "").strip()

        # Section headers pass-through
        if rtxt.lower().endswith("requirements") and not vid:
            rows.append({
                "verification_id": vid, "requirement_id": rid, "requirements": rtxt,
                "verification_method": "NA", "sample_size": "NA",
                "test_procedure": "NA", "acceptance_criteria": "NA",
            })
            continue

        # Retrieval context
        ctx_specs = _retrieve_specs(rtxt, k=4) if ENABLE_RETRIEVAL else []
        ctx = "\n".join(f"- {s}" for s in ctx_specs)

        block = _gen_test_block(rtxt, ctx)

        # Insert method + sample size first (overriding model if needed)
        method = _verification_method(rtxt)
        sample = _sample_size(rid, ha_rows or [])

        merged = {
            "Verification Method": method or block.get("Verification Method","NA"),
            "Sample Size": sample or block.get("Sample Size","30"),
            "Test Procedure": block.get("Test Procedure","TBD"),
            "Acceptance Criteria": block.get("Acceptance Criteria","TBD"),
        }

        # Heuristic acceptance criteria if model missed it but retrieval hit
        if (not merged["Acceptance Criteria"] or merged["Acceptance Criteria"] in ("TBD","NA")) and ctx_specs:
            merged["Acceptance Criteria"] = ctx_specs[0]

        # Validation / normalization
        cleaned, errors = _guard_validate(merged)

        if DEBUG_DVP:
            try:
                print({"req": rtxt[:120], "ctx_hits": len(ctx_specs), "cleaned": cleaned, "errors": errors})
            except Exception:
                pass

        rows.append({
            "verification_id": vid,
            "requirement_id": rid,
            "requirements": rtxt,
            "verification_method": cleaned["Verification Method"],
            "sample_size": cleaned["Sample Size"],
            "test_procedure": cleaned["Test Procedure"],
            "acceptance_criteria": cleaned["Acceptance Criteria"],
        })
    return rows
