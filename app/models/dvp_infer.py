from __future__ import annotations
import os, re, json, random
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
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")
FORCE_CPU   = os.getenv("FORCE_CPU", "0") == "1"

MAX_NEW     = int(os.getenv("DVP_MAX_NEW_TOKENS", "160"))
INPUT_MAX   = int(os.getenv("DVP_INPUT_MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("DVP_TEMPERATURE", "0.3"))
TOP_P       = float(os.getenv("DVP_TOP_P", "0.9"))
DO_SAMPLE   = os.getenv("DVP_DO_SAMPLE", "1") == "1"
NUM_BEAMS   = int(os.getenv("DVP_NUM_BEAMS", "1"))
REPETITION_PENALTY = float(os.getenv("DVP_REPETITION_PENALTY", "1.05"))

# Local RAG over standards KB
DVP_RAG_PATH     = os.getenv("DVP_RAG_PATH", "app/rag_sources/standards_kb.jsonl")
DVP_SIM_THRESHOLD = float(os.getenv("DVP_SIM_THRESHOLD", "0.62"))
DVP_TOP_K         = int(os.getenv("DVP_TOP_K", "4"))

DEBUG_DVP = os.getenv("DEBUG_DVP", "1") == "1"

def _token_cache_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {"cache_dir": CACHE_DIR}
    if HF_TOKEN: kw["token"] = HF_TOKEN
    return kw

# ================================
# Validation / guardrails-lite
# ================================
_HAS_GUARDRAILS = False
try:
    from guardrails import Guard  # type: ignore
    _HAS_GUARDRAILS = True
except Exception:
    _HAS_GUARDRAILS = False

def _guard_validate(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errs: List[str] = []
    out = {
        "Verification Method": str(obj.get("Verification Method","")).strip() or "NA",
        "Sample Size": str(obj.get("Sample Size","")).strip() or "NA",
        "Test Procedure": str(obj.get("Test Procedure","")).strip() or "TBD",
        "Acceptance Criteria": str(obj.get("Acceptance Criteria","")).strip() or "TBD",
    }
    if not re.match(r"^[A-Za-z][A-Za-z \-/()]+$", out["Verification Method"]):
        errs.append("Verification Method format invalid")
    if not out["Sample Size"].isdigit():
        errs.append("Sample Size must be a number")
    # 2–4 bullets with measurable values (rough check for numbers/units)
    bullets = [b.strip() for b in out["Test Procedure"].splitlines() if b.strip().startswith("- ")]
    if not (2 <= len(bullets) <= 4):
        errs.append("Test Procedure should be 2–4 bullets")
    unit_pat = r"(mA|µA|V|kV|ms|s|min|h|mL/h|kPa|Pa|%|dB|Ω|°C|g|cycles)"
    if not any(re.search(rf"\d+(?:\.\d+)?\s?{unit_pat}", b) for b in bullets):
        errs.append("At least one bullet must include measurable values")
    if out["Acceptance Criteria"] in ("", "TBD", "NA"):
        errs.append("Acceptance Criteria missing")
    return out, errs

# ================================
# Simple standards heuristics (fallback)
# ================================
_TEST_SPEC_MAP = {
    "insulation": "Insulation ≥ 50 MΩ at 500 V DC (IEC 60601-1:2012+A1:2020)",
    "leakage": "Leakage current ≤ 100 µA at rated voltage (IEC 60601-1:2012+A1:2020)",
    "dielectric": "Dielectric 1500 V AC for 1 min; no breakdown (IEC 60601-1 Ed 3.2)",
    "earth": "Earth continuity ≤ 0.1 Ω at 25 A for 1 min (IEC 60601-1 Ed 3.2)",
    "alarm": "Audible alarm ≥ 45 dB(A) at 1 m (IEC 60601-1-8)",
    "flow": "Flow accuracy within ±5% across 0.1–999 mL/h (IEC 60601-2-24)",
    "occlusion": "Occlusion alarm triggers ≤ 30 s at 100 kPa back pressure (IEC 60601-2-24)",
    "luer": "No leakage under 300 kPa for 30 s (ISO 80369-7:2021)",
    "esd": "±6 kV contact, ±8 kV air discharge (IEC 61000-4-2)",
    "ri": "Radiated immunity 3 V/m, 80 MHz–1 GHz (IEC 61000-4-3)",
    "temp cycle": "−25°C to +70°C, 10 cycles (IEC 60068-2-14)",
    "vibration": "10–500 Hz sweep, 0.5 g, 2 h/axis (IEC 60068-2-6)",
    "drop": "Drop 1.2 m × 10; no functional damage (IEC 60601-1)",
    "label": "Symbols per ISO 15223-1; legible at 30 cm",
    "biocompatibility": "Cytotoxicity, sensitization, irritation per ISO 10993-1",
    "sterility": "SAL ≤ 10⁻⁶ (ISO 11135:2014, EO sterilization)",
    "flammability": "Material rating V-0 (UL 94 V-0)",
    "usability": "Simulated-use ≥15 participants/user group (IEC 62366)",
    "software": "Risk-based validation traceability established (IEC 62304)",
    "interop": "HL7 FHIR / IEEE 11073 PHD conformance",
}

# ================================
# Optional local FAISS retriever
# ================================
_HAS_RETRIEVAL = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    import faiss  # type: ignore
    _HAS_RETRIEVAL = True
except Exception:
    _HAS_RETRIEVAL = False

_emb = None
_index = None
_kb_rows: List[Dict[str, Any]] = []
_kb_texts: List[str] = []

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
    except Exception as e:
        if DEBUG_DVP:
            print(f"[dvp] KB load failed for {path}: {e}")
    return rows

def _init_retriever():
    """Load KB and build FAISS once (CPU is fine)."""
    global _emb, _index, _kb_rows, _kb_texts
    if _index is not None:
        return
    _kb_rows = _jsonl_load(DVP_RAG_PATH)
    _kb_texts = []
    for r in _kb_rows:
        bits = []
        for k in ("standard","clause","topic","test_name"):
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                bits.append(v.strip())
        for b in r.get("bullets", []) or []:
            if isinstance(b, str) and b.strip():
                bits.append(b.strip())
        if bits:
            _kb_texts.append(" | ".join(bits))
    if _kb_texts and _HAS_RETRIEVAL:
        _emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=CACHE_DIR)
        vecs = _emb.encode(_kb_texts, convert_to_numpy=True)
        _index = faiss.IndexFlatL2(vecs.shape[1])
        _index.add(vecs)
        if DEBUG_DVP:
            print(f"[dvp] KB loaded: {len(_kb_rows)} rows (FAISS ready)")
    else:
        if DEBUG_DVP:
            print(f"[dvp] KB loaded: {len(_kb_rows)} rows (FAISS off or empty)")

def _kb_retrieve(req_text: str, k: int = DVP_TOP_K) -> List[Dict[str, Any]]:
    """Return top-k KB entries relevant to the requirement."""
    _init_retriever()
    if _index is None or _emb is None or not _kb_rows:
        # fallback to heuristic list turned into pseudo KB rows
        hits = []
        t = req_text.lower()
        for kw, spec in _TEST_SPEC_MAP.items():
            if kw in t:
                hits.append({"standard":"heuristic","topic":kw,"test_name":kw,
                             "bullets":[spec],"acceptance_template":spec})
        return hits[:k] if hits else [{"standard":"heuristic","topic":"generic",
                                       "test_name":"Generic Test",
                                       "bullets":[s] for s in list(_TEST_SPEC_MAP.values())[:k]}]
    qv = _emb.encode([req_text], convert_to_numpy=True)
    import numpy as np
    # quick nprobe-less flatL2 search
    # Will cap at corpus size to avoid faiss errors
    n = min(k, len(_kb_texts))
    D, I = _index.search(qv, n)
    hits: List[Dict[str, Any]] = []
    for idx in I[0]:
        if 0 <= idx < len(_kb_rows):
            hits.append(_kb_rows[idx])
    return hits

# ================================
# Verification method / sample size
# ================================
USABILITY = ["usability", "human factors", "user", "ui", "use error"]
VISUAL    = ["label", "marking", "display", "color", "symbol"]
TECH      = ["electrical", "mechanical", "flow", "pressure", "occlusion",
             "accuracy", "alarm", "battery", "software", "leakage", "current",
             "voltage", "connector", "luer", "emc", "esd", "vibration"]

# from ENV or default mapping
def _ssize_by_sev_map() -> Dict[int, int]:
    raw = os.getenv("DVP_SSIZE_BY_SEV")
    if raw:
        try:
            nums = [int(x) for x in raw.split(",")]
            if len(nums) >= 5:
                return {1:nums[0],2:nums[1],3:nums[2],4:nums[3],5:nums[4]}
        except Exception:
            pass
    return {1:10,2:20,3:30,4:40,5:50}

def _verification_method(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in USABILITY): return "Usability Validation"
    if any(w in t for w in VISUAL):    return "Visual Inspection"
    if any(w in t for w in TECH):      return "Physical Testing"
    return "Inspection"

def _sample_size(requirement_id: str, ha: List[Dict[str, Any]]) -> str:
    sev_map = _ssize_by_sev_map()
    sev = None
    for h in ha or []:
        rid = str(h.get("requirement_id") or h.get("Requirement ID") or "")
        if rid == requirement_id:
            s = h.get("severity_of_harm") or h.get("Severity of Harm")
            try:
                s_int = int(str(s))
                sev = max(sev or 0, s_int)
            except Exception:
                pass
    if sev is not None and 1 <= sev <= 5:
        return str(sev_map.get(sev, 30))
    # deterministic fallback
    try:
        digit = int(re.sub(r"\D", "", requirement_id)) % 5
        return str([10,20,30,40,50][digit])
    except Exception:
        return "30"

# ================================
# Model loader (prefers _peft_loader)
# ================================
_tokenizer = None
_model     = None
_logged    = False

def _try_peft_loader():
    """Use project-specific PEFT loader if available; return (tok, model) or (None, None)."""
    try:
        from app.models import _peft_loader as pl  # type: ignore
        for name in ("load_peft_causal","load_peft","load_lora_model"):
            fn = getattr(pl, name, None)
            if fn:
                if DEBUG_DVP: print(f"[dvp] using _peft_loader.{name}()")
                obj = fn(
                    base_model_id=BASE_MODEL_ID,
                    adapter_repo=DVP_ADAPTER_REPO,
                    cache_dir=CACHE_DIR,
                    offload_dir=OFFLOAD_DIR,
                    force_cpu=FORCE_CPU,
                    return_tokenizer=True,
                )
                tok = obj.get("tokenizer")
                mdl = obj.get("model")
                return tok, mdl
    except Exception as e:
        if DEBUG_DVP:
            print(f"[dvp] _peft_loader not used: {e}")
    return None, None

def _load_model():
    global _tokenizer, _model, _logged
    if _model is not None and _tokenizer is not None:
        _init_retriever()
        return

    tok, mdl = _try_peft_loader()
    if tok is not None and mdl is not None:
        _tokenizer, _model = tok, mdl
        _init_retriever()
        if DEBUG_DVP and not _logged:
            dev = getattr(_model, "device", "cpu")
            print(f"[dvp] model via _peft_loader on {dev}; cache={CACHE_DIR} offload={OFFLOAD_DIR}")
            _logged = True
        return

    # Fallback: local 4-bit PEFT load
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    bnb_ok = False
    dtype = torch.bfloat16 if (torch.cuda.is_available() and not FORCE_CPU and hasattr(torch,"bfloat16")) else (
            torch.float16 if (torch.cuda.is_available() and not FORCE_CPU) else torch.float32)
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        bnb_ok = True
    except Exception:
        bnb_cfg = None
        bnb_ok = False

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **_token_cache_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available() and not FORCE_CPU:
        load_kwargs.update(dict(device_map="auto", torch_dtype=dtype, offload_folder=OFFLOAD_DIR))
    if bnb_ok and not FORCE_CPU:
        load_kwargs.update(dict(quantization_config=bnb_cfg))

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)

    from peft import PeftModel
    mdl = PeftModel.from_pretrained(base, DVP_ADAPTER_REPO, **_token_cache_kwargs())
    try:
        mdl.config.pad_token_id = tok.pad_token_id  # type: ignore
    except Exception:
        pass

    _tokenizer, _model = tok, mdl
    _init_retriever()
    if DEBUG_DVP and not _logged:
        dev = getattr(_model, "device", "cpu")
        print(f"[dvp] model loaded (bnb={bnb_ok}) on {dev}; cache={CACHE_DIR} offload={OFFLOAD_DIR}")
        _logged = True

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
  "Verification Method": "Inspection|Visual Inspection|Physical Testing|Software Test|Usability Validation|Analysis",
  "Sample Size": "<integer>",
  "Test Procedure": "2-4 bullet points, each with measurable values (units/thresholds/cycles)",
  "Acceptance Criteria": "Short line restating the requirement as a measurable compliance statement"
}}
"""

def _clean_bullets(s: str) -> str:
    lines = []
    for line in (s or "").splitlines():
        ln = line.strip().lstrip("-• ").strip()
        # keep reasonably informative lines
        if len(ln.split()) >= 4:
            lines.append(f"- {ln}")
        if len(lines) == 4:
            break
    # Ensure 2–4 lines
    if len(lines) < 2:
        return "TBD"
    return "\n".join(lines[:4])

def _numbers_from_req(req: str) -> List[str]:
    unit_pat = r"(mA|µA|V|kV|ms|s|min|h|mL/h|kPa|Pa|%|dB|Ω|°C|g|cycles|cm)"
    hits = re.findall(rf"\d+(?:\.\d+)?\s?{unit_pat}", req)
    return hits

def _rewrite_acceptance(req: str, bullets: str, template: Optional[str]) -> str:
    nums = _numbers_from_req(req)
    if nums:
        # echo key numeric specs
        uniq = []
        for n in nums:
            if n not in uniq: uniq.append(n)
        joined = ", ".join(uniq[:3])
        return f"System shall comply with the stated requirement under test conditions, meeting {joined}."
    if template and isinstance(template, str) and template.strip():
        return template.strip()
    # last resort: compress the requirement
    req_short = re.sub(r"\s+", " ", req.strip())
    return f"System shall meet the requirement: {req_short}"

def _compose_ctx(req_text: str) -> Tuple[str, Optional[str]]:
    """Retrieve KB bullets and produce a compact context string + acceptance template (if any)."""
    hits = _kb_retrieve(req_text, k=DVP_TOP_K)
    bullets: List[str] = []
    tmpl = None
    for h in hits:
        for b in (h.get("bullets") or []):
            if isinstance(b, str) and b.strip():
                bullets.append(f"- {b.strip()}")
        if not tmpl and isinstance(h.get("acceptance_template"), str):
            tmpl = h.get("acceptance_template")
        if len(bullets) >= 6:
            break
    # fallback to heuristic specs if KB empty
    if not bullets:
        for spec in list(_TEST_SPEC_MAP.values())[:3]:
            bullets.append(f"- {spec}")
    return ("\n".join(bullets[:6]) if bullets else "(no context)"), tmpl

def _gen_test_block(req: str) -> Dict[str, str]:
    """Generate JSON block using LoRA model + local context, then sanitize."""
    import torch
    _load_model()
    ctx, tmpl = _compose_ctx(req)

    if _model is None or _tokenizer is None:
        # fallback when model not available
        fb = [
            "- Test at 3 setpoints; record measured vs setpoint",
            "- Repeatability across 3 cycles; compute deviation",
            "- Boundary at min/max; confirm alarms",
        ]
        return {
            "Verification Method": _verification_method(req),
            "Sample Size": "30",
            "Test Procedure": "\n".join(fb),
            "Acceptance Criteria": _rewrite_acceptance(req, "\n".join(fb), tmpl),
        }

    prompt = _PROMPT.format(req=req, ctx=ctx or "(no context)")
    enc = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=INPUT_MAX, padding=True)
    # move to model device
    try:
        dev = next(_model.parameters()).device  # type: ignore
        enc = {k: v.to(dev) for k, v in enc.items()}
    except Exception:
        pass

    with torch.no_grad():
        out = _model.generate(  # type: ignore
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=NUM_BEAMS if not DO_SAMPLE else 1,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
        )

    decoded = _tokenizer.decode(out[0], skip_special_tokens=True)

    # Extract trailing JSON block
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

    bullets = _clean_bullets(obj.get("Test Procedure",""))
    ac = obj.get("Acceptance Criteria","").strip()
    if not ac:
        ac = _rewrite_acceptance(req, bullets, tmpl)

    return {
        "Verification Method": str(obj.get("Verification Method","")).strip() or _verification_method(req),
        "Sample Size": str(obj.get("Sample Size","")).strip() or "30",
        "Test Procedure": bullets,
        "Acceptance Criteria": ac,
    }

# ================================
# Public API
# ================================
def dvp_predict(requirements: List[Dict[str, Any]], ha_rows: List[Dict[str, Any]]):
    """
    Inputs per requirement:
      - "Requirement ID" (or "requirement_id")
      - "Verification ID" (or "verification_id") optional
      - "Requirements"   (text)

    Returns rows with:
      - verification_id, requirement_id, requirements
      - verification_method, sample_size, test_procedure, acceptance_criteria
    """
    _load_model()

    rows: List[Dict[str, Any]] = []

    # precompute running IDs for rows missing a VID
    missing_idx = 1

    for r in requirements or []:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "").strip()
        vid_in = str(r.get("Verification ID") or r.get("verification_id") or "").strip()
        rtxt = str(r.get("Requirements") or r.get("requirements") or "").strip()

        # Section headers pass-through
        if rtxt.lower().endswith("requirements") and not vid_in:
            rows.append({
                "verification_id": "",
                "requirement_id": rid,
                "requirements": rtxt,
                "verification_method": "NA",
                "sample_size": "NA",
                "test_procedure": "NA",
                "acceptance_criteria": "NA",
            })
            continue

        # Generate running VID if missing
        if vid_in:
            vid = vid_in
        else:
            vid = f"DV-{missing_idx:03d}"
            missing_idx += 1

        block = _gen_test_block(rtxt)

        # Override with deterministic method & sample from HA
        method = _verification_method(rtxt)
        sample = _sample_size(rid, ha_rows or [])

        merged = {
            "Verification Method": method or block.get("Verification Method","Inspection"),
            "Sample Size": sample or block.get("Sample Size","30"),
            "Test Procedure": block.get("Test Procedure","TBD"),
            "Acceptance Criteria": block.get("Acceptance Criteria","TBD"),
        }

        cleaned, errors = _guard_validate(merged)
        if DEBUG_DVP:
            try:
                print({"rid": rid, "vid": vid, "errors": errors})
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
