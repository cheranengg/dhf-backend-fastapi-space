# app/models/ha_infer.py
from __future__ import annotations

import os
import re
import gc
import json
import random
from typing import List, Dict, Any, Optional

import torch

# ---------------------------
# Environment / toggles
# ---------------------------
USE_HA_ADAPTER: bool = os.getenv("USE_HA_ADAPTER", "1") == "1"
BASE_MODEL_ID: str = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HA_ADAPTER_REPO: str = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")

# RAG: generic JSONL (can be a synthetic HA file or a distilled MAUDE-like file)
HA_RAG_PATH: str = os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl")

# Local MAUDE (fast, no network)
MAUDE_LOCAL_PATH: str = os.getenv("MAUDE_LOCAL_JSONL", "app/rag_sources/maude_sigma_spectrum.jsonl")
MAUDE_LOCAL_ONLY: bool = os.getenv("MAUDE_LOCAL_ONLY", "1") == "1"
MAUDE_FRACTION: float = float(os.getenv("MAUDE_FRACTION", "0.70"))

# Generation controls
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

# Input length cap for the prompt (prevents OOM on long requirements + context)
HA_INPUT_MAX_TOKENS: int = int(os.getenv("HA_INPUT_MAX_TOKENS", "512"))

# Optional paraphrase control to avoid verbatim copies from RAG
PARAPHRASE_FROM_RAG: bool = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
PARAPHRASE_MAX_WORDS: int = int(os.getenv("PARAPHRASE_MAX_WORDS", "22"))

# Cap rows defensively (backend already caps)
ROW_LIMIT: int = int(os.getenv("HA_ROW_LIMIT", "50"))

# Debug logging controls
DEBUG_HA: bool = os.getenv("DEBUG_HA", "1") == "1"
DEBUG_PEEK_CHARS: int = int(os.getenv("DEBUG_PEEK_CHARS", "320"))

# ---------------------------
# Globals
# ---------------------------
_tokenizer = None
_model = None
_RAG_DB: List[Dict[str, Any]] = []      # generic RAG rows
_MAUDE_ROWS: List[Dict[str, Any]] = []  # parsed local MAUDE jsonl (fast)
_logged_model_banner = False

# =====================================================
# NEW: fixed infusion-pump Risk to Health choices + harms
# =====================================================
RISK_TO_HEALTH_CHOICES = [
    "Air Embolism", "Allergic response", "Delay of therapy", "Environmental Hazard",
    "Incorrect Therapy", "Infection", "Overdose", "Particulate", "Trauma", "Underdose"
]

HARM_CATALOG: Dict[str, List[str]] = {
    "Air Embolism": ["Pulmonary Embolism", "Shortness of breath", "Stroke", "Death"],
    "Allergic response": ["Allergic reaction (Systemic / Localized)", "Severe Injury"],
    "Delay of therapy": ["Disease Progression", "Severe Injury"],
    "Environmental Hazard": ["Chemical burns", "Toxic effects"],
    "Incorrect Therapy": [
        "Hypertension", "Hypotension", "Cardiac Arrhythmia", "Seizure", "Organ Failure"
    ],
    "Infection": ["Sepsis", "Severe Septic Shock", "Cellulitis"],
    "Overdose": ["Toxic effects", "Organ damage", "Seizure", "Hypertension", "Cardiac Arrhythmia"],
    "Particulate": ["Pulmonary Embolism", "Stroke"],
    "Trauma": ["Severe Injury", "Organ damage"],
    "Underdose": ["Disease Progression", "Organ Failure"],
}

def choose_harm(risk_to_health: str, hazard: str, hs: str) -> str:
    """Pick a patient-centric harm from the catalog using simple cues."""
    r = (risk_to_health or "").strip()
    hazard_l = (hazard or "").lower()
    hs_l = (hs or "").lower()
    options = HARM_CATALOG.get(r) or ["Severe Injury"]

    # bias selection using hazard / HS keywords
    def pick(pref: List[str]) -> Optional[str]:
        for p in pref:
            if p in options:
                return p
        return None

    if "air" in hazard_l or "air-in-line" in hazard_l or "bubble" in hs_l:
        return pick(["Pulmonary Embolism", "Shortness of breath"]) or options[0]
    if "occlusion" in hazard_l or "restriction" in hs_l:
        return pick(["Therapy interruption or incorrect dose"]) or options[0]
    if any(k in hazard_l for k in ["flow", "dose", "accuracy"]) or "incorrect volume" in hs_l:
        return pick(["Hypertension", "Hypotension", "Seizure"]) or options[0]
    if any(k in hazard_l for k in ["shock", "drop", "impact", "vibration"]):
        return pick(["Severe Injury"]) or options[0]
    if any(k in hazard_l for k in ["infection", "contamination"]):
        return pick(["Sepsis", "Cellulitis"]) or options[0]
    if "particulate" in hazard_l:
        return pick(["Pulmonary Embolism", "Stroke"]) or options[0]
    return options[0]

# =====================================================
# NEW: specific risk-control suggestions (with standards)
# =====================================================
def _pp(s: str) -> str:
    """tiny paraphraser to avoid repeating exact strings."""
    return (s.replace("ensure", "make sure")
             .replace("shall", "must")
             .replace("verify", "confirm")
             .replace("verification", "confirmation"))

def suggest_control(req_text: str, hazard: str, rth: str) -> str:
    t = f"{req_text} {hazard}".lower()

    # standards / themes
    if any(k in t for k in ["flow", "accuracy"]):
        return _pp("Flow calibration; confirmation per IEC 60601-2-24; closed-loop checks and drift limits")
    if "occlusion" in t:
        return _pp("Occlusion detection with alarm; trip thresholds verified per IEC 60601-2-24")
    if any(k in t for k in ["air-in-line", "air in line", "bubble"]):
        return _pp("Air-in-line detector with alarm; purge / priming procedure in IFU")
    if any(k in t for k in ["leakage", "patient leakage"]):
        return _pp("Patient leakage current test per IEC 60601-1; periodic production checks")
    if any(k in t for k in ["dielectric", "hi-pot", "hipot"]):
        return _pp("Dielectric strength test per IEC 60601-1; insulation design review")
    if "insulation resistance" in t or "insulation" in t:
        return _pp("Insulation resistance ≥50 MΩ at 500 V DC per IEC 60601-1")
    if "earth" in t or "protective earth" in t:
        return _pp("Protective earth continuity ≤0.1 Ω at 25 A per IEC 60601-1")
    if "alarm" in t:
        return _pp("Alarm design and audibility per IEC 60601-1-8; tone mapping and response-time verification")
    if any(k in t for k in ["emc", "emission", "immunity", "esd", "radiated"]):
        return _pp("EMC compliance per IEC 60601-1-2 (emissions + immunity test plan and confirmation)")
    if "ip" in t or "ingress" in t:
        return _pp("Ingress protection per IEC 60529 (targeted IP rating) with orientation tests")
    if any(k in t for k in ["drop", "shock", "impact", "vibration"]):
        return _pp("Rough-handling/mechanical tests per IEC 60068 and IEC 60601-1; post-test functional check")
    if "battery" in t:
        return _pp("Battery safety per IEC 62133-2 and IEC 60601-1; charge/short-circuit protections and tests")
    if any(k in t for k in ["usability", "human factors", "use error", "ui"]):
        return _pp("Usability validation per IEC 62366-1 with representative users; mitigations in design/IFU")
    if "software" in t:
        return _pp("Software lifecycle controls per IEC 62304; unit/integration verification and traceability")
    if any(k in t for k in ["label", "marking", "symbol"]):
        return _pp("Labeling per ISO 15223-1; permanence/legibility checks and IFU clarity review")
    if any(k in t for k in ["luer", "connector"]):
        return _pp("Small-bore connector conformity per ISO 80369-7; leakage and mis-connection prevention")
    if "temperature rise" in t or "clause 11" in t:
        return _pp("Temperature rise tests per IEC 60601-1 Clause 11; monitoring of accessible parts")

    # fallback — still a bit more specific than the old generic
    return _pp("Design controls with targeted tests to the applicable IEC/ISO clause; alarms and IFU warnings")

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
    except Exception as e:
        if DEBUG_HA:
            print(f"[ha] _jsonl_load failed for {path}: {e}")
    return rows

def _maybe_truncate_words(s: str, max_words: int) -> str:
    toks = re.split(r"\s+", s.strip())
    if len(toks) <= max_words:
        return s.strip()
    return " ".join(toks[:max_words]).strip()

def _paraphrase_sentence(s: str) -> str:
    out = s.strip()
    out = out.replace("shall", "must").replace("patient", "the patient").replace("device", "system")
    out = out.replace("ensure", "make sure").replace("maintain", "keep")
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

def _log_once_banner(extra: str = ""):
    global _logged_model_banner
    if _logged_model_banner or not DEBUG_HA:
        return
    _logged_model_banner = True
    print(
        "[ha] init: "
        f"BASE_MODEL_ID={BASE_MODEL_ID} | ADAPTER={HA_ADAPTER_REPO if USE_HA_ADAPTER else 'disabled'} | "
        f"CACHE_DIR={CACHE_DIR} | OFFLOAD_DIR={OFFLOAD_DIR} | "
        f"RAG_PATH={HA_RAG_PATH} | MAUDE_LOCAL_PATH={MAUDE_LOCAL_PATH} | " + extra
    )

# ---------------------------
# Model loader (explicit 4-bit if available)
# ---------------------------
def _load_model():
    global _tokenizer, _model

    if _model is not None and _tokenizer is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "auto" if (torch.cuda.is_available() and not FORCE_CPU) else None
    want_cuda = (device_map == "auto")
    bnb_ok = False
    bnb_cfg = None
    dtype = torch.float16
    if want_cuda:
        dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    else:
        dtype = torch.float32

    try:
        from transformers import BitsAndBytesConfig
        bnb_ok = True
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    except Exception:
        bnb_ok = False

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    load_kwargs = dict(cache_dir=CACHE_DIR, low_cpu_mem_usage=True)
    if want_cuda:
        load_kwargs.update(dict(device_map="auto", torch_dtype=dtype, offload_folder=OFFLOAD_DIR))

    try:
        if bnb_ok and want_cuda:
            load_kwargs.update(dict(quantization_config=bnb_cfg))
            print("[ha] loading base in 4-bit...")
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
        elif want_cuda:
            print("[ha] bitsandbytes not available → loading half on GPU...")
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
        else:
            print("[ha] GPU not available/forced CPU → loading on CPU fp32...")
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)

        if USE_HA_ADAPTER:
            from peft import PeftModel
            _model = PeftModel.from_pretrained(base, HA_ADAPTER_REPO, cache_dir=CACHE_DIR)
        else:
            _model = base

        try:
            _model.config.pad_token_id = _tokenizer.pad_token_id
        except Exception:
            pass

        if device_map is None:
            _model.to("cuda" if want_cuda else "cpu")

        if DEBUG_HA:
            eff_device = getattr(_model, "device", "cpu")
            _log_once_banner(extra=f"device={eff_device} | bnb={bnb_ok} | dtype={dtype}")
            print(f"[ha] model loaded. device={eff_device} bnb={bnb_ok} dtype={dtype}")

    except Exception as e:
        _gc_cuda()
        print(f"[ha] WARNING: GPU/4-bit load failed → falling back to CPU fp32. err={e}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        try:
            _model.config.pad_token_id = _tokenizer.pad_token_id
        except Exception:
            pass
        if DEBUG_HA:
            _log_once_banner(extra="device=cpu (fallback)")
            print("[ha] model loaded on CPU (fallback).")

# ---------------------------
# RAG + MAUDE loaders
# ---------------------------
def _load_rag_once():
    global _RAG_DB
    if _RAG_DB:
        return
    _RAG_DB = _jsonl_load(HA_RAG_PATH)
    if DEBUG_HA:
        print(f"[ha] RAG loaded: {len(_RAG_DB)} rows from {HA_RAG_PATH}")

def _load_maude_once():
    global _MAUDE_ROWS
    if _MAUDE_ROWS:
        return
    _MAUDE_ROWS = _jsonl_load(MAUDE_LOCAL_PATH)
    if DEBUG_HA:
        print(f"[ha] MAUDE loaded: {len(_MAUDE_ROWS)} rows from {MAUDE_LOCAL_PATH}")

def _pick_rag_seed() -> Optional[str]:
    if not _RAG_DB:
        return None
    candidate_fields = [
        "Hazard", "hazard", "risk_to_health", "Risk to Health", "hazardous_situation", "Hazardous Situation",
        "harm", "Harm", "text", "event_text", "event_description", "device_problem_text",
    ]
    for _ in range(12):
        row = random.choice(_RAG_DB)
        for k in candidate_fields:
            v = row.get(k)
            if isinstance(v, str) and len(v.strip()) > 8:
                return _maybe_truncate_words(v.strip(), PARAPHRASE_MAX_WORDS + 6)
    return None

def _maude_snippets(k: int = 6) -> List[str]:
    _load_maude_once()
    if not _MAUDE_ROWS:
        return []
    texts: List[str] = []
    for r in _MAUDE_ROWS:
        for key in ("event_description", "device_problem_text", "event_text", "text", "event"):
            val = r.get(key)
            if isinstance(val, str) and len(val.strip()) > 20:
                texts.append(val.strip())
        mdr = r.get("mdr_text", [])
        if isinstance(mdr, list):
            for t in mdr:
                txt = t.get("text") if isinstance(t, dict) else None
                if isinstance(txt, str) and len(txt.strip()) > 20:
                    texts.append(txt.strip())
    if not texts:
        return []
    random.shuffle(texts)
    keep: List[str] = []
    for t in texts[: 4 * k]:
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
        rag_line = _paraphrase_sentence(rag_seed) if PARAPHRASE_FROM_RAG else rag_seed
        context += f"\nRAG seed (paraphrased): {rag_line}"
    if maude_bits:
        context += "\nMAUDE snippets:\n- " + "\n- ".join(maude_bits)
    return f"""You are a safety engineer performing Hazard Analysis for an infusion pump.

Requirement: {req_text}

Use professional language. Prefer concise phrases.
If some items truly cannot be inferred, pick reasonable, generic infusion-pump patterns.

{_SCHEMA}

{context}

Now output ONLY the JSON:"""

def _decode_json_from_text(txt: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    blob = m.group(0) if m else None
    if not blob:
        brace = txt.rfind("{")
        if brace >= 0:
            blob = txt[brace:]
    if not blob:
        return {}
    blob = re.sub(r",\s*}", "}", blob)
    blob = re.sub(r",\s*\]", "]", blob)
    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _generate_json(prompt: str) -> Dict[str, Any]:
    _load_model()
    device = _model.device  # type: ignore
    enc = _tokenizer(prompt, return_tensors="pt", truncation=True,
                     max_length=HA_INPUT_MAX_TOKENS, padding=True)  # type: ignore
    inputs = {k: v.to(device) for k, v in enc.items()}
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
    if DEBUG_HA:
        peek = txt[:DEBUG_PEEK_CHARS].replace("\n", " ")
        print(f"[ha] output peek: {peek} ...")
    return _decode_json_from_text(txt)

# ---------------------------
# Post processing / defaults
# ---------------------------
def _ensure_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    def s(k, default="TBD"):
        v = d.get(k)
        return str(v).strip() if v is not None and str(v).strip() else default

    # constrain risk_to_health to the fixed list if model wanders
    rth = s("risk_to_health", "Incorrect Therapy")
    if rth not in RISK_TO_HEALTH_CHOICES:
        # try to map by keyword
        rlow = rth.lower()
        for choice in RISK_TO_HEALTH_CHOICES:
            if choice.split()[0].lower() in rlow:
                rth = choice
                break
        else:
            rth = "Incorrect Therapy"

    out = {
        "risk_id": s("risk_id"),
        "risk_to_health": rth,
        "hazard": s("hazard", "Device malfunction"),
        "hazardous_situation": s("hazardous_situation", "Patient exposed to device fault"),
        "harm": s("harm", "Severe Injury"),
        "sequence_of_events": s("sequence_of_events", "Design or use issue leads to hazardous condition"),
        "severity_of_harm": s("severity_of_harm", "3"),
        "p0": s("p0", "Medium"),
        "p1": s("p1", "Medium"),
        "poh": s("poh", "Low"),
        "risk_index": s("risk_index", "Medium"),
        "risk_control": s("risk_control", "Design controls with tests and IFU warnings"),
    }

    # normalize severity / scales
    if not re.fullmatch(r"[1-5]", out["severity_of_harm"]):
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
    Output rows with keys:
      requirement_id, risk_id, risk_to_health, hazard, hazardous_situation, harm,
      sequence_of_events, severity_of_harm, p0, p1, poh, risk_index, risk_control
    """
    _load_rag_once()
    _load_maude_once()

    rows: List[Dict[str, Any]] = []
    if not isinstance(requirements, list):
        return rows

    reqs = requirements[:ROW_LIMIT]

    running_idx = 1  # for HA-001, HA-002, ...

    for r in reqs:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "").strip() or "REQ-XXX"
        req_text = str(r.get("Requirements") or r.get("requirements") or "").strip()

        rag_seed = _pick_rag_seed()
        maude_bits: List[str] = []
        if MAUDE_LOCAL_ONLY and random.random() < MAUDE_FRACTION:
            maude_bits = _maude_snippets(k=4)

        prompt = _build_prompt(req_text, rag_seed, maude_bits)
        try:
            raw = _generate_json(prompt)
        except Exception as e:
            if DEBUG_HA:
                print(f"[ha] generation error: {e}")
            raw = {}

        data = _ensure_fields(raw)

        # enforce sequential HA IDs
        data["risk_id"] = f"HA-{running_idx:03d}"
        running_idx += 1

        # compute patient-centric harm & better control
        data["harm"] = choose_harm(data["risk_to_health"], data["hazard"], data["hazardous_situation"])
        data["risk_control"] = suggest_control(req_text, data["hazard"], data["risk_to_health"])

        rows.append({
            "requirement_id": rid,              # keep in payload; UI can hide if desired
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
            "risk_control": data["risk_control"],
        })

    return rows
