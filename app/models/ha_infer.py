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
CACHE_DIR: str = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf")

# Input length cap for the prompt (prevents OOM on long requirements + context)
HA_INPUT_MAX_TOKENS: int = int(os.getenv("HA_INPUT_MAX_TOKENS", "512"))

# Paraphrase control to avoid verbatim copies from RAG
PARAPHRASE_FROM_RAG: bool = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
PARAPHRASE_MAX_WORDS: int = int(os.getenv("PARAPHRASE_MAX_WORDS", "22"))

# Cap rows defensively (backend already caps)
ROW_LIMIT: int = int(os.getenv("HA_ROW_LIMIT", "50"))

# Debug
DEBUG_HA: bool = os.getenv("DEBUG_HA", "1") == "1"
DEBUG_PEEK_CHARS: int = int(os.getenv("DEBUG_PEEK_CHARS", "320"))

# ---------------------------
# Globals
# ---------------------------
_tokenizer = None
_model = None
_RAG_DB: List[Dict[str, Any]] = []
_MAUDE_ROWS: List[Dict[str, Any]] = []
_logged_model_banner = False

# ---------------------------
# Controlled vocab
# ---------------------------
RISK_TO_HEALTH_CHOICES = [
    "Air Embolism",
    "Allergic response",
    "Delay of therapy",
    "Environmental Hazard",
    "Incorrect Therapy",
    "Infection",
    "Overdose",
    "Particulate",
    "Trauma",
    "Underdose",
]

HARM_BY_RTH = {
    "Air Embolism": [
        "Pulmonary Embolism", "Stroke", "Shortness of breath", "Severe Injury", "Death"
    ],
    "Allergic response": [
        "Allergic reaction (Systemic / Localized)", "Toxic effects", "Severe Injury"
    ],
    "Delay of therapy": [
        "Disease Progression", "Severe Injury", "Death"
    ],
    "Environmental Hazard": [
        "Toxic effects", "Chemical burns", "Severe Injury"
    ],
    "Incorrect Therapy": [
        "Hypertension", "Hypotension", "Cardiac Arrhythmia",
        "Tachycardia", "Bradycardia", "Seizure", "Organ damage"
    ],
    "Infection": [
        "Sepsis", "Cellulitis", "Severe Septic Shock"
    ],
    "Overdose": [
        "Organ Failure", "Cardiac Arrhythmia", "Toxic effects"
    ],
    "Underdose": [
        "Progression of untreated condition", "Severe Injury"
    ],
    "Particulate": [
        "Embolism", "Organ damage", "Severe Injury"
    ],
    "Trauma": [
        "Severe Injury", "Organ damage", "Bradycardia"
    ],
}

# requirement → (hazard, hazardous_situation, risk_to_health)
REQ_TO_HA_PATTERNS = [
    (["air-in-line", "air in line", "bubble", "air detection"],
     ("Air-in-line not detected", "Patient receives air", "Air Embolism")),
    (["occlusion", "blockage", "line occlusion"],
     ("Line occlusion", "Flow restricted during therapy", "Delay of therapy")),
    (["flow", "accuracy", "rate"],
     ("Inaccurate flow rate", "Incorrect volume delivered", "Incorrect Therapy")),
    (["leakage current", "patient leakage"],
     ("Electrical leakage", "Patient contacted by leakage current", "Trauma")),
    (["dielectric", "hi-pot", "hipot"],
     ("Insulation breakdown", "Breakdown under high potential", "Trauma")),
    (["insulation resistance", "insulation"],
     ("Insulation degradation", "Compromised insulation", "Trauma")),
    (["protective earth", "earth continuity"],
     ("Protective earth failure", "Accessible parts not bonded", "Trauma")),
    (["alarm"],
     ("Alarm failure", "Alarm not triggered or inaudible", "Delay of therapy")),
    (["emc", "immunity", "emission", "esd", "radiated", "conducted"],
     ("Electromagnetic interference", "Device behavior affected by EM field", "Incorrect Therapy")),
    (["ip", "ingress", "water", "drip"],
     ("Liquid ingress", "Moisture enters enclosure", "Incorrect Therapy")),
    (["drop", "shock", "impact", "vibration"],
     ("Mechanical shock/vibration", "Component/connection damage", "Trauma")),
    (["battery", "power", "shutdown"],
     ("Battery failure", "Unexpected shutdown", "Delay of therapy")),
    (["usability", "human factors", "ui", "use error", "lockout"],
     ("Use error", "User action leads to incorrect setup", "Incorrect Therapy")),
    (["label", "marking", "symbol", "udi"],
     ("Labeling error", "User misinterprets label/IFU", "Incorrect Therapy")),
    (["luer", "connector", "80369"],
     ("Misconnection", "Wrong small-bore connection", "Incorrect Therapy")),
    (["temperature rise", "clause 11", "overheating"],
     ("Overheating", "Accessible parts exceed safe temp", "Trauma")),
]

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
    return s.strip() if len(toks) <= max_words else " ".join(toks[:max_words]).strip()

def _paraphrase_sentence(s: str) -> str:
    out = s.strip()
    out = out.replace("shall", "must").replace("device", "system").replace("ensure", "make sure")
    out = out.replace("maintain", "keep").replace("patient", "the patient")
    return _maybe_truncate_words(out, PARAPHRASE_MAX_WORDS)

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
# Model loader
# ---------------------------
def _load_model():
    global _tokenizer, _model

    if _model is not None and _tokenizer is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "auto" if (torch.cuda.is_available() and not FORCE_CPU) else None
    want_cuda = device_map == "auto"
    dtype = torch.bfloat16 if (want_cuda and hasattr(torch, "bfloat16")) else (torch.float16 if want_cuda else torch.float32)

    bnb_ok = False
    bnb_cfg = None
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
    if bnb_ok and want_cuda:
        load_kwargs.update(dict(quantization_config=bnb_cfg))

    try:
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
        print(f"[ha] WARNING: GPU/4-bit load failed → CPU fallback. err={e}")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        from transformers import AutoModelForCausalLM
        _model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        try:
            _model.config.pad_token_id = _tokenizer.pad_token_id
        except Exception:
            pass
        if DEBUG_HA:
            _log_once_banner(extra="device=cpu (fallback)")
            print("[ha] model loaded on CPU (fallback).")

# ---------------------------
# RAG + MAUDE
# ---------------------------
def _load_rag_once():
    global _RAG_DB
    if _RAG_DB:
        return
    _RAG_DB = _jsonl_load(HA_RAG_PATH)
    if DEBUG_HA:
        print(f"[ha] RAG loaded: {len(_RAG_DB)} rows")

def _load_maude_once():
    global _MAUDE_ROWS
    if _MAUDE_ROWS:
        return
    _MAUDE_ROWS = _jsonl_load(MAUDE_LOCAL_PATH)
    if DEBUG_HA:
        print(f"[ha] MAUDE loaded: {len(_MAUDE_ROWS)} rows")

def _pick_rag_seed() -> Optional[str]:
    if not _RAG_DB:
        return None
    fields = [
        "Hazard","hazard","risk_to_health","Risk to Health","hazardous_situation","Hazardous Situation",
        "harm","Harm","text","event_text","event_description","device_problem_text",
    ]
    for _ in range(12):
        row = random.choice(_RAG_DB)
        for k in fields:
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
# Prompt
# ---------------------------
_SCHEMA = """
Return a compact JSON object with keys:
- "risk_id": id like "HA-####" (optional)
- "risk_to_health": phrase
- "hazard": phrase
- "hazardous_situation": phrase
- "harm": phrase (patient-impacting)
- "sequence_of_events": phrase
- "severity_of_harm": "1".."5"
- "p0": "Very Low"|"Low"|"Medium"|"High"|"Very High"
- "p1": same scale
- "poh": same scale
- "risk_index": "Low"|"Medium"|"High"|"Very High"
- "risk_control": short preventive/mitigating statement
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
If some items truly cannot be inferred, leave them blank.

{_SCHEMA}

{context}

Now output ONLY the JSON:
"""

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
        print(f"[ha] output peek: {txt[:DEBUG_PEEK_CHARS].replace(chr(10),' ')} ...")
    return _decode_json_from_text(txt)

# ---------------------------
# Inference helpers
# ---------------------------
def infer_from_requirement(req_text: str) -> Dict[str, str]:
    t = (req_text or "").lower()
    for keys, (haz, hs, rth) in REQ_TO_HA_PATTERNS:
        if any(k in t for k in keys):
            return {
                "hazard": haz,
                "hazardous_situation": hs,
                "risk_to_health": rth,
                "sequence_of_events": "Design or use condition triggers the stated hazard",
            }
    return {"hazard": "", "hazardous_situation": "", "risk_to_health": "", "sequence_of_events": ""}

def choose_harm(risk_to_health: str, hazard: str, hs: str) -> str:
    rth = risk_to_health if risk_to_health in HARM_BY_RTH else "Incorrect Therapy"
    choices = HARM_BY_RTH.get(rth, ["Severe Injury"])
    # bias selection using keywords
    txt = f"{hazard} {hs}".lower()
    if "pressure" in txt or "occlusion" in txt:
        return "Hypertension"
    if "air" in txt:
        return "Pulmonary Embolism"
    if "label" in txt or "use error" in txt:
        return "Incorrect Therapy" if "Incorrect Therapy" in RISK_TO_HEALTH_CHOICES else random.choice(choices)
    return random.choice(choices)

def suggest_control(req_text: str, hazard: str, risk_to_health: str) -> str:
    t = (req_text or "").lower()
    if any(k in t for k in ["flow", "accuracy", "rate"]):
        return "Flow calibration; verification per IEC 60601-2-24"
    if any(k in t for k in ["air-in-line", "air in line", "bubble"]):
        return "Air-in-line detector with alarm; purge procedures"
    if any(k in t for k in ["occlusion", "blockage"]):
        return "Occlusion detection with alarms; tubing inspection"
    if any(k in t for k in ["leakage current"]):
        return "Patient leakage tests per IEC 60601-1"
    if any(k in t for k in ["dielectric", "hipot", "hi-pot"]):
        return "Dielectric withstand per IEC 60601-1"
    if any(k in t for k in ["insulation resistance"]):
        return "Insulation resistance per IEC 60601-1"
    if any(k in t for k in ["protective earth", "earth continuity"]):
        return "Protective earth continuity per IEC 60601-1"
    if any(k in t for k in ["emc", "immunity", "emission", "esd"]):
        return "EMC compliance per IEC 60601-1-2 (ed.4.1)"
    if any(k in t for k in ["ip", "ingress", "drip"]):
        return "Ingress protection per IEC 60529 (e.g., IP2X/IPX2)"
    if any(k in t for k in ["drop", "shock", "impact"]):
        return "Rough handling/impact tests per IEC 60601-1"
    if any(k in t for k in ["vibration"]):
        return "Vibration test per IEC 60068-2-6"
    if any(k in t for k in ["battery", "power", "shutdown"]):
        return "Battery safety per IEC 62133-2; backup & alarms"
    if any(k in t for k in ["usability", "human factors", "lockout"]):
        return "Usability validation per IEC 62366; lockout verified"
    if any(k in t for k in ["label", "symbol", "udi"]):
        return "Labels per ISO 15223-1; UDI & warnings verified"
    if any(k in t for k in ["luer", "connector", "80369"]):
        return "Small-bore connections per ISO 80369-7"
    if any(k in t for k in ["temperature rise", "clause 11", "overheating"]):
        return "Temperature rise test per IEC 60601-1, Cl.11"
    # generic but still specific
    return "Design controls, alarms, verification, and labeling per applicable standards"

def _ensure_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    """Light normalization only; no aggressive defaulting (prevents repetition)."""
    def s(k):
        v = d.get(k)
        return str(v).strip() if v is not None and str(v).strip() else ""

    rth = s("risk_to_health")
    if rth and rth not in RISK_TO_HEALTH_CHOICES:
        rl = rth.lower()
        for choice in RISK_TO_HEALTH_CHOICES:
            if choice.split()[0].lower() in rl:
                rth = choice
                break

    out = {
        "risk_id": s("risk_id"),
        "risk_to_health": rth,
        "hazard": s("hazard"),
        "hazardous_situation": s("hazardous_situation"),
        "harm": s("harm"),
        "sequence_of_events": s("sequence_of_events"),
        "severity_of_harm": s("severity_of_harm"),
        "p0": s("p0"),
        "p1": s("p1"),
        "poh": s("poh"),
        "risk_index": s("risk_index"),
        "risk_control": s("risk_control"),
    }

    if not re.fullmatch(r"[1-5]", out["severity_of_harm"] or ""):
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
    Returns rows (we keep requirement_id so DVP/TM can join):
      - requirement_id, risk_id, risk_to_health, hazard, hazardous_situation,
        harm, sequence_of_events, severity_of_harm, p0, p1, poh, risk_index, risk_control
    """
    _load_rag_once()
    _load_maude_once()

    rows: List[Dict[str, Any]] = []
    if not isinstance(requirements, list):
        return rows

    reqs = requirements[:ROW_LIMIT]
    running_idx = 1

    for r in reqs:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "").strip() or f"REQ-{running_idx:03d}"
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

        # Sequential HA IDs
        data["risk_id"] = f"HA-{running_idx:03d}"
        running_idx += 1

        # 1) Requirement-driven inference fills blanks (prevents copy-paste defaults)
        inferred = infer_from_requirement(req_text)
        for k in ("hazard", "hazardous_situation", "risk_to_health", "sequence_of_events"):
            if not data[k]:
                data[k] = inferred[k]

        # 2) Constrain Risk to Health; fallback if still empty
        if not data["risk_to_health"]:
            data["risk_to_health"] = "Incorrect Therapy"

        # 3) Minimal generic fallback only for still-empty cells
        if not data["hazard"]:
            data["hazard"] = "Device malfunction"
        if not data["hazardous_situation"]:
            data["hazardous_situation"] = "Patient exposed to device fault"
        if not data["sequence_of_events"]:
            data["sequence_of_events"] = "Design or use issue leads to hazardous condition"

        # 4) Patient-centric harm
        data["harm"] = choose_harm(data["risk_to_health"], data["hazard"], data["hazardous_situation"])

        # 5) Specific control aligned with requirement/hazard
        data["risk_control"] = suggest_control(req_text, data["hazard"], data["risk_to_health"]) or data["risk_control"] or \
            "Design controls, alarms, verification, and labeling per applicable standards"

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
            "risk_control": data["risk_control"],
        })

    return rows
