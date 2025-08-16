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

# RAG: generic JSONL (synthetic HA or MAUDE distilled)
HA_RAG_PATH: str = os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl")

# Local MAUDE
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
CACHE_DIR: str = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"  # ✅ fixed

# Input length cap
HA_INPUT_MAX_TOKENS: int = int(os.getenv("HA_INPUT_MAX_TOKENS", "512"))

# Paraphrase RAG
PARAPHRASE_FROM_RAG: bool = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
PARAPHRASE_MAX_WORDS: int = int(os.getenv("PARAPHRASE_MAX_WORDS", "22"))

# Cap rows
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
    "Air Embolism", "Allergic response", "Delay of therapy", "Environmental Hazard",
    "Incorrect Therapy", "Infection", "Overdose", "Particulate", "Trauma", "Underdose",
]

HARM_BY_RTH = {
    "Air Embolism": ["Pulmonary Embolism", "Stroke", "Shortness of breath", "Severe Injury", "Death"],
    "Allergic response": ["Allergic reaction (Systemic / Localized)", "Toxic effects", "Severe Injury"],
    "Delay of therapy": ["Disease Progression", "Severe Injury", "Death"],
    "Environmental Hazard": ["Toxic effects", "Chemical burns", "Severe Injury"],
    "Incorrect Therapy": ["Hypertension", "Hypotension", "Cardiac Arrhythmia", "Tachycardia", "Bradycardia", "Seizure", "Organ damage"],
    "Infection": ["Sepsis", "Cellulitis", "Severe Septic Shock"],
    "Overdose": ["Organ Failure", "Cardiac Arrhythmia", "Toxic effects"],
    "Underdose": ["Progression of untreated condition", "Severe Injury"],
    "Particulate": ["Embolism", "Organ damage", "Severe Injury"],
    "Trauma": ["Severe Injury", "Organ damage", "Bradycardia"],
}

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
# Utils
# ---------------------------
def _jsonl_load(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _maybe_truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def _paraphrase_sentence(text: str) -> str:
    return re.sub(r"\b(device|system|pump)\b", "infusion system", text, flags=re.I)

def _gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _log_once_banner():
    global _logged_model_banner
    if not _logged_model_banner:
        print(f"[ha_infer] Using base={BASE_MODEL_ID}, adapter={HA_ADAPTER_REPO if USE_HA_ADAPTER else 'None'}, cache={CACHE_DIR}")
        _logged_model_banner = True

# ---------------------------
# Model loader
# ---------------------------
def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
    _tokenizer.pad_token = _tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=CACHE_DIR,
        offload_folder=OFFLOAD_DIR,
    )

    if USE_HA_ADAPTER:
        model = PeftModel.from_pretrained(model, HA_ADAPTER_REPO, cache_dir=CACHE_DIR)

    _model = model
    _log_once_banner()
    return _tokenizer, _model

# ---------------------------
# RAG / MAUDE
# ---------------------------
def _load_rag_once():
    global _RAG_DB
    if not _RAG_DB:
        _RAG_DB = _jsonl_load(HA_RAG_PATH)

def _load_maude_once():
    global _MAUDE_ROWS
    if not _MAUDE_ROWS and os.path.exists(MAUDE_LOCAL_PATH):
        _MAUDE_ROWS = _jsonl_load(MAUDE_LOCAL_PATH)

def _pick_rag_seed(requirement: str) -> Optional[Dict[str, Any]]:
    _load_rag_once()
    if not _RAG_DB:
        return None
    return random.choice(_RAG_DB)

def _maude_snippets(n: int = 3) -> List[str]:
    _load_maude_once()
    if not _MAUDE_ROWS:
        return []
    return [r.get("event_desc", "")[:200] for r in random.sample(_MAUDE_ROWS, min(n, len(_MAUDE_ROWS)))]

# ---------------------------
# Prompt + Generation
# ---------------------------
def _build_prompt(requirement: str, rag_seed: Optional[Dict[str, Any]]) -> str:
    context = ""
    if rag_seed:
        ctext = rag_seed.get("hazard", "") + " " + rag_seed.get("hazardous_situation", "")
        if PARAPHRASE_FROM_RAG:
            ctext = _paraphrase_sentence(ctext)
        context = f"\nContext: {ctext}"

    maude_ctx = ""
    if not MAUDE_LOCAL_ONLY:
        snippets = _maude_snippets(2)
        if snippets:
            maude_ctx = "\nMAUDE Evidence: " + " | ".join(snippets)

    return (
        f"### Instruction:\n"
        f"Generate a structured hazard analysis JSON for the given product requirement.\n"
        f"Requirement: {requirement}{context}{maude_ctx}\n\n"
        f"### Response:\n"
    )

def _decode_json_from_text(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {}

def _generate_json(prompt: str) -> Dict[str, Any]:
    tok, model = _load_model()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=HA_INPUT_MAX_TOKENS).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=HA_MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(output[0], skip_special_tokens=True)
    if DEBUG_HA:
        print("[DEBUG decoded]", decoded[:DEBUG_PEEK_CHARS])
    return _decode_json_from_text(decoded)

# ---------------------------
# Mapping helpers
# ---------------------------
def choose_harm(risk_to_health: str) -> str:
    harms = HARM_BY_RTH.get(risk_to_health, [])
    if harms:
        return random.choice(harms)
    return "Severe Injury"

def suggest_control(hazard: str) -> str:
    if "electrical" in hazard.lower():
        return "Design insulation barriers and perform dielectric testing"
    if "occlusion" in hazard.lower():
        return "Include pressure sensors and occlusion alarms"
    if "air" in hazard.lower():
        return "Add ultrasonic air-in-line detector"
    if "label" in hazard.lower():
        return "Perform IFU validation and labeling verification"
    return "Mitigate via design control and risk management per ISO 14971"

def _ensure_fields(obj: Dict[str, Any], requirement: str, idx: int) -> Dict[str, Any]:
    risk_id = f"HA-{idx+1:03d}"

    # Try requirement heuristics
    matched = None
    for keys, (haz, sit, rth) in REQ_TO_HA_PATTERNS:
        if any(k in requirement.lower() for k in keys):
            matched = (haz, sit, rth)
            break

    hazard = matched[0] if matched else obj.get("hazard", "Device malfunction")
    situation = matched[1] if matched else obj.get("hazardous_situation", "Patient exposed to device fault")
    risk_to_health = matched[2] if matched else obj.get("risk_to_health", random.choice(RISK_TO_HEALTH_CHOICES))

    harm = choose_harm(risk_to_health)
    seq = obj.get("sequence_of_events", "Design or use issue leads to hazardous condition")
    sev = int(obj.get("severity", 3))

    control = suggest_control(hazard)

    return {
        "risk_id": risk_id,
        "risk_to_health": risk_to_health,
        "hazard": hazard,
        "hazardous_situation": situation,
        "harm": harm,
        "sequence_of_events": seq,
        "severity": sev,
        "risk_control": control,
    }

# ---------------------------
# Public
# ---------------------------
def infer_from_requirement(requirement: str, idx: int) -> Dict[str, Any]:
    rag_seed = _pick_rag_seed(requirement)
    prompt = _build_prompt(requirement, rag_seed)
    raw = _generate_json(prompt)
    return _ensure_fields(raw, requirement, idx)

def infer_ha(requirements: List[str]) -> List[Dict[str, Any]]:
    results = []
    for idx, req in enumerate(requirements[:ROW_LIMIT]):
        res = infer_from_requirement(req, idx)
        results.append(res)
    _gc_cuda()
    return results
