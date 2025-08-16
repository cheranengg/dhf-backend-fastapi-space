# app/models/ha_infer.py
from __future__ import annotations
import os, re, gc, json, random
from typing import List, Dict, Any, Optional
import torch

USE_HA_ADAPTER  = os.getenv("USE_HA_ADAPTER", "1") == "1"
BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HA_ADAPTER_REPO = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")

HA_RAG_PATH        = os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl")
MAUDE_LOCAL_PATH   = os.getenv("MAUDE_LOCAL_JSONL", "app/rag_sources/maude_sigma_spectrum.jsonl")
MAUDE_LOCAL_ONLY   = os.getenv("MAUDE_LOCAL_ONLY", "1") == "1"
MAUDE_FRACTION     = float(os.getenv("MAUDE_FRACTION", "0.70"))

HA_MAX_NEW_TOKENS  = int(os.getenv("HA_MAX_NEW_TOKENS", "192"))
NUM_BEAMS          = int(os.getenv("NUM_BEAMS", "1"))
DO_SAMPLE          = os.getenv("do_sample", "1") == "1"
TOP_P              = float(os.getenv("HA_TOP_P", "0.90"))
TEMPERATURE        = float(os.getenv("HA_TEMPERATURE", "0.35"))
REPETITION_PENALTY = float(os.getenv("HA_REPETITION_PENALTY", "1.05"))

FORCE_CPU   = os.getenv("FORCE_CPU", "0") == "1"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")
CACHE_DIR   = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"

HA_INPUT_MAX_TOKENS = int(os.getenv("HA_INPUT_MAX_TOKENS", "512"))
PARAPHRASE_FROM_RAG = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
PARAPHRASE_MAX_WORDS = int(os.getenv("PARAPHRASE_MAX_WORDS", "22"))
ROW_LIMIT = int(os.getenv("HA_ROW_LIMIT", "50"))
DEBUG_HA  = os.getenv("DEBUG_HA", "1") == "1"
DEBUG_PEEK_CHARS = int(os.getenv("DEBUG_PEEK_CHARS", "320"))

# Canonical Risk-to-Health set
RISK_CANON = [
    "Air Embolism", "Allergic response", "Delay of therapy", "Environmental Hazard",
    "Incorrect Therapy", "Infection", "Overdose", "Particulate", "Trauma", "Underdose"
]

_tokenizer = None
_model = None
_RAG_DB: List[Dict[str, Any]] = []
_MAUDE_ROWS: List[Dict[str, Any]] = []
_logged_model_banner = False

def _jsonl_load(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: rows.append(json.loads(line))
                except Exception: continue
    except Exception as e:
        if DEBUG_HA: print(f"[ha] _jsonl_load failed for {path}: {e}")
    return rows

def _maybe_truncate_words(s: str, max_words: int) -> str:
    toks = re.split(r"\s+", s.strip())
    return s.strip() if len(toks) <= max_words else " ".join(toks[:max_words]).strip()

def _paraphrase_sentence(s: str) -> str:
    out = s.strip().replace("shall", "must").replace("patient", "the patient").replace("device", "system")
    out = out.replace("ensure", "make sure").replace("maintain", "keep")
    return _maybe_truncate_words(out, PARAPHRASE_MAX_WORDS)

def _gc_cuda():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception: pass

def _log_once_banner(extra: str = ""):
    global _logged_model_banner
    if _logged_model_banner or not DEBUG_HA: return
    _logged_model_banner = True
    print("[ha] init: "
          f"BASE_MODEL_ID={BASE_MODEL_ID} | ADAPTER={HA_ADAPTER_REPO if USE_HA_ADAPTER else 'disabled'} | "
          f"CACHE_DIR={CACHE_DIR} | OFFLOAD_DIR={OFFLOAD_DIR} | "
          f"RAG_PATH={HA_RAG_PATH} | MAUDE_LOCAL_PATH={MAUDE_LOCAL_PATH} | " + extra)

def _load_model():
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None: return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device_map = "auto" if (torch.cuda.is_available() and not FORCE_CPU) else None
    want_cuda = (device_map == "auto")
    dtype = torch.bfloat16 if (want_cuda and hasattr(torch, "bfloat16")) else (torch.float16 if want_cuda else torch.float32)
    bnb_ok, bnb_cfg = False, None
    try:
        from transformers import BitsAndBytesConfig
        bnb_ok = True
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    except Exception:
        bnb_ok = False
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
    if _tokenizer.pad_token is None: _tokenizer.pad_token = _tokenizer.eos_token
    load_kwargs = dict(cache_dir=CACHE_DIR, low_cpu_mem_usage=True)
    if want_cuda: load_kwargs.update(dict(device_map="auto", torch_dtype=dtype, offload_folder=OFFLOAD_DIR))
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
        try: _model.config.pad_token_id = _tokenizer.pad_token_id
        except Exception: pass
        if device_map is None:
            _model.to("cuda" if want_cuda else "cpu")
        if DEBUG_HA:
            eff_device = getattr(_model, "device", "cpu")
            _log_once_banner(extra=f"device={eff_device} | bnb={bnb_ok} | dtype={dtype}")
            print(f"[ha] model loaded. device={eff_device} bnb={bnb_ok} dtype={dtype}")
    except Exception as e:
        _gc_cuda()
        print(f"[ha] WARNING: GPU/4-bit load failed → CPU fp32. err={e}")
        _tokenizer = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR).get_input_embeddings()  # type: ignore
        from transformers import AutoTokenizer as _ATok, AutoModelForCausalLM as _AM
        _tokenizer = _ATok.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        if _tokenizer.pad_token is None: _tokenizer.pad_token = _tokenizer.eos_token
        _model = _AM.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
        try: _model.config.pad_token_id = _tokenizer.pad_token_id
        except Exception: pass
        if DEBUG_HA:
            _log_once_banner(extra="device=cpu (fallback)")
            print("[ha] model loaded on CPU (fallback).")

def _load_rag_once():
    global _RAG_DB
    if _RAG_DB: return
    _RAG_DB = _jsonl_load(HA_RAG_PATH)
    if DEBUG_HA: print(f"[ha] RAG loaded: {len(_RAG_DB)} rows from {HA_RAG_PATH}")

def _load_maude_once():
    global _MAUDE_ROWS
    if _MAUDE_ROWS: return
    _MAUDE_ROWS = _jsonl_load(MAUDE_LOCAL_PATH)
    if DEBUG_HA: print(f"[ha] MAUDE loaded: {len(_MAUDE_ROWS)} rows from {MAUDE_LOCAL_PATH}")

def _pick_rag_seed() -> Optional[str]:
    if not _RAG_DB: return None
    fields = ["Hazard","hazard","risk_to_health","Risk to Health","hazardous_situation","Hazardous Situation",
              "harm","Harm","text","event_text","event_description","device_problem_text"]
    for _ in range(12):
        row = random.choice(_RAG_DB)
        for k in fields:
            v = row.get(k)
            if isinstance(v, str) and len(v.strip()) > 8:
                return _maybe_truncate_words(v.strip(), PARAPHRASE_MAX_WORDS + 6)
    return None

def _maude_snippets(k: int = 6) -> List[str]:
    _load_maude_once()
    if not _MAUDE_ROWS: return []
    texts: List[str] = []
    for r in _MAUDE_ROWS:
        for key in ("event_description","device_problem_text","event_text","text","event"):
            val = r.get(key)
            if isinstance(val, str) and len(val.strip()) > 20: texts.append(val.strip())
        mdr = r.get("mdr_text", [])
        if isinstance(mdr, list):
            for t in mdr:
                txt = t.get("text") if isinstance(t, dict) else None
                if isinstance(txt, str) and len(txt.strip()) > 20:
                    texts.append(txt.strip())
    if not texts: return []
    random.shuffle(texts)
    keep: List[str] = []
    for t in texts[: 4 * k]:
        keep.append(_maybe_truncate_words(t, 40))
        if len(keep) >= k: break
    return keep

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

def _balanced_json_last(text: str) -> Optional[str]:
    depth, start, last = 0, -1, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    last = text[start:i+1]
    return last

def _decode_json_from_text(txt: str) -> Dict[str, Any]:
    blob = _balanced_json_last(txt)
    if not blob: return {}
    blob = re.sub(r",\s*}", "}", blob)
    blob = re.sub(r",\s*\]", "]", blob)
    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# Canonicalization
def _canon_risk_to_health(candidate: str, req_text: str) -> str:
    t = f"{candidate} {req_text}".lower()

    def is_in(words): return any(w in t for w in words)

    if is_in(["air-in-line", "air in line", "air bubble", "embol"]): return "Air Embolism"
    if is_in(["allerg", "latex", "dermis", "biocompat"]): return "Allergic response"
    if is_in(["delay", "not trigger", "not triggered", "alarm", "timeout", "boot", "ready within"]): return "Delay of therapy"
    if is_in(["spill", "leak", "environment", "hazardous substance"]): return "Environmental Hazard"
    if is_in(["dose", "incorrect", "wrong", "accuracy", "flow", "program", "software", "calibration"]): return "Incorrect Therapy"
    if is_in(["infect", "steril", "contaminat", "bio-burden", "bioburden"]): return "Infection"
    if is_in(["overdose", "too high", "over infusion", "over-infusion"]): return "Overdose"
    if is_in(["particle", "particulate", "debris", "shed"]): return "Particulate"
    if is_in(["sharp", "edge", "trauma", "impact", "drop", "shock", "vibration"]): return "Trauma"
    if is_in(["underdose", "too low", "occlusion", "under-infusion", "under infusion"]): return "Underdose"

    # map common generic fallbacks
    if "embol" in t: return "Air Embolism"
    return "Incorrect Therapy"

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

# Heuristic fallback for full row (kept from earlier, trimmed)
def _fallback_from_requirement(req: str) -> Dict[str, Any]:
    # hazard + fields; risk_to_health is canonicalized later
    t = (req or "").lower()
    if any(k in t for k in ["air-in-line","air in line","bubble","embol"]):
        return dict(hazard="Air-in-line not detected",
                    hazardous_situation="Patient receives air",
                    harm="Embolism",
                    sequence_of_events="Air bubble bypasses detector leading to infusion of air",
                    severity_of_harm="4", p0="Medium", p1="Medium", poh="Low", risk_index="High",
                    risk_control="Air-in-line detector with alarm; purge procedures")
    if "occlusion" in t:
        return dict(hazard="Line occlusion", hazardous_situation="Flow restricted during therapy",
                    harm="Therapy interruption", sequence_of_events="Blockage increases pressure; pump stops improperly",
                    severity_of_harm="3", p0="Medium", p1="Medium", poh="Low", risk_index="Medium",
                    risk_control="Occlusion detection with alarms")
    if any(k in t for k in ["flow","accuracy","dose"]):
        return dict(hazard="Inaccurate flow rate", hazardous_situation="Incorrect volume delivered",
                    harm="Dose error", sequence_of_events="Calibration drift or mechanism wear causes deviation",
                    severity_of_harm="3", p0="Medium", p1="Medium", poh="Low", risk_index="Medium",
                    risk_control="Flow calibration; verification per IEC 60601-2-24")
    if "alarm" in t or "ready within" in t:
        return dict(hazard="Alarm not triggered", hazardous_situation="User not alerted to hazard",
                    harm="Patient harm due to delay", sequence_of_events="Fault occurs; alarm logic fails to trigger",
                    severity_of_harm="3", p0="Medium", p1="Medium", poh="Low", risk_index="Medium",
                    risk_control="Alarm design per IEC 60601-1-8; response-time verification")
    if any(k in t for k in ["label","udi","marking"]):
        return dict(hazard="Mislabeling/unclear symbols", hazardous_situation="User misinterprets information",
                    harm="Use error", sequence_of_events="Symbols/text not clear/complete",
                    severity_of_harm="2", p0="Low", p1="Low", poh="Low", risk_index="Low",
                    risk_control="Labeling per ISO 15223-1; UDI compliance")
    if any(k in t for k in ["drop","shock","vibration","impact","edges"]):
        return dict(hazard="Mechanical shock/vibration", hazardous_situation="Damage after handling/transport",
                    harm="Break/Leak/Trauma", sequence_of_events="Impact exceeds structural margins",
                    severity_of_harm="3", p0="Medium", p1="Medium", poh="Low", risk_index="Medium",
                    risk_control="Robustness tests (60068/60601-1)")
    return dict(
        hazard="Device malfunction",
        hazardous_situation="Patient exposed to device fault",
        harm="Therapy interruption or incorrect dose",
        sequence_of_events="Design or use issue results in unsafe state",
        severity_of_harm="3", p0="Medium", p1="Medium", poh="Low", risk_index="Medium",
        risk_control="Design controls, alarms, verification, and labeling per standards",
    )

def _ensure_fields(d: Dict[str, Any], req_text: str) -> Dict[str, Any]:
    def s(k, default="TBD"):
        v = d.get(k)
        return str(v).strip() if v is not None and str(v).strip() else default
    out = {
        "risk_to_health": _canon_risk_to_health(s("risk_to_health", ""), req_text),
        "hazard": s("hazard", "Device malfunction"),
        "hazardous_situation": s("hazardous_situation", "Patient exposed to device fault"),
        "harm": s("harm", "Adverse physiological effects"),
        "sequence_of_events": s("sequence_of_events", "Improper setup or device issue led to patient exposure"),
        "severity_of_harm": s("severity_of_harm", "3"),
        "p0": s("p0", "Medium"), "p1": s("p1", "Medium"), "poh": s("poh", "Low"),
        "risk_index": s("risk_index", "Medium"),
        "risk_control": s("risk_control", "System operating manual provides clear purging and setup cautions/warnings"),
    }
    if not re.fullmatch(r"[1-5]", out["severity_of_harm"]): out["severity_of_harm"] = "3"
    for k in ("p0","p1","poh"):
        if out[k] not in {"Very Low","Low","Medium","High","Very High"}: out[k] = "Medium"
    if out["risk_index"] not in {"Low","Medium","High","Very High"}: out["risk_index"] = "Medium"
    return out

def infer_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _load_rag_once(); _load_maude_once()
    rows: List[Dict[str, Any]] = []
    if not isinstance(requirements, list): return rows
    reqs = requirements[:ROW_LIMIT]

    running = 1  # HA-001, HA-002, ...
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
            if DEBUG_HA: print(f"[ha] generation error: {e}")
            raw = {}

        usable = isinstance(raw, dict) and len(set(raw.keys()) &
                 {"risk_to_health","hazard","hazardous_situation","harm","sequence_of_events"}) >= 2
        if not usable:
            raw = {**_fallback_from_requirement(req_text), **(raw or {})}

        data = _ensure_fields(raw, req_text)
        risk_id = f"HA-{running:03d}"; running += 1

        # Keep requirement_id in backend object (DVP uses it); app can hide on export.
        rows.append({
            "requirement_id": rid,
            "risk_id": risk_id,
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
