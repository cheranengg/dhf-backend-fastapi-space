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
DEBUG_HA: bool = os.getenv("DEBUG_HA", "1") == "1"  # prints device/quant + short output peek
DEBUG_PEEK_CHARS: int = int(os.getenv("DEBUG_PEEK_CHARS", "320"))

# ---------------------------
# Globals
# ---------------------------
_tokenizer = None
_model = None
_RAG_DB: List[Dict[str, Any]] = []      # generic RAG rows
_MAUDE_ROWS: List[Dict[str, Any]] = []  # parsed local MAUDE jsonl (fast)
_logged_model_banner = False


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
    out = out.replace("shall", "must")
    out = out.replace("patient", "the patient")
    out = out.replace("device", "system")
    out = out.replace("ensure", "make sure")
    out = out.replace("maintain", "keep")
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
        # L4 is Ampere; bfloat16 compute is usually great for bnb 4-bit
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

    # tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    load_kwargs = dict(
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    if want_cuda:
        load_kwargs.update(dict(device_map="auto", torch_dtype=dtype, offload_folder=OFFLOAD_DIR))

    try:
        # Prefer 4-bit if bnb is present; otherwise try fp16/bf16; fallback to CPU fp32 last
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
        # Final fallback: CPU fp32
        _gc_cuda()
        print(f"[ha] WARNING: GPU/4-bit load failed → falling back to CPU fp32. err={e}")
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
    """Load a generic RAG jsonl once. Accepts multiple possible field names."""
    global _RAG_DB
    if _RAG_DB:
        return
    _RAG_DB = _jsonl_load(HA_RAG_PATH)
    if DEBUG_HA:
        print(f"[ha] RAG loaded: {len(_RAG_DB)} rows from {HA_RAG_PATH}")


def _load_maude_once():
    """Load local MAUDE jsonl once (fast)."""
    global _MAUDE_ROWS
    if _MAUDE_ROWS:
        return
    _MAUDE_ROWS = _jsonl_load(MAUDE_LOCAL_PATH)
    if DEBUG_HA:
        print(f"[ha] MAUDE loaded: {len(_MAUDE_ROWS)} rows from {MAUDE_LOCAL_PATH}")


def _pick_rag_seed() -> Optional[str]:
    """Pull a short seed text from the RAG DB using tolerant fields."""
    if not _RAG_DB:
        return None
    # Try several common keys that could exist in synthetic HA or MAUDE-like entries
    candidate_fields = [
        "Hazard", "hazard",
        "risk_to_health", "Risk to Health",
        "hazardous_situation", "Hazardous Situation",
        "harm", "Harm",
        "text", "event_text", "event_description", "device_problem_text",
    ]
    for _ in range(12):
        row = random.choice(_RAG_DB)
        for k in candidate_fields:
            v = row.get(k)
            if isinstance(v, str) and len(v.strip()) > 8:
                return _maybe_truncate_words(v.strip(), PARAPHRASE_MAX_WORDS + 6)
    return None


def _maude_snippets(k: int = 6) -> List[str]:
    """Pick k short narratives from local MAUDE rows."""
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

    prompt = f"""You are a safety engineer performing Hazard Analysis for an infusion pump.

Requirement: {req_text}

Use professional language. Prefer concise phrases.
If some items truly cannot be inferred, pick reasonable, generic infusion-pump patterns.

{_SCHEMA}

{context}

Now output ONLY the JSON:"""
    return prompt


def _decode_json_from_text(txt: str) -> Dict[str, Any]:
    """More robust JSON extraction with multiple fallbacks."""
    # 1) Strict: first {...} block
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    blob = None
    if m:
        blob = m.group(0)

    # 2) If none, try to find the last '{' and slice till end
    if not blob:
        brace = txt.rfind("{")
        if brace >= 0:
            blob = txt[brace:]

    # 3) Last fallback: try to collect key:value lines inside braces
    if not blob:
        return {}

    # Fix a few common tailing commas
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

    # Tokenize with an input cap to avoid OOM
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

    data = _decode_json_from_text(txt)
    return data


# ---------------------------
# Post processing / defaults
# ---------------------------
def _ensure_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    def s(k, default="TBD"):
        v = d.get(k)
        return str(v).strip() if v is not None and str(v).strip() else default

    out = {
        "risk_id": s("risk_id", f"HA-{random.randint(1000, 9999)}"),
        "risk_to_health": s("risk_to_health", "Air Embolism"),
        "hazard": s("hazard", "Device malfunction"),
        "hazardous_situation": s("hazardous_situation", "Patient exposed to device fault"),
        "harm": s("harm", "Adverse physiological effects"),
        "sequence_of_events": s("sequence_of_events", "Improper setup or device issue led to patient exposure"),
        "severity_of_harm": s("severity_of_harm", "3"),
        "p0": s("p0", "Medium"),
        "p1": s("p1", "Medium"),
        "poh": s("poh", "Low"),
        "risk_index": s("risk_index", "Medium"),
        "risk_control": s("risk_control", "System operating manual provides clear purging and setup cautions/warnings"),
    }
    # Normalize severity to "1".."5"
    sev = out["severity_of_harm"]
    if not re.fullmatch(r"[1-5]", sev):
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
    Output rows with keys used by the UI/export pipeline.
    """
    _load_rag_once()
    _load_maude_once()

    rows: List[Dict[str, Any]] = []
    if not isinstance(requirements, list):
        return rows

    reqs = requirements[:ROW_LIMIT]

    for r in reqs:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "").strip() or "REQ-XXX"
        req_text = str(r.get("Requirements") or r.get("requirements") or "").strip()

        # RAG seed (tolerant field search), then lightly paraphrase
        rag_seed = _pick_rag_seed()

        # MAUDE snippets for a fraction of rows
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
