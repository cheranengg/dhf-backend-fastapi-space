# app/models/ha_infer.py
from __future__ import annotations

import os, json, random, hashlib, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ----------------- ENV / CONFIG -----------------
USE_HA_ADAPTER = os.getenv("USE_HA_ADAPTER", "1") == "1"
HA_ADAPTER_REPO = os.getenv("HA_ADAPTER_REPO", "cheranengg/dhf-ha-adapter")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

HA_RAG_PATH = os.getenv("HA_RAG_PATH", "app/rag_sources/ha_synthetic.jsonl")

# Local MAUDE (fast, preferred)
USE_MAUDE_RAG = os.getenv("USE_MAUDE_RAG", "1") == "1"
MAUDE_RAG_PATH = os.getenv("MAUDE_RAG_PATH", "app/rag_sources/maude_sigma_spectrum.jsonl")
MAUDE_SAMPLE_RATE = float(os.getenv("MAUDE_SAMPLE_RATE", "0.7"))
MAUDE_SNIPPETS = int(os.getenv("MAUDE_SNIPPETS", "6"))

# generation knobs (kept mild)
HA_MAX_NEW_TOKENS = int(os.getenv("HA_MAX_NEW_TOKENS", "192"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
DO_SAMPLE = os.getenv("do_sample", "1") == "1"
TEMPERATURE = float(os.getenv("HA_TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("HA_TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("HA_REPETITION_PENALTY", "1.05"))

# quick row cap for smoke/testing
QUICK_LIMIT = int(os.getenv("QUICK_LIMIT", "0"))

# paraphrase settings (to reduce verbatim copies from synthetic JSONL)
PARAPHRASE_FROM_RAG = os.getenv("PARAPHRASE_FROM_RAG", "1") == "1"
PARAPHRASE_MAX_WORDS = int(os.getenv("PARAPHRASE_MAX_WORDS", "24"))

FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)
CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"


def _token_cache_kwargs():
    kw = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw


# ----------------- LIGHT STORAGE -----------------
_RAG_DB: List[Dict[str, Any]] = []       # synthetic HA rows
_MAUDE_DB: List[Dict[str, Any]] = []     # local MAUDE rows (narrative)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _lazy_load_rag():
    global _RAG_DB
    if not _RAG_DB and HA_RAG_PATH:
        _RAG_DB = _load_jsonl(HA_RAG_PATH)


def _lazy_load_maude_rag():
    global _MAUDE_DB
    if not _MAUDE_DB and USE_MAUDE_RAG and MAUDE_RAG_PATH:
        _MAUDE_DB = _load_jsonl(MAUDE_RAG_PATH)
        # keep only entries that have a narrative
        _MAUDE_DB = [r for r in _MAUDE_DB if r.get("narrative")]


# ----------------- MODEL (ADAPTER ONLY) -----------------
_tokenizer = None
_model = None


def _load_adapter_model():
    """Load base + LoRA adapter (no merged weights)."""
    global _tokenizer, _model
    if _model is not None or not USE_HA_ADAPTER:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device_map = "auto" if (torch.cuda.is_available() and not FORCE_CPU) else None
    dtype = torch.float16 if (torch.cuda.is_available() and not FORCE_CPU) else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **_token_cache_kwargs())
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        offload_folder=OFFLOAD_DIR if device_map == "auto" else None,
        **_token_cache_kwargs(),
    )
    _model = PeftModel.from_pretrained(
        base,
        HA_ADAPTER_REPO,
        **_token_cache_kwargs(),
    )

    try:
        _model.config.pad_token_id = _tokenizer.pad_token_id
    except Exception:
        pass
    if device_map is None:
        _model.to("cpu" if (FORCE_CPU or not torch.cuda.is_available()) else "cuda")


def _generate(text: str, max_new: int = None) -> str:
    """Small wrapper to generate text from adapter model; safe default fallbacks."""
    _load_adapter_model()
    if _model is None or _tokenizer is None:
        # adapter disabled/unavailable → return prompt tail as-is
        return "TBD"

    device = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
    max_new = max_new or HA_MAX_NEW_TOKENS

    inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
            use_cache=True,
        )
    text = _tokenizer.decode(out[0], skip_special_tokens=True)
    return text


# ----------------- HELPERS -----------------
def _hash_id(s: str) -> str:
    return "HA-" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:4_000][:4]


def _rand(seed: str) -> random.Random:
    return random.Random(int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16) % (10**8))


def _clean(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _paraphrase_fragment(text: str, max_words: int = 24) -> str:
    """Very small non-LLM paraphrase: trims and light transforms to reduce verbatim copying."""
    if not text:
        return text
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    s = " ".join(words)
    # light tweaks
    s = s.replace(" shall ", " must ")
    s = s.replace(" device ", " system ")
    s = s.replace(" patient ", " the patient ")
    return _clean(s)


def _maude_context(seed: str, want: int) -> List[str]:
    """Pick 'want' narratives from local MAUDE JSONL using a seeded sampler (no embeddings)."""
    _lazy_load_maude_rag()
    if not _MAUDE_DB:
        return []
    rnd = _rand(seed)
    pool = _MAUDE_DB
    if len(pool) <= want:
        return [r["narrative"] for r in pool]
    idxs = rnd.sample(range(len(pool)), want)
    return [_MAUDE_DB[i]["narrative"] for i in idxs]


def _synthetic_context(seed: str, want: int) -> List[str]:
    """Light synthetic snippets (just short paraphrased fields)."""
    _lazy_load_rag()
    if not _RAG_DB:
        return []
    rnd = _rand(seed)
    pool = _RAG_DB
    snippets: List[str] = []
    picks = rnd.sample(range(len(pool)), min(want, len(pool)))
    for i in picks:
        row = pool[i]
        bits = []
        for k in ("Hazard", "Hazardous situation", "Harm", "Sequence of Events", "Risk Control"):
            v = row.get(k)
            if v and isinstance(v, str):
                bits.append(_paraphrase_fragment(v, PARAPHRASE_MAX_WORDS) if PARAPHRASE_FROM_RAG else _clean(v))
        s = "; ".join([b for b in bits if b])
        if s:
            snippets.append(s)
    return snippets


PROMPT = """You are a medical device risk analyst.
Given a product requirement and supporting incident/context, fill the hazard analysis fields:
- hazard
- hazardous_situation
- harm
- sequence_of_events
Write concise phrases in plain English tailored to the requirement.

Requirement:
{req}

Context (real-world incidents & prior knowledge):
{ctx}

Respond as 4 labeled lines:
Hazard: ...
Hazardous Situation: ...
Harm: ...
Sequence of Events: ...
"""

def _infer_row(req_id: str, req_text: str) -> Dict[str, Any]:
    # Context assembly
    seed = req_id + "|" + req_text
    ctx_parts: List[str] = []

    # MAUDE: include by sample rate
    if USE_MAUDE_RAG and MAUDE_SAMPLE_RATE > 0.0 and random.random() < MAUDE_SAMPLE_RATE:
        ctx_parts.extend(_maude_context(seed, max(2, MAUDE_SNIPPETS)))

    # Synthetic fragments (lightly paraphrased)
    ctx_parts.extend(_synthetic_context(seed, 4))

    ctx_block = "\n- " + "\n- ".join(ctx_parts[:12]) if ctx_parts else "- (no additional context)"

    # Ask the model (adapter). If adapter not loaded, we'll parse as TBD below.
    out = _generate(PROMPT.format(req=req_text, ctx=ctx_block), max_new=192)

    # parse 4 labeled lines
    hz = hs = hm = seq = None
    for line in out.splitlines():
        l = line.strip()
        m = re.match(r"(?i)hazard\s*:\s*(.+)", l)
        if m and not hz:
            hz = _clean(m.group(1))
        m = re.match(r"(?i)hazardous\s*situation\s*:\s*(.+)", l)
        if m and not hs:
            hs = _clean(m.group(1))
        m = re.match(r"(?i)harm\s*:\s*(.+)", l)
        if m and not hm:
            hm = _clean(m.group(1))
        m = re.match(r"(?i)sequence\s*of\s*events\s*:\s*(.+)", l)
        if m and not seq:
            seq = _clean(m.group(1))

    # fallbacks if model returned something unusable
    def dflt(which: str) -> str:
        if which == "hazard": return "TBD"
        if which == "hazardous_situation": return "TBD"
        if which == "harm": return "TBD"
        if which == "sequence_of_events": return "TBD"
        return "TBD"

    risk_to_health = "Air Embolism"  # keep a consistent target risk for the demo
    risk_id = _hash_id(req_id + risk_to_health)

    row = {
        "requirement_id": req_id,
        "risk_id": risk_id,
        "risk_to_health": risk_to_health,
        "hazard": hz or dflt("hazard"),
        "hazardous_situation": hs or dflt("hazardous_situation"),
        "harm": hm or dflt("harm"),
        "sequence_of_events": seq or dflt("sequence_of_events"),
        "severity_of_harm": "3",
        "p0": "Medium",
        "p1": "Medium",
        "poh": "Medium",
        "risk_index": "Medium",
        # high-level risk control suggestion is allowed to be generic
        "risk_control": "Refer to IEC 60601 and ISO 14971 risk controls",
    }
    return row


# ----------------- PUBLIC API -----------------
def infer_ha(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Main entry used by FastAPI."""
    _lazy_load_rag()
    _lazy_load_maude_rag()

    reqs = requirements or []
    if QUICK_LIMIT and len(reqs) > QUICK_LIMIT:
        reqs = reqs[:QUICK_LIMIT]

    rows: List[Dict[str, Any]] = []
    for r in reqs:
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "").strip() or "REQ-NA"
        rtxt = str(r.get("Requirements") or r.get("requirements") or "").strip()
        if not rtxt:
            continue
        rows.append(_infer_row(rid, rtxt))
    return rows
