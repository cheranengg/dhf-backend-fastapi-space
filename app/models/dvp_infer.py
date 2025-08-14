from __future__ import annotations
import os
from typing import List, Dict, Any, Optional

# --------- ENV / toggles ----------
USE_DVP_MODEL = os.getenv("USE_DVP_MODEL", "0") == "1"
DVP_MODEL_DIR = os.getenv("DVP_MODEL_DIR", "cheranengg/dhf-dvp-merged")  # your merged repo

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)
CACHE_DIR   = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"
FORCE_CPU   = os.getenv("FORCE_CPU", "0") == "1"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")
MAX_NEW     = int(os.getenv("DVP_MAX_NEW_TOKENS", "300"))

def _token_cache_kwargs():
    kw = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

# --------- Heuristics & lookups ----------
USABILITY_KWS = ["usability", "user", "human factors", "ui", "interface"]
VISUAL_KWS    = ["label", "marking", "display", "visual", "color"]
TECH_KWS      = ["electrical", "mechanical", "flow", "pressure", "occlusion", "accuracy", "alarm"]

TEST_SPEC_LOOKUP = {
    "insulation": "≥ 50 MΩ at 500 V DC (IEC 60601-1)",
    "leakage":    "≤ 100 µA at rated voltage (IEC 60601-1)",
    "dielectric": "1500 V AC for 1 min, no breakdown (IEC 60601-1)",
    "earth":      "Earth continuity ≤ 0.1 Ω at 25 A for 1 min (IEC 60601-1)",
    "flow":       "±5% from set value (IEC 60601-2-24)",
    "occlusion":  "Alarm ≤ 30 s at 100 kPa back pressure (IEC 60601-2-24)",
    "luer":       "No leakage under 300 kPa for 30 s (ISO 80369-7)",
    "emc":        "±6 kV contact, ±8 kV air (IEC 61000-4-2)",
    "vibration":  "10–500 Hz, 0.5 g, 2 h/axis (IEC 60068-2-6)",
    "drop":       "Drop 1.2 m × 10, no functional damage (IEC 60601-1)",
}

SEVERITY_TO_N = {5: 50, 4: 40, 3: 30, 2: 20, 1: 10}

def _get_verification_method(req_text: str) -> str:
    t = (req_text or "").lower()
    if any(k in t for k in USABILITY_KWS): return "NA"
    if any(k in t for k in VISUAL_KWS):    return "Visual Inspection"
    if any(k in t for k in TECH_KWS):      return "Physical Testing"
    return "Physical Inspection"

def _get_sample_size(requirement_id: str, ha_items: List[Dict[str, Any]]) -> str:
    sev = None
    for h in (ha_items or []):
        rid = str(h.get("requirement_id") or h.get("Requirement ID") or "")
        if rid and rid == str(requirement_id):
            s = h.get("severity_of_harm") or h.get("Severity of Harm")
            try:
                s_int = int(str(s))
                sev = s_int if (sev is None or s_int > sev) else sev
            except Exception:
                pass
    if sev is not None:
        return str(SEVERITY_TO_N.get(int(sev), 30))
    try:
        digit = int(str(requirement_id).split("-")[-1]) % 5
        return str(20 + digit * 5)
    except Exception:
        return "30"

# --------- Model (fine-tuned only) ----------
_tokenizer: Optional["AutoTokenizer"] = None
_model: Optional["AutoModelForCausalLM"] = None

def _load_tokenizer():
    """Prefer slow tokenizer (SentencePiece) to avoid protobuf/fast-tokenizer issues."""
    from transformers import AutoTokenizer
    global _tokenizer

    if _tokenizer is not None:
        return _tokenizer

    last_exc: Optional[Exception] = None

    try:
        tok = AutoTokenizer.from_pretrained(
            DVP_MODEL_DIR, use_fast=False, trust_remote_code=True, **_token_cache_kwargs()
        )
        try:
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            tok.legacy = True  # type: ignore[attr-defined]
        except Exception:
            pass
        _tokenizer = tok
        return _tokenizer
    except Exception as e:
        last_exc = e

    try:
        tok = AutoTokenizer.from_pretrained(
            DVP_MODEL_DIR, use_fast=True, trust_remote_code=True, **_token_cache_kwargs()
        )
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token  # type: ignore[attr-defined]
        _tokenizer = tok
        return _tokenizer
    except Exception as e:
        last_exc = e
        raise RuntimeError(
            f"DVP tokenizer load failed. Tried: [{DVP_MODEL_DIR}]. Last error: {last_exc}"
        )

def _load_model():
    """Load ONLY the fine-tuned DVP model; device_map=auto + optional offload; pad_token fix."""
    global _tokenizer, _model
    if not USE_DVP_MODEL or _model is not None:
        return

    from transformers import AutoModelForCausalLM
    import torch

    _load_tokenizer()
    dtype = torch.float16 if (torch.cuda.is_available() and not FORCE_CPU) else torch.float32
    device_map = "auto" if (torch.cuda.is_available() and not FORCE_CPU) else None

    _model = AutoModelForCausalLM.from_pretrained(
        DVP_MODEL_DIR,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        offload_folder=OFFLOAD_DIR if device_map == "auto" else None,
        trust_remote_code=True,
        **_token_cache_kwargs()
    )
    try:
        _model.config.pad_token_id = _tokenizer.pad_token_id  # type: ignore
    except Exception:
        pass
    if device_map is None:
        _model.to("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")

def _model_device():
    """Return the device the model actually resides on."""
    import torch
    if _model is None:
        return torch.device("cpu")
    try:
        return next(_model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")

def _clear_cuda():
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

GEN_PROMPT_TMPL = (
    "You are a compliance engineer.\n\n"
    "Generate ONLY a Design Verification Test Procedure for the requirement below.\n\n"
    "Requirement: {req}\n\n"
    "- Output exactly 3–4 concise bullets.\n"
    "- Each bullet must include measurable values (units/thresholds/cycles).\n"
    "- Stay strictly on-topic; avoid unrelated checks."
)

def _gen_test_procedure_model(requirement_text: str) -> str:
    """Generate bullets; keep inputs on the SAME device as the model."""
    _load_model()
    if _model is None:
        return _gen_test_procedure_stub(requirement_text)

    import torch, re
    dev = _model_device()
    prompt = GEN_PROMPT_TMPL.format(req=requirement_text or "the feature")

    def _run(max_new: int) -> str:
        inputs = _tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        # Only move to CUDA if the model is on CUDA; otherwise leave on CPU.
        if dev.type == "cuda":
            inputs = inputs.to(dev)  # type: ignore

        with torch.no_grad():
            out = _model.generate(  # type: ignore
                **inputs,
                max_new_tokens=max_new,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )
        text = _tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore

        bullets = []
        for line in text.split("\n"):
            line = line.strip(" -•\t")
            if line and len(line.split()) > 3:
                bullets.append(f"- {line}")
            if len(bullets) == 4:
                break
        cleaned = "\n".join(bullets)
        if not cleaned:
            m = re.findall(r"\d+\..{10,}", text)
            bullets = [f"- {s.strip()}" for s in m[:4]]
            cleaned = "\n".join(bullets) if bullets else "TBD"
        return cleaned or "TBD"

    try:
        return _run(MAX_NEW)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        _clear_cuda()
        try:
            _model.to("cpu")  # type: ignore
        except Exception:
            pass
        try:
            return _run(min(160, MAX_NEW))
        except Exception:
            return "TBD"

def _gen_test_procedure_stub(requirement_text: str) -> str:
    req = (requirement_text or "").strip() or "the feature"
    return (
        f"- Verify {req} at three setpoints; record measured vs setpoint (n=3).\n"
        f"- Confirm repeatability across 5 cycles; compute max deviation and std dev.\n"
        f"- Boundary test at min/max; record pass/fail vs spec and alarms.\n"
        f"- Capture equipment IDs and calibration; attach raw data."
    )

def _gen_test_procedure(requirement_text: str) -> str:
    return _gen_test_procedure_model(requirement_text) if USE_DVP_MODEL else _gen_test_procedure_stub(requirement_text)

# --------- Public API ----------
def dvp_predict(requirements: List[Dict[str, Any]], ha: List[Dict[str, Any]]):
    """
    Input rows must include:
      - 'Requirement ID' (or 'requirement_id')
      - 'Verification ID' (optional passthrough)
      - 'Requirements'   (main text)
    """
    _load_model()  # may be no-op if USE_DVP_MODEL=0

    rows: List[Dict[str, Any]] = []
    for r in (requirements or []):
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "")
        vid = str(r.get("Verification ID") or r.get("verification_id") or "")
        rtxt = str(r.get("Requirements") or r.get("requirements") or "").strip()

        if rtxt.lower().endswith("requirements") and not vid:
            rows.append({
                "verification_id": vid, "requirement_id": rid, "requirements": rtxt,
                "verification_method": "NA", "sample_size": "NA",
                "test_procedure": "NA", "acceptance_criteria": "NA",
            })
            continue

        method = _get_verification_method(rtxt)
        sample = _get_sample_size(rid, ha or [])
        bullets = _gen_test_procedure(rtxt)

        ac = "TBD"
        low = rtxt.lower()
        for kw, spec in TEST_SPEC_LOOKUP.items():
            if kw in low:
                ac = spec
                break

        rows.append({
            "verification_id": vid,
            "requirement_id": rid,
            "requirements": rtxt,
            "verification_method": method or "NA",
            "sample_size": sample or "NA",
            "test_procedure": bullets or "TBD",
            "acceptance_criteria": ac or "TBD",
        })
    return rows
