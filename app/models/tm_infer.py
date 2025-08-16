# app/models/tm_infer.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

# ================================
# Environment / feature toggles
# ================================
USE_TM_ADAPTER  = os.getenv("USE_TM_ADAPTER", "1") == "1"
TM_ADAPTER_REPO = os.getenv("TM_ADAPTER_REPO", "cheranengg/dhf-tm-adapter")
BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)

# Prefer HF_HOME if present, else TRANSFORMERS_CACHE, else /tmp/hf
CACHE_DIR = (
    os.getenv("HF_HOME")
    or os.getenv("TRANSFORMERS_CACHE")
    or "/tmp/hf"
)

OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")
FORCE_CPU   = os.getenv("FORCE_CPU", "0") == "1"
DEBUG_TM    = os.getenv("DEBUG_TM", "1") == "1"

# Make sure the cache/offload dirs exist (avoid permission errors)
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
except Exception as _e:
    if DEBUG_TM:
        print(f"[tm] mkdirs failed (non-fatal): { _e }")

def _token_cache_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {"cache_dir": CACHE_DIR}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

# ================================
# (Optional) Model loader
#   We load the adapter to keep parity with HA/DVP.
#   TM logic below is deterministic joins; the model
#   can be used later for semantic scoring if desired.
# ================================
_tokenizer = None
_model     = None
_logged    = False

def _try_peft_loader():
    """Use project-specific PEFT loader if available; return (tok, model) or (None, None)."""
    try:
        from app.models._peft_loader import load_base_plus_adapter
        tok, mdl, device = load_base_plus_adapter(
            base_repo=BASE_MODEL_ID,
            adapter_repo=(TM_ADAPTER_REPO if USE_TM_ADAPTER else None),
            load_4bit=True,
            force_cpu=FORCE_CPU,
        )
        if DEBUG_TM:
            print(f"[tm] using _peft_loader on {device}; cache={CACHE_DIR} offload={OFFLOAD_DIR} adapter={USE_TM_ADAPTER}")
        return tok, mdl
    except Exception as e:
        if DEBUG_TM:
            print(f"[tm] _peft_loader load failed: {e}")
        return None, None

def _load_model():
    """Mirror dvp_infer loader pattern (adapter over base)."""
    global _tokenizer, _model, _logged
    if _tokenizer is not None and _model is not None:
        return

    tok, mdl = _try_peft_loader()
    if tok is not None and mdl is not None:
        _tokenizer, _model = tok, mdl
        if DEBUG_TM and not _logged:
            dev = getattr(_model, "device", "cpu")
            print(f"[tm] model via _peft_loader on {dev}")
            _logged = True
        return

    # ---- Manual fallback load (4-bit if possible) ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        dtype = (
            torch.bfloat16 if (torch.cuda.is_available() and not FORCE_CPU and hasattr(torch, "bfloat16"))
            else (torch.float16 if (torch.cuda.is_available() and not FORCE_CPU) else torch.float32)
        )
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    except Exception:
        bnb_cfg = None

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **_token_cache_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(cache_dir=CACHE_DIR, low_cpu_mem_usage=True)
    if torch.cuda.is_available() and not FORCE_CPU:
        load_kwargs.update(dict(
            device_map="auto",
            torch_dtype=torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16,
            offload_folder=OFFLOAD_DIR,
        ))
    if bnb_cfg and not FORCE_CPU:
        load_kwargs.update(dict(quantization_config=bnb_cfg))

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)

    if USE_TM_ADAPTER:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(base, TM_ADAPTER_REPO, **_token_cache_kwargs())
    else:
        mdl = base

    try:
        mdl.config.pad_token_id = tok.pad_token_id  # type: ignore
    except Exception:
        pass

    _tokenizer, _model = tok, mdl
    if DEBUG_TM and not _logged:
        dev = getattr(_model, "device", "cpu")
        print(f"[tm] model loaded (fallback) on {dev}; cache={CACHE_DIR} offload={OFFLOAD_DIR} adapter={USE_TM_ADAPTER}")
        _logged = True


# ================================
# Deterministic join helpers
# ================================
def _unique_join(values: List[str], default: str = "NA") -> str:
    clean = []
    for v in values:
        s = (v or "").strip()
        if s and s.upper() != "NA" and s not in clean:
            clean.append(s)
    return ", ".join(clean) if clean else default

def _is_header(text: str) -> bool:
    return (text or "").strip().lower().endswith("requirements")

# ================================
# Public API
# ================================
def tm_predict(
    requirements: List[Dict[str, Any]],
    ha_rows: List[Dict[str, Any]],
    dvp_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build the Trace Matrix by combining:
      - Product Requirements
      - Hazard Analysis rows (by Requirement ID)
      - DVP rows (by Verification ID, fallback by Requirement ID)

    Returns rows with canonical, Excel-ready headers:
      "Requirement ID", "Requirements", "Requirement (Yes/No)",
      "Risk ID", "Risk to Health", "HA Risk Control",
      "Verification ID", "Verification Method", "Acceptance Criteria"

    Also includes snake_case aliases for backward compatibility:
      risk_ids, risks_to_health, ha_risk_controls, verification_method, acceptance_criteria
    """
    # Load adapter (optional; keeps parity with HA/DVP)
    try:
        _load_model()
    except Exception as e:
        if DEBUG_TM:
            print(f"[tm] model load non-fatal: {e}")

    # Index HA by Requirement ID
    from collections import defaultdict
    ha_by_req: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in ha_rows or []:
        rid = str(h.get("requirement_id") or h.get("Requirement ID") or "").strip()
        if rid:
            ha_by_req[rid].append(h)

    # Index DVP by Verification ID and also by Requirement ID (fallback)
    dvp_by_vid: Dict[str, Dict[str, Any]] = {}
    dvp_by_req: Dict[str, Dict[str, Any]] = {}
    for d in dvp_rows or []:
        vid = str(d.get("verification_id") or d.get("Verification ID") or "").strip()
        rid = str(d.get("requirement_id") or d.get("Requirement ID") or "").strip()
        if vid and vid not in dvp_by_vid:
            dvp_by_vid[vid] = d
        if rid and rid not in dvp_by_req:
            dvp_by_req[rid] = d

    rows: List[Dict[str, Any]] = []

    for r in (requirements or []):
        rid = str(r.get("Requirement ID") or r.get("requirement_id") or "").strip()
        vid = str(r.get("Verification ID") or r.get("verification_id") or "").strip()
        rtxt = str(r.get("Requirements") or r.get("requirements") or "").strip()

        # Decide "Requirement (Yes/No)" column
        req_yesno = "Reference" if _is_header(rtxt) else "Requirement"

        if _is_header(rtxt):
            # Header rows: NA for analysis/verification fields
            row = {
                "Requirement ID": rid or "NA",
                "Requirements": rtxt,
                "Requirement (Yes/No)": req_yesno,
                "Risk ID": "NA",
                "Risk to Health": "NA",
                "HA Risk Control": "NA",
                "Verification ID": vid or "NA",
                "Verification Method": "NA",
                "Acceptance Criteria": "NA",
                # aliases for backward compatibility
                "risk_ids": "NA",
                "risks_to_health": "NA",
                "ha_risk_controls": "NA",
                "verification_id": vid or "NA",
                "verification_method": "NA",
                "acceptance_criteria": "NA",
                "requirement_id": rid or "NA",
                "requirements": rtxt,
            }
            rows.append(row)
            continue

        # ---- Non-header rows ----
        ha_slice = ha_by_req.get(rid, [])
        risk_ids = _unique_join([str(h.get("risk_id") or h.get("Risk ID") or "") for h in ha_slice], default="TBD - Human / SME input")
        risks_to_health = _unique_join([str(h.get("risk_to_health") or h.get("Risk to Health") or "") for h in ha_slice], default="TBD - Human / SME input")
        risk_controls = _unique_join([str(h.get("risk_control") or h.get("HA Risk Control") or "") for h in ha_slice], default="TBD - Human / SME input")

        drow = dvp_by_vid.get(vid)
        if not drow:
            drow = dvp_by_req.get(rid)  # fallback if VID missing

        vmethod = str(drow.get("verification_method") or drow.get("Verification Method") or "").strip() if drow else ""
        vaccept = str(drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "").strip() if drow else ""
        vmethod = vmethod if vmethod else "TBD - Human / SME input"
        vaccept = vaccept if vaccept else "TBD - Human / SME input"
        vid_eff = str(drow.get("verification_id") or drow.get("Verification ID") or vid or "NA") if drow else (vid or "NA")

        row = {
            # Canonical, Excel-ready names (match your template order)
            "Requirement ID": rid or "NA",
            "Requirements": rtxt,
            "Requirement (Yes/No)": req_yesno,
            "Risk ID": risk_ids,
            "Risk to Health": risks_to_health,
            "HA Risk Control": risk_controls,
            "Verification ID": vid_eff,
            "Verification Method": vmethod,
            "Acceptance Criteria": vaccept,
            # Aliases (snake_case) so older UI code won't break
            "requirement_id": rid or "NA",
            "requirements": rtxt,
            "risk_ids": risk_ids,
            "risks_to_health": risks_to_health,
            "ha_risk_controls": risk_controls,
            "verification_id": vid_eff,
            "verification_method": vmethod,
            "acceptance_criteria": vaccept,
        }
        rows.append(row)

    return rows
