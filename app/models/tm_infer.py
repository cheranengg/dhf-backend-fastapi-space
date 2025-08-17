from __future__ import annotations
import os, re
from typing import Any, Dict, List

USE_TM_ADAPTER  = os.getenv("USE_TM_ADAPTER", "1") == "1"
TM_ADAPTER_REPO = os.getenv("TM_ADAPTER_REPO", "cheranengg/dhf-tm-adapter")
BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")
FORCE_CPU   = os.getenv("FORCE_CPU", "0") == "1"
DEBUG_TM    = os.getenv("DEBUG_TM", "1") == "1"

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

_tokenizer = None
_model     = None
_logged    = False

def _try_peft_loader():
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
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    try:
        from transformers import BitsAndBytesConfig
        dtype = torch.bfloat16 if (torch.cuda.is_available() and not FORCE_CPU and hasattr(torch, "bfloat16")) else (
                torch.float16 if (torch.cuda.is_available() and not FORCE_CPU) else torch.float32)
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    except Exception:
        bnb_cfg = None
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **_token_cache_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = dict(cache_dir=CACHE_DIR, low_cpu_mem_usage=True)
    if torch.cuda.is_available() and not FORCE_CPU:
        load_kwargs.update(dict(device_map="auto", torch_dtype=dtype, offload_folder=OFFLOAD_DIR))
    if bnb_cfg and not FORCE_CPU:
        load_kwargs.update(dict(quantization_config=bnb_cfg))
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
    if USE_TM_ADAPTER:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(base, TM_ADAPTER_REPO, **_token_cache_kwargs())
    else:
        mdl = base
    try:
        mdl.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    _tokenizer, _model = tok, mdl
    if DEBUG_TM and not _logged:
        dev = getattr(_model, "device", "cpu")
        print(f"[tm] model loaded (fallback) on {dev}; cache={CACHE_DIR} offload={OFFLOAD_DIR} adapter={USE_TM_ADAPTER}")
        _logged = True

def _unique_join(values: List[str], default: str = "TBD - Human / SME input") -> str:
    clean = []
    for v in values:
        s = (v or "").strip()
        if s and s.upper() != "NA" and s not in clean:
            clean.append(s)
    return ", ".join(clean) if clean else default

def _is_header(text: str) -> bool:
    return (text or "").strip().lower().endswith("requirements")

# relaxed token overlap (same as UI metric)
def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]{4,}", (text or "").lower()))

def _mapped(req: str, rc: str) -> bool:
    rt = _tokenize(req); ct = _tokenize(rc)
    if not rt or not ct: return False
    if rt & ct: return True
    for tkn in sorted(rt, key=len, reverse=True)[:5]:
        if tkn in (rc or "").lower():
            return True
    return False

def tm_predict(
    requirements: List[Dict[str, Any]],
    ha_rows: List[Dict[str, Any]],
    dvp_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    try:
        _load_model()
    except Exception as e:
        if DEBUG_TM:
            print(f"[tm] model load non-fatal: {e}")

    from collections import defaultdict
    ha_by_req: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in ha_rows or []:
        rid = str(h.get("requirement_id") or h.get("Requirement ID") or "").strip()
        if rid:
            ha_by_req[rid].append(h)

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

        req_yesno = "Reference" if _is_header(rtxt) else "Requirement"

        if _is_header(rtxt):
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
                "requirement_id": rid or "NA",
                "requirements": rtxt,
                "verification_id": vid or "NA",
                "verification_method": "NA",
                "acceptance_criteria": "NA",
            }
            rows.append(row)
            continue

        ha_slice = ha_by_req.get(rid, [])
        risk_ids = _unique_join([str(h.get("risk_id") or h.get("Risk ID") or "") for h in ha_slice])
        risks_to_health = _unique_join([str(h.get("risk_to_health") or h.get("Risk to Health") or "") for h in ha_slice])
        risk_controls = _unique_join([str(h.get("risk_control") or h.get("HA Risk Control") or "") for h in ha_slice])

        drow = dvp_by_vid.get(vid) or dvp_by_req.get(rid)
        vmethod = (str(drow.get("verification_method") or drow.get("Verification Method") or "").strip() if drow else "") or "TBD - Human / SME input"
        vaccept = (str(drow.get("acceptance_criteria") or drow.get("Acceptance Criteria") or "").strip() if drow else "") or "TBD - Human / SME input"
        vid_eff = str(drow.get("verification_id") or drow.get("Verification ID") or vid or "NA") if drow else (vid or "NA")

        # If risk control looks generic AND requirement keywords exist, try light tailoring (optional)
        if "design controls" in risk_controls.lower() and _mapped(rtxt, risk_controls):
            pass  # already overlaps; leave as-is

        rows.append({
            "Requirement ID": rid or "NA",
            "Requirements": rtxt,
            "Requirement (Yes/No)": req_yesno,
            "Risk ID": risk_ids,
            "Risk to Health": risks_to_health,
            "HA Risk Control": risk_controls,
            "Verification ID": vid_eff,
            "Verification Method": vmethod,
            "Acceptance Criteria": vaccept,
            "requirement_id": rid or "NA",
            "requirements": rtxt,
            "verification_id": vid_eff,
            "verification_method": vmethod,
            "acceptance_criteria": vaccept,
        })

    return rows
