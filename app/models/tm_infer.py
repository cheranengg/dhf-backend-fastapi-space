# app/models/tm_infer.py
from __future__ import annotations
import os, re, json
from typing import Any, Dict, List, Optional, Tuple

# ========= Env / toggles =========
USE_TM_ADAPTER  = os.getenv("USE_TM_ADAPTER", "1") == "1"
TM_ADAPTER_REPO = os.getenv("TM_ADAPTER_REPO", "cheranengg/dhf-tm-adapter")
BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)
CACHE_DIR   = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/tmp/hf"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/tmp/offload")
FORCE_CPU   = os.getenv("FORCE_CPU", "0") == "1"

# Optional gen controls (kept for parity/future normalization)
TM_INPUT_MAX_TOKENS   = int(os.getenv("TM_INPUT_MAX_TOKENS", "512"))
TM_MAX_NEW_TOKENS     = int(os.getenv("TM_MAX_NEW_TOKENS", "160"))
TM_TEMPERATURE        = float(os.getenv("TM_TEMPERATURE", "0.20"))
TM_TOP_P              = float(os.getenv("TM_TOP_P", "0.90"))
TM_DO_SAMPLE          = os.getenv("TM_DO_SAMPLE", "1") == "1"
TM_NUM_BEAMS          = int(os.getenv("TM_NUM_BEAMS", "1"))
TM_REPETITION_PENALTY = float(os.getenv("TM_REPETITION_PENALTY", "1.05"))

DEBUG_TM    = os.getenv("DEBUG_TM", "1") == "1"
TOP_HA      = int(os.getenv("TM_TOP_HA", "4"))      # max HA matches aggregated per req
MIN_OVERLAP = int(os.getenv("TM_MIN_OVERLAP", "2")) # min token overlap to accept an HA match

# ========= Model (optional; deterministic TM logic does not require it) =========
_tokenizer = None
_model = None
_loaded_banner = False

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
            print(f"[tm] using _peft_loader (device={device}) cache={CACHE_DIR} offload={OFFLOAD_DIR} adapter={USE_TM_ADAPTER}")
        return tok, mdl
    except Exception as e:
        if DEBUG_TM:
            print(f"[tm] _peft_loader failed: {e}")
        return None, None

def _load_model():
    global _tokenizer, _model, _loaded_banner
    if _tokenizer is not None and _model is not None:
        return
    tok, mdl = _try_peft_loader()
    if tok is not None and mdl is not None:
        _tokenizer, _model = tok, mdl
        if DEBUG_TM and not _loaded_banner:
            dev = getattr(_model, "device", "cpu")
            print(f"[tm] model ready on {dev} (adapter={'on' if USE_TM_ADAPTER else 'off'})")
            _loaded_banner = True
        return
    if DEBUG_TM and not _loaded_banner:
        print("[tm] model not loaded (deterministic TM will still run).")
        _loaded_banner = True

# ========= Text utils / matching =========
_STOP = {
    "the","a","an","and","or","to","of","in","on","for","by","with","at","as","from",
    "shall","must","should","be","is","are","was","were","it","this","that","these","those",
    "within","under","per","per-","per—","per–","than","then","into","over","across"
}
_token_pat = re.compile(r"[a-z0-9µ°]+")  # alnum + micro/degree

def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())

def _tokens(s: str) -> List[str]:
    toks = [t for t in _token_pat.findall((s or "").lower()) if t and t not in _STOP and len(t) > 1]
    return toks

def _is_header(req_text: str) -> bool:
    t = (req_text or "").strip().lower()
    return t.endswith("requirements")

def _join_unique(values: List[str], fallback: str = "NA") -> str:
    out: List[str] = []
    for v in values:
        vv = _norm(v)
        if vv and vv.upper() != "NA" and vv not in out:
            out.append(vv)
    return ", ".join(out) if out else fallback

# ========= Index HA/DVP =========
def _index_ha(ha_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normed: List[Dict[str, Any]] = []
    for h in ha_rows or []:
        normed.append({
            "requirement_id":  _norm(h.get("requirement_id") or h.get("Requirement ID")),
            "risk_id":         _norm(h.get("risk_id") or h.get("Risk ID")),
            "risk_to_health":  _norm(h.get("risk_to_health") or h.get("Risk to Health")),
            "risk_control":    _norm(h.get("risk_control") or h.get("HA Risk Control") or h.get("HA Risk Controls")),
        })
    return normed

def _index_dvp(dvp_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_vid: Dict[str, Dict[str, Any]] = {}
    for d in dvp_rows or []:
        vid = _norm(d.get("verification_id") or d.get("Verification ID"))
        if vid and vid not in by_vid:
            by_vid[vid] = {
                "verification_id":      vid,
                "requirement_id":       _norm(d.get("requirement_id") or d.get("Requirement ID")),
                "verification_method":  _norm(d.get("verification_method") or d.get("Verification Method")),
                "acceptance_criteria":  _norm(d.get("acceptance_criteria") or d.get("Acceptance Criteria")),
            }
    return by_vid

def _score_overlap(req_text: str, ha_row: Dict[str, Any]) -> int:
    rt = set(_tokens(req_text))
    if not rt:
        return 0
    rc = set(_tokens(ha_row.get("risk_control","")))
    r2h= set(_tokens(ha_row.get("risk_to_health","")))
    return 2 * len(rt & rc) + 1 * len(rt & r2h)

def _pick_ha_matches(req_text: str, ha_idx: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for h in ha_idx:
        sc = _score_overlap(req_text, h)
        if sc >= MIN_OVERLAP:
            scored.append((sc, h))
    if not scored:
        return []
    scored.sort(key=lambda x: -x[0])
    keep = [h for _, h in scored[:TOP_HA]]
    return keep

def _pick_dvp(req_row: Dict[str, Any], dvp_by_vid: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    vid = _norm(req_row.get("Verification ID") or req_row.get("verification_id"))
    if vid and vid in dvp_by_vid:
        return dvp_by_vid[vid]
    req_text = _norm(req_row.get("Requirements") or req_row.get("requirements"))
    rtoks = set(_tokens(req_text))
    best, best_sc = None, 0
    for d in dvp_by_vid.values():
        ac = set(_tokens(d.get("acceptance_criteria","")))
        vm = set(_tokens(d.get("verification_method","")))
        sc = 2*len(rtoks & ac) + 1*len(rtoks & vm)
        if sc > best_sc:
            best, best_sc = d, sc
    return best

# ========= Public API =========
def tm_predict(requirements: List[Dict[str, Any]],
               ha_rows: List[Dict[str, Any]],
               dvp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Output columns:
      - verification_id, requirement_id, requirements,
      - Requirement (Yes/No),
      - risk_ids, risks_to_health, ha_risk_controls,
      - verification_method, acceptance_criteria
    """
    _load_model()  # harmless; TM logic works without the model

    ha_idx = _index_ha(ha_rows or [])
    dvp_by_vid = _index_dvp(dvp_rows or [])

    out: List[Dict[str, Any]] = []

    for r in requirements or []:
        rid = _norm(r.get("Requirement ID") or r.get("requirement_id"))
        rtxt = _norm(r.get("Requirements") or r.get("requirements"))

        # Section headers → structural row with NA and "Reference"
        if _is_header(rtxt):
            out.append({
                "verification_id": _norm(r.get("Verification ID") or r.get("verification_id") or "NA"),
                "requirement_id": rid,
                "requirements": rtxt,
                "Requirement (Yes/No)": "Reference",
                "risk_ids": "NA",
                "risks_to_health": "NA",
                "ha_risk_controls": "NA",
                "verification_method": "NA",
                "acceptance_criteria": "NA",
            })
            continue

        # Normal rows → "Requirement"
        req_yes_no = "Requirement"

        # HA: collect ALL matches (weighted by overlap with Risk Control, then Risk to Health)
        matches = _pick_ha_matches(rtxt, ha_idx)
        if not matches:
            # fallback: any HA rows tied to same Requirement ID (if present)
            matches = [h for h in ha_idx if h.get("requirement_id") == rid]

        risk_ids        = _join_unique([m.get("risk_id","") for m in matches]) or "TBD - Human / SME input"
        risks_to_health = _join_unique([m.get("risk_to_health","") for m in matches]) or "TBD - Human / SME input"
        ha_controls     = _join_unique([m.get("risk_control","") for m in matches]) or "TBD - Human / SME input"

        # DVP alignment
        drow = _pick_dvp(r, dvp_by_vid) or {}
        v_id   = drow.get("verification_id") or _norm(r.get("Verification ID") or r.get("verification_id") or "TBD - Human / SME input")
        v_meth = drow.get("verification_method") or "TBD - Human / SME input"
        v_ac   = drow.get("acceptance_criteria") or "TBD - Human / SME input"

        out.append({
            "verification_id": v_id,
            "requirement_id": rid,
            "requirements": rtxt,
            "Requirement (Yes/No)": req_yes_no,
            "risk_ids": risk_ids,
            "risks_to_health": risks_to_health,
            "ha_risk_controls": ha_controls,
            "verification_method": v_meth,
            "acceptance_criteria": v_ac,
        })

    return out
