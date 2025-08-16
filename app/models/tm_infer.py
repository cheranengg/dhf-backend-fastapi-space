# app/models/tm_infer.py
from __future__ import annotations
import os, re
from typing import Any, Dict, List, Optional, Tuple

# ========= Env / toggles =========
USE_TM_ADAPTER  = os.getenv("USE_TM_ADAPTER", "1") == "1"
TM_ADAPTER_REPO = os.getenv("TM_ADAPTER_REPO", "cheranengg/dhf-tm-adapter")
BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

CACHE_DIR   = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or "/data/hf"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "/data/offload")
FORCE_CPU   = os.getenv("FORCE_CPU", "0") == "1"

DEBUG_TM    = os.getenv("DEBUG_TM", "1") == "1"
TOP_HA      = int(os.getenv("TM_TOP_HA", "4"))      # max HA matches aggregated per req
MIN_OVERLAP = int(os.getenv("TM_MIN_OVERLAP", "2")) # min token overlap to accept an HA match

# ========= (Optional) adapter load — TM logic is deterministic even without it =========
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
            print(f"[tm] _peft_loader ok (device={device}) cache={CACHE_DIR} offload={OFFLOAD_DIR} adapter={USE_TM_ADAPTER}")
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
            print(f"[tm] model ready on {dev}")
            _loaded_banner = True
        return
    if DEBUG_TM and not _loaded_banner:
        print("[tm] model not loaded (deterministic TM join will still run).")
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
    return [t for t in _token_pat.findall((s or "").lower()) if t and t not in _STOP and len(t) > 1]

def _is_header(req_text: str) -> bool:
    return (req_text or "").strip().lower().endswith("requirements")

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
                # acceptance criteria intentionally NOT used in TM per requested columns
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
    return [h for _, h in scored[:TOP_HA]]

def _pick_dvp(req_row: Dict[str, Any], dvp_by_vid: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    vid = _norm(req_row.get("Verification ID") or req_row.get("verification_id"))
    if vid and vid in dvp_by_vid:
        return dvp_by_vid[vid]
    # fallback: best semantic overlap vs. method text
    req_text = _norm(req_row.get("Requirements") or req_row.get("requirements"))
    rtoks = set(_tokens(req_text))
    best, best_sc = None, 0
    for d in dvp_by_vid.values():
        vm = set(_tokens(d.get("verification_method","")))
        sc = len(rtoks & vm)
        if sc > best_sc:
            best, best_sc = d, sc
    return best

# ========= Public API =========
def tm_predict(requirements: List[Dict[str, Any]],
               ha_rows: List[Dict[str, Any]],
               dvp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns rows with EXACT column names and order expected by your Excel:
      1) Requirement ID
      2) Requirements
      3) Requirement (Yes/No)
      4) Risk ID
      5) Risk to Health
      6) HA Risk Control
      7) Verification ID
      8) Verification Method
    """
    _load_model()  # harmless if adapter not loaded; TM matching is deterministic

    ha_idx = _index_ha(ha_rows or [])
    dvp_by_vid = _index_dvp(dvp_rows or [])

    out: List[Dict[str, Any]] = []

    for r in requirements or []:
        rid  = _norm(r.get("Requirement ID") or r.get("requirement_id"))
        rtxt = _norm(r.get("Requirements") or r.get("requirements"))

        if _is_header(rtxt):
            # Section header: mark as Reference, keep everything else NA
            out.append({
                "Requirement ID": rid,
                "Requirements": rtxt,
                "Requirement (Yes/No)": "Reference",
                "Risk ID": "NA",
                "Risk to Health": "NA",
                "HA Risk Control": "NA",
                "Verification ID": _norm(r.get("Verification ID") or r.get("verification_id") or "NA"),
                "Verification Method": "NA",
            })
            continue

        # Normal requirement
        matches = _pick_ha_matches(rtxt, ha_idx)
        if not matches:
            # fallback: same requirement_id linkage
            matches = [h for h in ha_idx if h.get("requirement_id") == rid]

        risk_ids        = _join_unique([m.get("risk_id","") for m in matches]) or "TBD - Human / SME input"
        risks_to_health = _join_unique([m.get("risk_to_health","") for m in matches]) or "TBD - Human / SME input"
        ha_controls     = _join_unique([m.get("risk_control","") for m in matches]) or "TBD - Human / SME input"

        drow = _pick_dvp(r, dvp_by_vid) or {}
        vid   = drow.get("verification_id") or _norm(r.get("Verification ID") or r.get("verification_id") or "TBD - Human / SME input")
        vmet  = drow.get("verification_method") or "TBD - Human / SME input"

        out.append({
            "Requirement ID": rid,
            "Requirements": rtxt,
            "Requirement (Yes/No)": "Requirement",
            "Risk ID": risk_ids,
            "Risk to Health": risks_to_health,
            "HA Risk Control": ha_controls,
            "Verification ID": vid,
            "Verification Method": vmet,
        })

    return out
