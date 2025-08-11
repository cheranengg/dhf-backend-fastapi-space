# app/utils/guardrails.py
# -----------------------
# Lightweight guardrails + normalization helpers for DHF (HA / DVP / TM).
# - No hard deps beyond stdlib. If pandas is installed, DataFrame helpers are enabled.
# - Designed to be used in model wrappers before returning JSON to the client.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional pandas support (used only if present)
try:
    import pandas as _pd  # type: ignore
    _HAS_PD = True
except Exception:
    _HAS_PD = False

# -----------------------------
# Constants / simple vocab
# -----------------------------
TBD = "TBD - Human / SME input"

HEADING_TOKENS = {
    "functional requirements",
    "performance requirements",
    "safety requirements",
    "usability requirements",
    "environmental requirements",
    "design inputs",
    "general requirements",
}

# Allowlist for DVP verification methods (you can extend/override via function args)
DEFAULT_ALLOWED_METHODS = {
    "Physical Testing",
    "Physical Inspection",
    "Visual Inspection",
    "NA",
}

# Minimal length for acceptance criteria to be considered non-trivial
MIN_ACCEPTANCE_LEN = 10

# ISO 14971-ish convenience map (example; adjust per your org’s matrix)
SEVERITY_TO_SAMPLE_SIZE = {5: 50, 4: 40, 3: 30, 2: 20, 1: 10}

# JSON object extractor (last JSON object in a string)
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*?\}")

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class Issue:
    row_index: Optional[int]
    column: str
    issue: str
    suggestion: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailsResult:
    ok: bool
    issues: List[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    @property
    def count(self) -> int:
        return len(self.issues)


# -----------------------------
# Normalization utilities
# -----------------------------
def sanitize_text(x: Any) -> str:
    """Return a safe, single-line string."""
    if x is None:
        return ""
    s = str(x).strip()
    return re.sub(r"\s+", " ", s)


def ensure_tbd(value: Any) -> str:
    """Return value if meaningful; else the standard TBD token."""
    s = sanitize_text(value)
    return s if s and s.upper() != "NA" else TBD


def is_heading(requirements_text: str, verification_id: str | None) -> bool:
    """Detect section-heading rows that should propagate NA fields."""
    if not sanitize_text(verification_id):
        return True
    return sanitize_text(requirements_text).lower() in HEADING_TOKENS


def join_unique(values: Iterable[Any], sep: str = ", ") -> str:
    seen: List[str] = []
    for v in values:
        s = sanitize_text(v)
        if s and s.upper() != "NA" and s not in seen:
            seen.append(s)
    return sep.join(seen) if seen else "NA"


# -----------------------------
# JSON helpers
# -----------------------------
def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the last JSON object in a text block.
    Returns None if not found or parse fails.
    """
    if not text:
        return None
    matches = _JSON_OBJ_RE.findall(text)
    if not matches:
        return None
    js = (
        matches[-1]
        .replace("'", '"')
        .replace("\\n", " ")
    )
    js = re.sub(r"\s+", " ", js)
    js = re.sub(r",\s*\}", "}", js)
    js = re.sub(r",\s*\]", "]", js)
    try:
        return json.loads(js)
    except Exception:
        return None


# -----------------------------
# Field-level checks
# -----------------------------
def validate_required_fields(
    row: Dict[str, Any],
    required: Sequence[str],
    row_index: Optional[int] = None,
) -> List[Issue]:
    issues: List[Issue] = []
    for col in required:
        val = sanitize_text(row.get(col, ""))
        if not val or val in {"NA", TBD}:
            issues.append(Issue(row_index, col, "Missing or TBD"))
    return issues


def validate_verification_method(
    row: Dict[str, Any],
    allowed: Optional[Iterable[str]] = None,
    row_index: Optional[int] = None,
    column: str = "verification_method",
) -> List[Issue]:
    issues: List[Issue] = []
    allow = set(allowed) if allowed else set(DEFAULT_ALLOWED_METHODS)
    vm = sanitize_text(row.get(column, ""))
    if vm and vm not in {"NA", TBD} and vm not in allow:
        issues.append(Issue(row_index, column, f"Method '{vm}' not in allowlist"))
    return issues


def validate_acceptance_criteria(
    row: Dict[str, Any],
    row_index: Optional[int] = None,
    column: str = "acceptance_criteria",
    min_len: int = MIN_ACCEPTANCE_LEN,
) -> List[Issue]:
    issues: List[Issue] = []
    ac = sanitize_text(row.get(column, ""))
    if ac and ac not in {"NA", TBD} and len(ac) < min_len:
        issues.append(Issue(row_index, column, f"Criteria too short (<{min_len} chars)"))
    return issues


def validate_numeric(
    row: Dict[str, Any],
    field: str,
    row_index: Optional[int] = None,
) -> List[Issue]:
    issues: List[Issue] = []
    val = sanitize_text(row.get(field, ""))
    if val and val not in {"NA", TBD}:
        if not re.fullmatch(r"-?\d+(\.\d+)?", val):
            issues.append(Issue(row_index, field, "Expected numeric value"))
    return issues


# -----------------------------
# Row-level validators (HA / DVP / TM)
# -----------------------------
def validate_ha_row(row: Dict[str, Any], row_index: Optional[int] = None) -> List[Issue]:
    req = ["requirement_id", "risk_id", "risk_to_health"]
    issues = validate_required_fields(row, req, row_index)
    # Optional: check severity ranges if present
    sev = sanitize_text(row.get("severity_of_harm", ""))
    if sev and sev not in {"NA", TBD}:
        try:
            n = int(float(sev))
            if n < 1 or n > 5:
                issues.append(Issue(row_index, "severity_of_harm", "Out of range [1..5]"))
        except Exception:
            issues.append(Issue(row_index, "severity_of_harm", "Not an integer 1..5"))
    return issues


def validate_dvp_row(
    row: Dict[str, Any],
    row_index: Optional[int] = None,
    allowed_methods: Optional[Iterable[str]] = None,
) -> List[Issue]:
    issues: List[Issue] = []
    issues += validate_required_fields(row, ["verification_id"], row_index)
    issues += validate_verification_method(row, allowed_methods, row_index, "verification_method")
    issues += validate_acceptance_criteria(row, row_index, "acceptance_criteria")
    issues += validate_numeric(row, "sample_size", row_index)
    return issues


def validate_tm_row(
    row: Dict[str, Any],
    row_index: Optional[int] = None,
    allowed_methods: Optional[Iterable[str]] = None,
) -> List[Issue]:
    required = [
        "verification_id",
        "requirement_id",
        "requirements",
        "risk_ids",
        "risks_to_health",
        "ha_risk_controls",
        "verification_method",
        "acceptance_criteria",
    ]
    issues: List[Issue] = []
    issues += validate_required_fields(row, required, row_index)
    issues += validate_verification_method(row, allowed_methods, row_index, "verification_method")
    issues += validate_acceptance_criteria(row, row_index, "acceptance_criteria")
    return issues


# -----------------------------
# Enrichment helpers
# -----------------------------
def severity_to_sample_size(severity: Any, default: int = 30) -> int:
    """Map severity (1..5) to a suggested sample size."""
    try:
        n = int(float(sanitize_text(severity)))
        return SEVERITY_TO_SAMPLE_SIZE.get(n, default)
    except Exception:
        return default


def fill_missing_with_tbd(row: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    out = dict(row)
    for k in keys:
        out[k] = ensure_tbd(out.get(k))
    return out


def normalize_tm_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize/clean a TM row and apply TBDs.
    Keeps keys stable for downstream Excel export.
    """
    keys = [
        "verification_id",
        "requirement_id",
        "requirements",
        "risk_ids",
        "risks_to_health",
        "ha_risk_controls",
        "verification_method",
        "acceptance_criteria",
    ]
    out = {k: sanitize_text(row.get(k, "")) for k in keys}
    for k in keys:
        if not out[k] or out[k].upper() == "NA":
            out[k] = TBD
    return out


# -----------------------------
# Optional pandas helpers
# -----------------------------
def dataframe_guardrails(
    df: "Optional[_pd.DataFrame]",
    row_validator,
    allowed_methods: Optional[Iterable[str]] = None,
) -> "Optional[_pd.DataFrame]":
    """
    If pandas is available, run a row validator across the DataFrame and
    return a DataFrame of issues (row_index, column, issue, suggestion?).
    """
    if not _HAS_PD or df is None:
        return None
    issues: List[Tuple[int, str, str]] = []
    for i, row in df.iterrows():
        items = row_validator(row.to_dict(), i, allowed_methods=allowed_methods)  # type: ignore
        for it in items:
            issues.append((i, it.column, it.issue))
    return _pd.DataFrame(issues, columns=["row_index", "column", "issue"]) if issues else _pd.DataFrame(columns=["row_index", "column", "issue"])


def dataframe_fill_tbd(df: "Optional[_pd.DataFrame]", cols: Sequence[str]) -> "Optional[_pd.DataFrame]":
    if not _HAS_PD or df is None:
        return df
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = TBD
        out[c] = out[c].astype(str).fillna(TBD)
        out.loc[out[c].str.strip().eq("") | out[c].str.upper().eq("NA"), c] = TBD
    return out


# -----------------------------
# Convenience: one-shot TM guard
# -----------------------------
def guard_tm_rows(
    rows: List[Dict[str, Any]],
    allowed_methods: Optional[Iterable[str]] = None,
) -> GuardrailsResult:
    res = GuardrailsResult(ok=True)
    for idx, r in enumerate(rows):
        issues = validate_tm_row(r, idx, allowed_methods=allowed_methods)
        if issues:
            res.ok = False
            for it in issues:
                res.add(it)
    return res
