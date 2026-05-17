"""
Productivity-based loop detection helpers.

Replaces the keyword-based `is_failure` check (which only caught steps whose
output contained "failed" / "error") with an LLM-emitted verdict that classifies
every tool call into one of five productivity buckets:

    new_info      — the call revealed something we did not already know
    confirmation  — already suspected, this call only confirms
    no_progress   — call succeeded but yielded no usable information
    blocked       — WAF, 403, captcha, rate limit, auth wall
    duplicate     — output essentially identical to a recent call

The verdict lives on `OutputAnalysisInline.productivity` (see state.py). This
module exposes:

    is_unproductive(step)               read the verdict; returns bool
    audit_productivity_claim(step,      cross-check the LLM's claim against
                             before,    actual state growth; returns a
                             after)     discrepancy string or None
    build_productivity_audit_section(   compute the per-iteration prompt
        execution_trace, window)        block that shows the model its own
                                        recent fingerprints, so claiming
                                        "confirmation" 10 times in a row
                                        becomes visibly dishonest

The orchestrator owns three small responsibilities: show history in the
prompt, audit the claim against state delta, count unproductive steps. The
model owns the per-step judgment.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional


def _normalize_args_pattern(tool_name: str, tool_args: dict) -> str:
    """Generalize tool args to a 'shape' so /order/300500 and /order/300600
    collapse into the same pattern. Integers become <int>, hex tokens become
    <hex>, query-string values become <val>, IPs become <ip>.
    """
    try:
        raw = json.dumps(tool_args or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        raw = str(tool_args or {})
    # Strip every long alphanumeric token; the URL path shape is what matters.
    normalized = re.sub(r"\b\d+\b", "<int>", raw)
    normalized = re.sub(r"\b[a-f0-9]{8,}\b", "<hex>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "<ip>", normalized)
    normalized = re.sub(r"=[^&\"'\s]+", "=<val>", normalized)
    return f"{tool_name or '?'}::{normalized[:160]}"


def _output_fingerprint(step: dict) -> str:
    """Stable 8-hex fingerprint of the response body, normalized for trivial
    diffs (whitespace, timestamps, common varying tokens). Two responses with
    the same fingerprint are functionally identical."""
    raw = (step.get("tool_output") or "")[:8000]
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", raw).strip()
    # Strip ISO timestamps, UUIDs, RFC3339, request IDs
    normalized = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.\-+Z]+", "<ts>", normalized)
    normalized = re.sub(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "<uuid>", normalized)
    normalized = re.sub(r"\b\d{10,}\b", "<num>", normalized)
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()[:8]


def _output_size(step: dict) -> int:
    return len((step.get("tool_output") or ""))


def _read_productivity(step: dict) -> dict:
    """Read the productivity verdict from a step, tolerating both the
    top-level shape (step["productivity"]) and the nested shape
    (step["output_analysis"]["productivity"]). The think_node stores it at
    the top level; this dual-path lookup keeps the helper robust against
    future schema drift."""
    if not step:
        return {}
    top = step.get("productivity")
    if isinstance(top, dict) and top:
        return top
    nested = step.get("output_analysis")
    if isinstance(nested, dict):
        p = nested.get("productivity") or {}
        if isinstance(p, dict):
            return p
    return {}


def is_unproductive(step: dict) -> bool:
    """Read the LLM's productivity verdict for this step. The orchestrator's
    loop counter ORs this with the legacy keyword check, so a missing field
    just falls back to keyword behavior (productive-by-default)."""
    p = _read_productivity(step)
    if not p:
        return False
    if p.get("verdict") in ("no_progress", "duplicate", "blocked"):
        return True
    if p.get("new_information_gained") is False:
        return True
    return False


def _target_info_grew(before: dict, after: dict) -> bool:
    """True if any list-typed field in target_info grew between iterations."""
    b = (before or {}).get("target_info") or {}
    a = (after or {}).get("target_info") or {}
    for key in ("ports", "services", "technologies", "vulnerabilities",
                "credentials", "sessions", "subdomains", "endpoints"):
        if len(a.get(key, []) or []) > len(b.get(key, []) or []):
            return True
    return False


def audit_productivity_claim(
    productivity: dict,
    extracted_info: dict,
    actionable_findings: list,
    findings_grew: bool,
) -> Optional[str]:
    """Cross-check the LLM's productivity claim against actual state delta.

    Returns a one-line discrepancy string if the claim is inconsistent, else
    None. Callers typically downgrade the verdict to 'no_progress' in place
    and surface the reason in the next prompt.

    Inputs are plain dicts (no Pydantic dependency) so this helper is reusable
    from both root and fireteam paths and from tests.
    """
    if not productivity:
        return None

    verdict = productivity.get("verdict")
    claims_new = productivity.get("new_information_gained", False)

    extracted_any = any(
        (extracted_info or {}).get(k)
        for k in ("ports", "services", "technologies",
                  "vulnerabilities", "credentials", "sessions")
    )
    state_grew = bool(findings_grew or extracted_any or actionable_findings)

    if claims_new and not state_grew:
        return ("Claimed new_information_gained=true but no chain finding was "
                "appended, no extracted_info was populated, and no actionable "
                "finding was produced.")
    if verdict == "new_info" and not state_grew:
        return ("Verdict='new_info' but the engagement state did not grow this "
                "iteration.")
    return None


def downgrade_verdict_to_no_progress(productivity: dict, reason: str) -> dict:
    """Return a copy of the productivity dict with the verdict downgraded to
    'no_progress' and the reason recorded. Caller is responsible for writing
    the returned dict back onto whatever shape the step expects."""
    if not productivity:
        return {
            "verdict": "no_progress",
            "new_information_gained": False,
            "what_was_new": "",
            "should_repeat_similar_call": False,
            "rationale": "",
            "_original_verdict": None,
            "_downgrade_reason": reason,
        }
    out = dict(productivity)
    out["_original_verdict"] = out.get("verdict")
    out["verdict"] = "no_progress"
    out["new_information_gained"] = False
    out["_downgrade_reason"] = reason
    return out


def build_productivity_audit_section(
    execution_trace: list,
    current_tool_name: Optional[str] = None,
    current_tool_args: Optional[dict] = None,
    window: int = 6,
) -> str:
    """Build the prompt block that shows the model its own recent same-pattern
    fingerprints. Returns empty string if fewer than 3 same-pattern calls
    are in the recent window (no audit needed yet).

    The presence of this block is what makes the LLM verdict robust: when
    three of the last four calls share fingerprint a7c3 and produced no
    finding, claiming "confirmation" on the fourth is visibly dishonest.
    """
    if not execution_trace:
        return ""

    recent = execution_trace[-max(window, 1):]
    if current_tool_name and current_tool_args is not None:
        target_pattern = _normalize_args_pattern(current_tool_name, current_tool_args)
        same = [s for s in recent
                if _normalize_args_pattern(s.get("tool_name"), s.get("tool_args") or {}) == target_pattern]
    else:
        # No specific current step: pick the most-repeated pattern in the window.
        counts: dict[str, list] = {}
        for s in recent:
            sig = _normalize_args_pattern(s.get("tool_name"), s.get("tool_args") or {})
            counts.setdefault(sig, []).append(s)
        if not counts:
            return ""
        target_pattern, same = max(counts.items(), key=lambda kv: len(kv[1]))

    if len(same) < 3:
        return ""

    lines = []
    for s in same:
        fp = _output_fingerprint(s)
        size = _output_size(s)
        args_short = json.dumps(s.get("tool_args") or {}, ensure_ascii=False)[:90]
        lines.append(
            f"  [step {s.get('step_iteration', '?')}] "
            f"{s.get('tool_name', '?')} {args_short}  "
            f"{size}B  fp={fp}"
        )

    fingerprints = {_output_fingerprint(s) for s in same}
    diversity_hint = (
        "ALL identical fingerprints — definitely looping."
        if len(fingerprints) == 1
        else f"{len(fingerprints)} unique fingerprints across {len(same)} calls "
             f"({'high' if len(fingerprints) / len(same) > 0.7 else 'low'} variance)."
    )

    return f"""
## Productivity Audit (compare against your own recent calls)

Before filling `output_analysis.productivity`, honestly assess: did this call
yield new information, or did it repeat what you already saw?

Recent same-pattern tool calls (fp = sha256-truncated fingerprint of normalized
response body — same fp means functionally identical output):

{chr(10).join(lines)}

{diversity_hint}

Decision rules:
  - If 3+ recent same-pattern calls share the same fingerprint AND you have no
    new fact to cite in `what_was_new` → verdict MUST be `duplicate` or
    `no_progress`. Marking it `confirmation` is dishonest.
  - If the call hit 401/403/captcha/WAF → verdict is `blocked`.
  - If you can cite ONE specific new fact in `what_was_new` that is not already
    in your findings list → verdict is `new_info`.
  - If the output merely confirms a fact you already had → verdict is
    `confirmation` (acceptable for a single confirmation, not for repeats).

If your prior `productivity` claim was downgraded as inconsistent, the reason
appears below. Take it seriously — repeating the same dishonest claim wastes
budget.
"""
