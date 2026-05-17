"""Tests for the productivity-based loop detector.

Covers four layers:

1. Unit — every helper in productivity.py, with mock step dicts.
2. State integration — the new ProductivityVerdict + OutputAnalysisInline field.
3. Loop-detector behavior — does the orchestrator's "is this an unproductive
   streak?" decision match expectations across the failure modes that
   triggered the original XBEN-001-24 loop?
4. Regression — every legacy keyword failure case still trips.

Test fixtures construct step dicts that mirror what think_node persists onto
execution_trace, so the helpers receive realistic shapes (productivity at the
top level, plus the older nested fallback for safety).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

# Ensure agent/ is importable for the pydantic-dependent state tests below.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load productivity.py DIRECTLY (no package import) so the stdlib-only tests
# work without pydantic installed. The package __init__ pulls in state.py
# which requires pydantic; the productivity module itself does not.
_PROD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "orchestrator_helpers", "productivity.py"
)
_spec = importlib.util.spec_from_file_location("_prod_under_test", _PROD_PATH)
_prod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prod)

_normalize_args_pattern = _prod._normalize_args_pattern
_output_fingerprint = _prod._output_fingerprint
_read_productivity = _prod._read_productivity
audit_productivity_claim = _prod.audit_productivity_claim
build_productivity_audit_section = _prod.build_productivity_audit_section
downgrade_verdict_to_no_progress = _prod.downgrade_verdict_to_no_progress
is_unproductive = _prod.is_unproductive


def _make_step(*, tool="execute_curl", args=None, output="", success=True,
               productivity=None, step_iteration=1):
    """Build a step dict that mirrors what think_node persists."""
    step = {
        "step_id": "abc",
        "step_iteration": step_iteration,
        "iteration": step_iteration,
        "tool_name": tool,
        "tool_args": args or {},
        "tool_output": output,
        "success": success,
        "output_analysis": "",
        "actionable_findings": [],
    }
    if productivity is not None:
        step["productivity"] = productivity
    return step


def _verdict(verdict="new_info", gained=True, what="", repeat=False, why=""):
    return {
        "verdict": verdict,
        "new_information_gained": gained,
        "what_was_new": what,
        "should_repeat_similar_call": repeat,
        "rationale": why,
    }


class TestNormalizeArgsPattern(unittest.TestCase):
    """Different IDs at the same path must collapse to one pattern, but
    different paths or different tools must stay distinct."""

    def test_integer_ids_collapse(self):
        sig_a = _normalize_args_pattern("execute_curl", {"args": "GET /order/300500/receipt"})
        sig_b = _normalize_args_pattern("execute_curl", {"args": "GET /order/300600/receipt"})
        self.assertEqual(sig_a, sig_b, "Two URLs differing only in <int> must share a pattern")

    def test_different_paths_distinct(self):
        sig_a = _normalize_args_pattern("execute_curl", {"args": "GET /order/300500/receipt"})
        sig_b = _normalize_args_pattern("execute_curl", {"args": "GET /profile"})
        self.assertNotEqual(sig_a, sig_b)

    def test_different_tools_distinct(self):
        sig_a = _normalize_args_pattern("execute_curl", {"args": "GET /x"})
        sig_b = _normalize_args_pattern("execute_nmap", {"args": "-sV /x"})
        self.assertNotEqual(sig_a, sig_b)

    def test_hex_tokens_collapse(self):
        sig_a = _normalize_args_pattern("execute_curl", {"args": "GET /api/a1b2c3d4e5f6"})
        sig_b = _normalize_args_pattern("execute_curl", {"args": "GET /api/deadbeef1234"})
        self.assertEqual(sig_a, sig_b, "Long hex tokens must collapse to <hex>")

    def test_ips_collapse(self):
        sig_a = _normalize_args_pattern("execute_curl", {"args": "http://192.168.1.10/"})
        sig_b = _normalize_args_pattern("execute_curl", {"args": "http://10.0.0.5/"})
        self.assertEqual(sig_a, sig_b)

    def test_handles_none_args(self):
        sig = _normalize_args_pattern("execute_curl", None)
        self.assertIn("execute_curl", sig)

    def test_handles_none_tool(self):
        sig = _normalize_args_pattern(None, {"x": 1})
        self.assertIn("?", sig)


class TestOutputFingerprint(unittest.TestCase):
    """Same logical content must produce the same fingerprint, even with
    trivial diffs like whitespace, timestamps, or UUIDs."""

    def test_identical_outputs_same_fingerprint(self):
        a = _make_step(output="<html><body>Empty receipt</body></html>")
        b = _make_step(output="<html><body>Empty receipt</body></html>")
        self.assertEqual(_output_fingerprint(a), _output_fingerprint(b))

    def test_whitespace_does_not_change_fingerprint(self):
        a = _make_step(output="<html><body>Empty\nreceipt</body></html>")
        b = _make_step(output="<html><body>Empty    receipt</body></html>")
        self.assertEqual(_output_fingerprint(a), _output_fingerprint(b))

    def test_timestamps_normalized(self):
        a = _make_step(output="<p>2026-05-17T08:14:01.234Z OK</p>")
        b = _make_step(output="<p>2026-05-17T09:42:11.000Z OK</p>")
        self.assertEqual(_output_fingerprint(a), _output_fingerprint(b),
                         "ISO timestamps must collapse to <ts>")

    def test_uuids_normalized(self):
        a = _make_step(output="request_id=abcd1234-aaaa-bbbb-cccc-deadbeef1234 done")
        b = _make_step(output="request_id=99887766-5544-3322-1100-ffeeddccbbaa done")
        self.assertEqual(_output_fingerprint(a), _output_fingerprint(b))

    def test_different_content_different_fingerprint(self):
        a = _make_step(output="Empty receipt for missing order")
        b = _make_step(output="Order 300123 found, $50,000 trade in AAPL")
        self.assertNotEqual(_output_fingerprint(a), _output_fingerprint(b))

    def test_empty_output_stable(self):
        a = _make_step(output="")
        b = _make_step(output="")
        self.assertEqual(_output_fingerprint(a), _output_fingerprint(b))

    def test_fingerprint_length_is_8(self):
        fp = _output_fingerprint(_make_step(output="anything"))
        self.assertEqual(len(fp), 8)


class TestReadProductivity(unittest.TestCase):
    """The helper must accept both shapes the codebase uses:
    top-level step['productivity'] (preferred, used by think_node) and
    nested step['output_analysis']['productivity'] (forward-compat)."""

    def test_top_level(self):
        step = _make_step(productivity=_verdict(verdict="no_progress", gained=False))
        p = _read_productivity(step)
        self.assertEqual(p["verdict"], "no_progress")

    def test_nested(self):
        step = _make_step()
        step["output_analysis"] = {"productivity": _verdict(verdict="duplicate")}
        p = _read_productivity(step)
        self.assertEqual(p["verdict"], "duplicate")

    def test_top_level_wins(self):
        step = _make_step(productivity=_verdict(verdict="new_info"))
        step["output_analysis"] = {"productivity": _verdict(verdict="duplicate")}
        p = _read_productivity(step)
        self.assertEqual(p["verdict"], "new_info",
                         "When both shapes are present, top-level must take priority")

    def test_missing_returns_empty(self):
        step = _make_step()
        self.assertEqual(_read_productivity(step), {})

    def test_none_step(self):
        self.assertEqual(_read_productivity(None), {})

    def test_output_analysis_is_string(self):
        """Real-world: think_node stores output_analysis as a string
        (the interpretation). Must not crash."""
        step = _make_step()
        step["output_analysis"] = "some interpretation text"
        # No productivity anywhere — must return {}
        self.assertEqual(_read_productivity(step), {})


class TestIsUnproductive(unittest.TestCase):
    """The boolean dispatcher consumed by the loop counter."""

    def test_new_info_is_productive(self):
        step = _make_step(productivity=_verdict(verdict="new_info", gained=True))
        self.assertFalse(is_unproductive(step))

    def test_confirmation_is_productive(self):
        step = _make_step(productivity=_verdict(verdict="confirmation", gained=True))
        self.assertFalse(is_unproductive(step),
                         "confirmation is acceptable; only no_progress/duplicate/blocked count")

    def test_no_progress_is_unproductive(self):
        step = _make_step(productivity=_verdict(verdict="no_progress", gained=False))
        self.assertTrue(is_unproductive(step))

    def test_duplicate_is_unproductive(self):
        step = _make_step(productivity=_verdict(verdict="duplicate", gained=False))
        self.assertTrue(is_unproductive(step))

    def test_blocked_is_unproductive(self):
        step = _make_step(productivity=_verdict(verdict="blocked", gained=False))
        self.assertTrue(is_unproductive(step))

    def test_gained_false_overrides_optimistic_verdict(self):
        """If the LLM claims 'new_info' but flags gained=False, treat as unproductive.
        Defends against schema-confused responses."""
        step = _make_step(productivity=_verdict(verdict="new_info", gained=False))
        self.assertTrue(is_unproductive(step))

    def test_missing_productivity_is_productive_by_default(self):
        """No verdict field means we fall back to keyword detection (legacy
        behavior is preserved). is_unproductive itself returns False."""
        step = _make_step()
        self.assertFalse(is_unproductive(step))


class TestAuditProductivityClaim(unittest.TestCase):
    """The honesty cross-check that catches optimistic LLM claims."""

    def test_no_productivity_returns_none(self):
        self.assertIsNone(audit_productivity_claim({}, {}, [], False))

    def test_honest_new_info_passes(self):
        result = audit_productivity_claim(
            productivity=_verdict(verdict="new_info", gained=True),
            extracted_info={"ports": [80, 443]},
            actionable_findings=[],
            findings_grew=False,
        )
        self.assertIsNone(result, "extracted_info populated → claim is honest")

    def test_dishonest_claim_caught(self):
        result = audit_productivity_claim(
            productivity=_verdict(verdict="new_info", gained=True),
            extracted_info={},
            actionable_findings=[],
            findings_grew=False,
        )
        self.assertIsNotNone(result, "Claimed new info but nothing grew")
        self.assertIn("new_information_gained=true", result)

    def test_findings_growth_alone_is_enough(self):
        result = audit_productivity_claim(
            productivity=_verdict(verdict="new_info", gained=True),
            extracted_info={},
            actionable_findings=[],
            findings_grew=True,
        )
        self.assertIsNone(result)

    def test_actionable_findings_alone_is_enough(self):
        result = audit_productivity_claim(
            productivity=_verdict(verdict="new_info", gained=True),
            extracted_info={},
            actionable_findings=["explore /admin"],
            findings_grew=False,
        )
        self.assertIsNone(result)

    def test_no_progress_verdict_never_flagged(self):
        """An honest 'no_progress' claim with no growth must NOT be flagged
        as a discrepancy — it's already a self-admission."""
        result = audit_productivity_claim(
            productivity=_verdict(verdict="no_progress", gained=False),
            extracted_info={},
            actionable_findings=[],
            findings_grew=False,
        )
        self.assertIsNone(result)

    def test_extracted_info_with_only_primary_target_does_not_save(self):
        """primary_target is required for every iteration; it does not count
        as 'new info' on its own."""
        result = audit_productivity_claim(
            productivity=_verdict(verdict="new_info", gained=True),
            extracted_info={"primary_target": "host"},
            actionable_findings=[],
            findings_grew=False,
        )
        self.assertIsNotNone(result,
            "primary_target alone is required boilerplate, not new info")


class TestDowngradeVerdict(unittest.TestCase):
    """Verdict downgrade for dishonest claims."""

    def test_downgrades_verdict(self):
        v = _verdict(verdict="new_info", gained=True)
        out = downgrade_verdict_to_no_progress(v, "test reason")
        self.assertEqual(out["verdict"], "no_progress")
        self.assertFalse(out["new_information_gained"])
        self.assertEqual(out["_original_verdict"], "new_info")
        self.assertEqual(out["_downgrade_reason"], "test reason")

    def test_preserves_other_fields(self):
        v = _verdict(verdict="new_info", gained=True, what="found admin", why="cited evidence")
        out = downgrade_verdict_to_no_progress(v, "test reason")
        self.assertEqual(out["what_was_new"], "found admin")
        self.assertEqual(out["rationale"], "cited evidence")

    def test_handles_empty_input(self):
        out = downgrade_verdict_to_no_progress({}, "missing field")
        self.assertEqual(out["verdict"], "no_progress")
        self.assertFalse(out["new_information_gained"])
        self.assertEqual(out["_downgrade_reason"], "missing field")

    def test_does_not_mutate_input(self):
        v = _verdict(verdict="new_info", gained=True)
        _ = downgrade_verdict_to_no_progress(v, "x")
        self.assertEqual(v["verdict"], "new_info", "Input must be left untouched")


class TestBuildProductivityAuditSection(unittest.TestCase):
    """The prompt block that shows the model its own recent fingerprints."""

    def test_empty_trace_no_section(self):
        self.assertEqual(build_productivity_audit_section([]), "")

    def test_fewer_than_three_same_pattern_no_section(self):
        trace = [
            _make_step(args={"args": "GET /order/300500/receipt"}, output="empty"),
            _make_step(args={"args": "GET /order/300600/receipt"}, output="empty"),
        ]
        self.assertEqual(build_productivity_audit_section(trace), "")

    def test_three_same_pattern_triggers(self):
        trace = [
            _make_step(args={"args": "GET /order/300500/receipt"}, output="empty"),
            _make_step(args={"args": "GET /order/300600/receipt"}, output="empty"),
            _make_step(args={"args": "GET /order/300700/receipt"}, output="empty"),
        ]
        section = build_productivity_audit_section(trace)
        self.assertIn("Productivity Audit", section)
        self.assertIn("fp=", section, "Must show fingerprints")

    def test_diversity_hint_when_all_identical(self):
        trace = [
            _make_step(args={"args": "GET /order/300500/receipt"}, output="empty receipt"),
            _make_step(args={"args": "GET /order/300600/receipt"}, output="empty receipt"),
            _make_step(args={"args": "GET /order/300700/receipt"}, output="empty receipt"),
            _make_step(args={"args": "GET /order/300800/receipt"}, output="empty receipt"),
        ]
        section = build_productivity_audit_section(trace)
        self.assertIn("ALL identical", section)

    def test_diversity_hint_when_varied(self):
        trace = [
            _make_step(args={"args": "GET /order/1/receipt"}, output="result A"),
            _make_step(args={"args": "GET /order/2/receipt"}, output="result B"),
            _make_step(args={"args": "GET /order/3/receipt"}, output="result C"),
        ]
        section = build_productivity_audit_section(trace)
        self.assertIn("unique fingerprints", section)
        self.assertNotIn("ALL identical", section)

    def test_picks_most_repeated_pattern_when_no_current(self):
        """With no current step provided, the helper must surface whichever
        pattern is repeating the most in the recent window."""
        trace = [
            _make_step(args={"args": "GET /profile"}, output="profile"),
            _make_step(args={"args": "GET /order/1/receipt"}, output="empty"),
            _make_step(args={"args": "GET /order/2/receipt"}, output="empty"),
            _make_step(args={"args": "GET /order/3/receipt"}, output="empty"),
        ]
        section = build_productivity_audit_section(trace)
        self.assertIn("/order/", section)
        self.assertNotIn("/profile", section.split("Productivity Audit")[1])

    def test_filters_to_current_pattern(self):
        trace = [
            _make_step(args={"args": "GET /profile"}),
            _make_step(args={"args": "GET /order/1/receipt"}),
            _make_step(args={"args": "GET /order/2/receipt"}),
            _make_step(args={"args": "GET /order/3/receipt"}),
        ]
        section = build_productivity_audit_section(
            trace,
            current_tool_name="execute_curl",
            current_tool_args={"args": "GET /order/4/receipt"},
        )
        # Only the order-receipt pattern should appear in the listing block.
        listing = section.split("Recent same-pattern")[1] if "Recent same-pattern" in section else section
        self.assertIn("/order/", listing)
        self.assertNotIn("/profile", listing)

    def test_includes_decision_rules(self):
        trace = [
            _make_step(args={"args": f"GET /x/{i}/y"}, output="same") for i in range(4)
        ]
        section = build_productivity_audit_section(trace)
        self.assertIn("duplicate", section)
        self.assertIn("blocked", section)
        self.assertIn("confirmation", section)


# ---------------------------------------------------------------------------
# Layer 2: state-model smoke (requires pydantic; gracefully skipped if not).
# ---------------------------------------------------------------------------

try:
    from state import OutputAnalysisInline, ProductivityVerdict  # type: ignore
    _HAS_PYDANTIC = True
except Exception:
    _HAS_PYDANTIC = False


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed in this env")
class TestProductivitySchema(unittest.TestCase):
    """The new field on OutputAnalysisInline must accept valid verdicts,
    reject invalid ones, and round-trip cleanly through model_dump."""

    def test_default_construction(self):
        p = ProductivityVerdict()
        self.assertEqual(p.verdict, "new_info")
        self.assertTrue(p.new_information_gained)

    def test_each_verdict_value_accepted(self):
        for v in ("new_info", "confirmation", "no_progress", "blocked", "duplicate"):
            p = ProductivityVerdict(verdict=v, new_information_gained=False)
            self.assertEqual(p.verdict, v)

    def test_invalid_verdict_rejected(self):
        with self.assertRaises(Exception):
            ProductivityVerdict(verdict="bogus")

    def test_round_trip_through_model_dump(self):
        p = ProductivityVerdict(
            verdict="duplicate", new_information_gained=False,
            what_was_new="", should_repeat_similar_call=False,
            rationale="same fingerprint as last 3",
        )
        d = p.model_dump()
        self.assertEqual(d["verdict"], "duplicate")
        self.assertEqual(d["rationale"], "same fingerprint as last 3")
        p2 = ProductivityVerdict(**d)
        self.assertEqual(p2.verdict, "duplicate")

    def test_output_analysis_inline_has_productivity(self):
        oa = OutputAnalysisInline()
        self.assertIsInstance(oa.productivity, ProductivityVerdict)
        self.assertEqual(oa.productivity.verdict, "new_info")

    def test_output_analysis_inline_accepts_explicit_productivity(self):
        oa = OutputAnalysisInline(
            interpretation="probed receipt endpoint",
            productivity=ProductivityVerdict(
                verdict="no_progress", new_information_gained=False,
                what_was_new="", should_repeat_similar_call=False,
                rationale="empty receipt template",
            ),
        )
        self.assertEqual(oa.productivity.verdict, "no_progress")
        self.assertFalse(oa.productivity.new_information_gained)

    def test_backward_compat_missing_productivity_uses_default(self):
        """Old LLM outputs without productivity must still parse."""
        oa = OutputAnalysisInline.model_validate({
            "interpretation": "x",
            "actionable_findings": [],
            "recommended_next_steps": [],
            "exploit_succeeded": False,
        })
        self.assertEqual(oa.productivity.verdict, "new_info",
                         "Default verdict applied when LLM omits the field")


# ---------------------------------------------------------------------------
# Layer 3: loop-detector simulation.
# Reproduces the think_node Place-1 logic against a synthetic execution_trace.
# Verifies the new detector catches the XBEN-001-24 failure mode that the
# legacy keyword detector missed.
# ---------------------------------------------------------------------------

def _simulate_loop_detector(execution_trace, window=6, threshold=3):
    """Mirror of think_node Place-1 logic. Returns True if the orchestrator
    would fire Deep Think on the unproductive-streak condition."""
    if len(execution_trace) < threshold:
        return False
    count = 0
    for step in execution_trace[-window:]:
        out = ((step.get("tool_output") or "")[:500]).lower()
        kw_fail = (
            not step.get("success", True)
            or "failed" in out
            or "error" in out
            or "exploit completed, but no session" in out
        )
        if kw_fail or is_unproductive(step):
            count += 1
    return count >= threshold


class TestLoopDetectorEndToEnd(unittest.TestCase):
    """Integration: simulate think_node's logic against canned traces."""

    def test_xben_001_loop_now_caught(self):
        """The exact failure mode we observed: 200 OK with empty-receipt body
        repeated, LLM verdicts honestly mark them as no_progress."""
        trace = [
            _make_step(
                args={"args": f"GET /order/{i}/receipt"},
                output="HTTP 200 ... <code></code> ...",
                success=True,
                productivity=_verdict(verdict="no_progress", gained=False),
                step_iteration=i,
            )
            for i in range(300100, 300600, 100)
        ]
        self.assertTrue(_simulate_loop_detector(trace),
            "Empty-receipt streak with no_progress verdicts must trip detector")

    def test_xben_001_loop_legacy_detector_missed(self):
        """The same streak — but WITHOUT the productivity field. This simulates
        the world before this patch, and the detector should NOT fire (proving
        the legacy keyword check was blind here)."""
        trace = [
            _make_step(
                args={"args": f"GET /order/{i}/receipt"},
                output="HTTP 200 ... <code></code> ...",
                success=True,
                # No productivity field — pure legacy keyword path.
                step_iteration=i,
            )
            for i in range(300100, 300600, 100)
        ]
        self.assertFalse(_simulate_loop_detector(trace),
            "Legacy-only detector must NOT fire on this (proves the bug)")

    def test_productive_run_does_not_trigger(self):
        """A healthy run of varied productive steps must not trigger."""
        trace = [
            _make_step(
                args={"args": f"GET /endpoint/{i}"},
                output=f"unique content {i}",
                success=True,
                productivity=_verdict(verdict="new_info", gained=True),
                step_iteration=i,
            )
            for i in range(5)
        ]
        self.assertFalse(_simulate_loop_detector(trace))

    def test_mixed_run_below_threshold_does_not_trigger(self):
        """Two unproductive + four productive in a 6-window → below threshold."""
        trace = [
            _make_step(productivity=_verdict(verdict="no_progress", gained=False)),
            _make_step(productivity=_verdict(verdict="new_info", gained=True)),
            _make_step(productivity=_verdict(verdict="new_info", gained=True)),
            _make_step(productivity=_verdict(verdict="no_progress", gained=False)),
            _make_step(productivity=_verdict(verdict="new_info", gained=True)),
            _make_step(productivity=_verdict(verdict="new_info", gained=True)),
        ]
        self.assertFalse(_simulate_loop_detector(trace))

    def test_mixed_run_at_threshold_triggers(self):
        """Three unproductive in a six-window — at threshold, must fire."""
        trace = [
            _make_step(productivity=_verdict(verdict="no_progress", gained=False)),
            _make_step(productivity=_verdict(verdict="new_info", gained=True)),
            _make_step(productivity=_verdict(verdict="duplicate", gained=False)),
            _make_step(productivity=_verdict(verdict="blocked", gained=False)),
            _make_step(productivity=_verdict(verdict="new_info", gained=True)),
            _make_step(productivity=_verdict(verdict="confirmation", gained=True)),
        ]
        self.assertTrue(_simulate_loop_detector(trace))

    def test_blocked_streak_triggers(self):
        """WAF returning 403s — keyword 'forbidden' may or may not appear, but
        the LLM's 'blocked' verdict must trip the detector regardless."""
        trace = [
            _make_step(
                output="<html>403 Forbidden</html>",
                productivity=_verdict(verdict="blocked", gained=False),
            )
            for _ in range(4)
        ]
        self.assertTrue(_simulate_loop_detector(trace))

    def test_window_respected(self):
        """An old unproductive streak outside the window must not count."""
        trace = (
            [_make_step(productivity=_verdict(verdict="no_progress", gained=False))] * 3
            + [_make_step(productivity=_verdict(verdict="new_info", gained=True))] * 6
        )
        # Last 6 are all productive, so detector must not fire.
        self.assertFalse(_simulate_loop_detector(trace))


# ---------------------------------------------------------------------------
# Layer 4: regression — every legacy keyword failure still fires.
# Guarantees we did not weaken any existing case.
# ---------------------------------------------------------------------------

class TestLegacyKeywordRegressions(unittest.TestCase):
    """Every keyword case the old detector caught must STILL trip the new one,
    because we OR'd the keyword check with is_unproductive."""

    def test_success_false_streak_triggers(self):
        trace = [
            _make_step(success=False, output="connection reset")
            for _ in range(3)
        ]
        self.assertTrue(_simulate_loop_detector(trace))

    def test_failed_keyword_streak_triggers(self):
        trace = [
            _make_step(success=True, output="[-] Failed to bind socket")
            for _ in range(3)
        ]
        self.assertTrue(_simulate_loop_detector(trace))

    def test_error_keyword_streak_triggers(self):
        trace = [
            _make_step(success=True, output="HTTP 500 Internal Error")
            for _ in range(3)
        ]
        self.assertTrue(_simulate_loop_detector(trace))

    def test_metasploit_no_session_phrase_triggers(self):
        trace = [
            _make_step(
                tool="metasploit_console", success=True,
                output="[*] Exploit completed, but no session was created.",
            )
            for _ in range(3)
        ]
        self.assertTrue(_simulate_loop_detector(trace))

    def test_legacy_and_new_compose(self):
        """Two legacy failures + one LLM-flagged unproductive = 3 → trips."""
        trace = [
            _make_step(success=False, output="error"),
            _make_step(success=True, output="HTTP 200 OK", productivity=_verdict(verdict="duplicate", gained=False)),
            _make_step(success=True, output="failed to read"),
        ]
        self.assertTrue(_simulate_loop_detector(trace))


# ---------------------------------------------------------------------------
# Layer 5: smoke — productivity helpers under odd inputs must not crash.
# ---------------------------------------------------------------------------

class TestSettingsDefaults(unittest.TestCase):
    """Verify the two new settings are declared in DEFAULT_AGENT_SETTINGS so
    the orchestrator picks up sensible defaults if a project has not been
    upgraded. Reads project_settings.py as text to avoid needing pydantic."""

    def test_settings_declared(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "project_settings.py"
        )
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("'PRODUCTIVITY_AUDIT_WINDOW'", content)
        self.assertIn("'UNPRODUCTIVE_STREAK_THRESHOLD'", content)

    def test_defaults_are_sensible(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "project_settings.py"
        )
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Window default 6, threshold default 3 — explicit so behavior is locked.
        self.assertIn("'PRODUCTIVITY_AUDIT_WINDOW': 6", content)
        self.assertIn("'UNPRODUCTIVE_STREAK_THRESHOLD': 3", content)


class TestThinkNodeWiring(unittest.TestCase):
    """Verify the patched think_node imports and call sites are syntactically
    integrated. Reads the file as text — catches typos, missed imports, and
    accidental removal of legacy logic. Cheap and high-value."""

    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "orchestrator_helpers",
            "nodes", "think_node.py",
        )
        with open(path, "r", encoding="utf-8") as f:
            self.content = f.read()

    def test_productivity_imports_present(self):
        self.assertIn("from orchestrator_helpers.productivity import", self.content)
        self.assertIn("is_unproductive", self.content)
        self.assertIn("audit_productivity_claim", self.content)
        self.assertIn("build_productivity_audit_section", self.content)
        self.assertIn("downgrade_verdict_to_no_progress", self.content)

    def test_legacy_keyword_check_preserved_place_1(self):
        """The keyword path must stay so legacy failure cases still trigger."""
        self.assertIn('"failed" in _out', self.content)
        self.assertIn('"error" in _out', self.content)
        self.assertIn('"exploit completed, but no session" in _out', self.content)

    def test_legacy_keyword_check_preserved_place_2(self):
        # Place 2 uses output_lower (different variable name)
        self.assertIn('"failed" in output_lower', self.content)
        self.assertIn('"error" in output_lower', self.content)

    def test_new_check_or_composed_with_legacy(self):
        """The new is_unproductive call must be OR-ed with the legacy check,
        not replacing it."""
        # Look for the OR pattern in either Place 1 or Place 2.
        self.assertIn("is_unproductive(_step)", self.content)
        self.assertIn("is_unproductive(step)", self.content)

    def test_audit_section_injected(self):
        self.assertIn("build_productivity_audit_section", self.content)
        self.assertIn("_last_productivity_discrepancy", self.content)

    def test_productivity_persisted_on_step(self):
        """Each exec_step (wave + single) must persist the productivity dict."""
        self.assertIn('"productivity": dict(_wave_productivity)', self.content)
        self.assertIn('pending_step["productivity"]', self.content)

    def test_settings_referenced(self):
        self.assertIn("PRODUCTIVITY_AUDIT_WINDOW", self.content)
        self.assertIn("UNPRODUCTIVE_STREAK_THRESHOLD", self.content)


class TestFireteamMemberWiring(unittest.TestCase):
    """Same syntactic-integration check for the fireteam member node."""

    def setUp(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "orchestrator_helpers",
            "nodes", "fireteam_member_think_node.py",
        )
        with open(path, "r", encoding="utf-8") as f:
            self.content = f.read()

    def test_productivity_imports_present(self):
        self.assertIn("from orchestrator_helpers.productivity import", self.content)
        self.assertIn("audit_productivity_claim", self.content)
        self.assertIn("build_productivity_audit_section", self.content)

    def test_audit_section_injection_present(self):
        self.assertIn("build_productivity_audit_section", self.content)
        self.assertIn("_last_productivity_discrepancy", self.content)

    def test_productivity_persisted_on_completed_step(self):
        self.assertIn('completed_step["productivity"]', self.content)

    def test_productivity_persisted_on_wave_steps(self):
        self.assertIn('"productivity": dict(_wave_productivity)', self.content)

    def test_existing_stall_counters_preserved(self):
        """We added productivity as a SECOND layer on top of the existing
        fireteam stall counter — must not have removed the original."""
        self.assertIn("iterations_since_new_finding", self.content)
        self.assertIn("fallback_uses_this_run", self.content)


class TestPromptSchemaSync(unittest.TestCase):
    """Verify the prompt JSON-schema example documents the new productivity
    field, so the LLM knows it must emit it."""

    def setUp(self):
        base_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "base.py"
        )
        with open(base_path, "r", encoding="utf-8") as f:
            self.base = f.read()
        member_path = os.path.join(
            os.path.dirname(__file__), "..", "orchestrator_helpers",
            "nodes", "fireteam_member_think_node.py",
        )
        with open(member_path, "r", encoding="utf-8") as f:
            self.member = f.read()

    def test_root_single_section_has_productivity(self):
        # Find the PENDING_OUTPUT_ANALYSIS_SECTION block.
        section = self.base.split("PENDING_OUTPUT_ANALYSIS_SECTION = ")[1].split(
            "PENDING_PLAN_OUTPUTS_SECTION"
        )[0]
        self.assertIn('"productivity"', section)
        self.assertIn("new_info", section)
        self.assertIn("no_progress", section)
        self.assertIn("duplicate", section)
        self.assertIn("blocked", section)

    def test_root_plan_section_has_productivity(self):
        section = self.base.split("PENDING_PLAN_OUTPUTS_SECTION = ")[1]
        self.assertIn('"productivity"', section)
        self.assertIn("no_progress", section)

    def test_member_single_section_has_productivity(self):
        section = self.member.split("_MEMBER_PENDING_OUTPUT_SECTION = ")[1].split(
            "_MEMBER_PENDING_PLAN_OUTPUTS_SECTION"
        )[0]
        self.assertIn('"productivity"', section)
        self.assertIn("no_progress", section)

    def test_member_plan_section_has_productivity(self):
        section = self.member.split("_MEMBER_PENDING_PLAN_OUTPUTS_SECTION = ")[1]
        self.assertIn('"productivity"', section)

    def test_all_five_verdicts_documented(self):
        """The model must see all 5 verdict values somewhere in the prompt."""
        all_text = self.base + self.member
        for verdict in ("new_info", "confirmation", "no_progress", "blocked", "duplicate"):
            self.assertIn(verdict, all_text, f"Missing verdict {verdict!r} from prompts")


class TestDowngradeIdempotence(unittest.TestCase):
    """Calling downgrade twice on the same dict must remain coherent: the
    second call sees an already-no_progress verdict and behaves sensibly."""

    def test_double_downgrade_preserves_no_progress(self):
        v = _verdict(verdict="new_info", gained=True)
        once = downgrade_verdict_to_no_progress(v, "first")
        twice = downgrade_verdict_to_no_progress(once, "second")
        self.assertEqual(twice["verdict"], "no_progress")
        self.assertFalse(twice["new_information_gained"])

    def test_double_downgrade_keeps_latest_reason(self):
        v = _verdict(verdict="new_info", gained=True)
        once = downgrade_verdict_to_no_progress(v, "first")
        twice = downgrade_verdict_to_no_progress(once, "second")
        self.assertEqual(twice["_downgrade_reason"], "second")


class TestSmokeRobustness(unittest.TestCase):
    """Defensive checks: every helper must handle missing/None/odd inputs
    without crashing. The orchestrator may pass partial state during
    interrupted runs."""

    def test_is_unproductive_handles_missing_keys(self):
        self.assertFalse(is_unproductive({}))
        self.assertFalse(is_unproductive({"foo": "bar"}))

    def test_normalize_handles_none(self):
        sig = _normalize_args_pattern(None, None)
        self.assertIsInstance(sig, str)
        self.assertGreater(len(sig), 0)

    def test_fingerprint_handles_none_output(self):
        step = _make_step(output=None)
        step["tool_output"] = None
        fp = _output_fingerprint(step)
        self.assertEqual(len(fp), 8)

    def test_fingerprint_handles_very_long_output(self):
        step = _make_step(output="X" * 100000)
        fp = _output_fingerprint(step)
        self.assertEqual(len(fp), 8)

    def test_audit_handles_none_inputs(self):
        self.assertIsNone(audit_productivity_claim(None, None, None, False))

    def test_audit_handles_partial_productivity(self):
        # Productivity dict missing fields — must not crash, must not falsely
        # flag (verdict defaults to None, new claim defaults to False).
        result = audit_productivity_claim(
            productivity={"verdict": "no_progress"},
            extracted_info={},
            actionable_findings=[],
            findings_grew=False,
        )
        self.assertIsNone(result)

    def test_build_section_handles_step_without_args(self):
        trace = [
            {"tool_name": "execute_curl", "tool_output": "x"}
            for _ in range(3)
        ]
        # Should not crash even though args/iteration keys are missing.
        section = build_productivity_audit_section(trace)
        self.assertIsInstance(section, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
