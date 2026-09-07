"""MAT-003: keeping a static match from reading as a support statement."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validators.capability_validator import validate_capability_match  # noqa: E402

CONTRACT_PATH = ROOT / "tasks" / "mat-003-capability-match" / "task.yaml"

# The real 2026-09-07 Qwen3-8B match: every axis satisfied, and the model still
# does not serve on P800.
MATCH = {
    "state": "MATCH_READY",
    "runtime_verified": False,
    "matched_in": "dongxinyu03-kdp001-qwen3-8b-7c969df894-hsz7d-0",
    "provided": {"quantization_methods": ["compressed-tensors"], "moe": "MODULE"},
    "axes": [
        {"axis": "attention", "required": "gqa", "evidence": "MODULE", "verdict": "PROVIDED_MODULE_ONLY"},
        {"axis": "quantization", "required": None, "evidence": "N/A", "verdict": "NOT_REQUIRED"},
        {"axis": "moe", "required": None, "evidence": "N/A", "verdict": "NOT_REQUIRED"},
        {"axis": "multimodal", "required": False, "evidence": "N/A", "verdict": "NOT_REQUIRED"},
        {"axis": "speculative_decode", "required": None, "evidence": "N/A", "verdict": "NOT_REQUIRED"},
    ],
    "verdict": "MATCHED",
    "blocking_axes": [],
    "unknown_axes": [],
}


class CapabilityValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        import yaml

        self.contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.match = copy.deepcopy(MATCH)

    def check(self) -> list[str]:
        return validate_capability_match(self.match, self.contract)

    def test_the_real_qwen3_match_passes(self):
        self.assertEqual(self.check(), [])

    def test_a_match_may_not_claim_a_runtime_result(self):
        self.match["runtime_verified"] = True
        errors = self.check()
        self.assertTrue(any("cannot" in error and "runtime" in error for error in errors), errors)

    def test_registry_evidence_is_required_for_provided(self):
        """A module on disk is weaker evidence and must say so."""
        self.match["axes"][0]["verdict"] = "PROVIDED"
        errors = self.check()
        self.assertTrue(any("requires registry evidence" in error for error in errors), errors)

    def test_an_unknown_axis_cannot_hide_behind_matched(self):
        self.match["axes"][3].update(required=True, verdict="UNKNOWN")
        self.match["unknown_axes"] = ["multimodal"]
        errors = self.check()
        self.assertTrue(any("hide an open question" in error for error in errors), errors)

    def test_a_blocking_axis_forces_mismatch(self):
        self.match["axes"][0].update(evidence="ABSENT", verdict="NOT_PROVIDED")
        self.match["blocking_axes"] = ["attention"]
        errors = self.check()
        self.assertTrue(any("overall verdict is 'MATCHED'" in error for error in errors), errors)

    def test_a_summary_that_disagrees_with_its_axes_is_rejected(self):
        self.match["blocking_axes"] = ["quantization"]
        self.assertIn("blocking_axes disagrees with the per-axis verdicts", self.check())

    def test_not_provided_needs_absent_evidence(self):
        self.match["axes"][0].update(verdict="NOT_PROVIDED")
        self.match["blocking_axes"] = ["attention"]
        self.match["verdict"] = "MISMATCH"
        errors = self.check()
        self.assertTrue(any("backed by absent evidence" in error for error in errors), errors)

    def test_the_inspected_pod_must_be_recorded(self):
        self.match.pop("matched_in")
        errors = self.check()
        self.assertTrue(any("matched_in" in error for error in errors), errors)

    def test_the_provided_inventory_must_be_recorded(self):
        self.match.pop("provided")
        errors = self.check()
        self.assertTrue(any("provided-capability inventory" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
