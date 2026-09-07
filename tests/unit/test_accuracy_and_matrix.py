"""MAT-013 and MAT-016: what may be called correct, and what may be called supported."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tools.accuracy_differential import compare  # noqa: E402
from tools.update_support_matrix import build_entry  # noqa: E402
from validators.accuracy_validator import validate_accuracy_report  # noqa: E402
from validators.matrix_validator import validate_matrix_entry  # noqa: E402

ACCURACY_CONTRACT = yaml.safe_load(
    (ROOT / "tasks" / "mat-013-accuracy-differential" / "task.yaml").read_text(encoding="utf-8")
)
MATRIX_CONTRACT = yaml.safe_load(
    (ROOT / "tasks" / "mat-016-support-matrix" / "task.yaml").read_text(encoding="utf-8")
)
THRESHOLDS = (ACCURACY_CONTRACT.get("checks") or {}).get("thresholds") or {}

# The real 2026-09-07 differential: P800 server against transformers on CPU.
REPORT = {
    "status": "ACCURACY_PASS",
    "subject": "Qwen3-8B",
    "revision": "e962c91b" + "0" * 56,
    "reference": {"implementation": "transformers on CPU", "device": "cpu", "dtype": "float32"},
    "candidate": {"pod": "dongxinyu03-vllm-kunlun-dev-74557f86c6-tvnz8-0"},
    "metric": "top-1 next-token identity and top-5 set overlap",
    "threshold_source": "tasks/mat-013-accuracy-differential/task.yaml",
    "top1_agreement": "3/3",
    "cases": [
        {"prompt": "The capital of France is", "top1_candidate": " Paris", "top1_reference": " Paris",
         "top1_match": True, "top5_overlap": 5, "candidate_top5": [" Paris"], "reference_top5": [" Paris"]},
    ],
}
PLAN = {
    "revision": REPORT["revision"],
    "stack_commit": "3ced109af2510479e1b2eb846a8aca1fbdcbdf62",
    "parameters": [
        {"parameter": "dtype", "value": "bfloat16"},
        {"parameter": "tensor_parallel_size", "value": 1},
        {"parameter": "max_model_len", "value": 40960},
        {"parameter": "enforce_eager", "value": True},
    ],
}


class AccuracyDifferentialTest(unittest.TestCase):
    def test_the_real_qwen3_differential_passes(self):
        self.assertEqual(validate_accuracy_report(REPORT, THRESHOLDS), [])

    def test_a_disagreeing_top_token_cannot_pass(self):
        report = copy.deepcopy(REPORT)
        report["cases"][0].update(top1_candidate=" London", top1_match=False)
        errors = validate_accuracy_report(report, THRESHOLDS)
        self.assertTrue(any("top-1 disagreed" in error for error in errors), errors)

    def test_a_thin_top5_overlap_cannot_pass(self):
        report = copy.deepcopy(REPORT)
        report["cases"][0]["top5_overlap"] = 1
        errors = validate_accuracy_report(report, THRESHOLDS)
        self.assertTrue(any("top-5 overlap is below" in error for error in errors), errors)

    def test_a_recorded_failure_is_a_valid_report(self):
        """A fail must not be rejected as malformed; only a pass must be earned."""
        report = copy.deepcopy(REPORT)
        report["status"] = "ACCURACY_FAIL"
        report["cases"][0].update(top1_match=False, top5_overlap=1)
        self.assertEqual(validate_accuracy_report(report, THRESHOLDS), [])

    def test_a_reference_on_the_candidate_s_device_proves_nothing(self):
        report = copy.deepcopy(REPORT)
        report["reference"]["device"] = "xpu"
        report["candidate"]["device"] = "xpu"
        errors = validate_accuracy_report(report, THRESHOLDS)
        self.assertTrue(any("shares the candidate's device" in error for error in errors), errors)

    def test_comparison_uses_distributions_not_strings(self):
        candidate = [{"token": " Paris", "logprob": -0.1}, {"token": " a", "logprob": -3.0}]
        reference = [{"token": " Paris", "logprob": -0.2}, {"token": " the", "logprob": -2.5}]
        result = compare(candidate, reference, 20)
        self.assertTrue(result["top1_match"])
        self.assertEqual(result["top5_overlap"], 1)


class SupportMatrixTest(unittest.TestCase):
    def entry(self, deployment: str = "DEPLOYMENT_READY", accuracy: str = "ACCURACY_PASS") -> dict:
        return build_entry(
            "Qwen3-8B", "Kunlunxin-3-P800",
            {"state": deployment}, {"status": accuracy, "revision": PLAN["revision"], "cases": [1, 2, 3]},
            {"state": "BUDGET_ACCEPTABLE"}, PLAN,
        )

    def test_evidence_on_both_sides_earns_validated(self):
        entry = self.entry()
        self.assertEqual(entry["status"], "validated")
        self.assertEqual(validate_matrix_entry(entry, MATRIX_CONTRACT), [])

    def test_serving_without_accuracy_is_not_validated(self):
        """A server that answers is not a server that is right."""
        entry = self.entry(accuracy="")
        self.assertEqual(entry["status"], "runs_unverified_accuracy")
        self.assertEqual(validate_matrix_entry(entry, MATRIX_CONTRACT), [])

    def test_claiming_validated_without_accuracy_evidence_is_refused(self):
        entry = self.entry(accuracy="")
        entry["status"] = "validated"
        errors = validate_matrix_entry(entry, MATRIX_CONTRACT)
        self.assertTrue(any("ACCURACY_PASS" in error for error in errors), errors)

    def test_a_row_must_say_what_it_is_true_of(self):
        entry = self.entry()
        entry["stack_commit"] = None
        errors = validate_matrix_entry(entry, MATRIX_CONTRACT)
        self.assertTrue(any("only true of one stack_commit" in error for error in errors), errors)

    def test_a_conditional_result_must_carry_its_limitation(self):
        entry = self.entry()
        self.assertIn("enforce_eager=True", entry["conditions"])
        entry["limitations"] = []
        errors = validate_matrix_entry(entry, MATRIX_CONTRACT)
        self.assertTrue(any("eager execution" in error for error in errors), errors)

    def test_a_small_differential_is_recorded_as_a_limitation(self):
        self.assertTrue(any("not a dataset score" in text for text in self.entry()["limitations"]))


if __name__ == "__main__":
    unittest.main()
