import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validators.correctness_validator import (  # noqa: E402
    validate_end_to_end,
    validate_kernel_grade,
    validate_long_context,
)


def kernel_report():
    return {
        "state": "PASS",
        "relative_l2": 0.001,
        "max_abs_error": 0.02,
        "max_relative_l2": 0.01,
        "shape_candidate": [2, 2],
        "shape_reference": [2, 2],
        "reference_provenance": {
            "implementation": "torch",
            "device": "cpu",
            "dtype": "float32",
            "independently_written": True,
        },
        "control_discriminates": True,
    }


class CorrectnessValidatorTest(unittest.TestCase):
    def test_kernel_grade_accepts_independent_controlled_pass(self):
        self.assertEqual(validate_kernel_grade(kernel_report()), [])

    def test_kernel_grade_rejects_missing_provenance_and_shape(self):
        report = kernel_report()
        report["reference_provenance"].pop("dtype")
        report["shape_reference"] = [4]
        errors = validate_kernel_grade(report)
        self.assertTrue(any("dtype" in error for error in errors))
        self.assertTrue(any("shapes" in error for error in errors))

    def test_end_to_end_requires_integrated_case_evidence(self):
        report = {
            "state": "ACCURACY_PASS",
            "composition": "integrated_serving_path",
            "candidate": {"device": "P800"},
            "reference_provenance": {"device": "cpu"},
            "cases": [{"prompt": "p0", "evidence": ["case-0.json"]}],
        }
        self.assertEqual(validate_end_to_end(report), [])
        report["cases"][0].pop("evidence")
        self.assertTrue(validate_end_to_end(report))

    def test_long_context_requires_sparse_path_and_crossed_boundary(self):
        report = {
            "state": "PASS",
            "path": "sparse",
            "geometry": {"context_len": 513, "block_size": 64, "topk": 4},
            "selected_blocks": [0, 3, 7],
            "relative_l2": 0.001,
            "max_relative_l2": 0.01,
        }
        self.assertEqual(validate_long_context(report), [])
        report["path"] = "dense"
        self.assertTrue(any("sparse path" in e for e in validate_long_context(report)))


if __name__ == "__main__":
    unittest.main()
