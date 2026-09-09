import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tensor_diff import compare, grade  # noqa: E402


class TensorDiffTest(unittest.TestCase):
    def test_compare_reports_relative_l2_and_shape(self):
        report = compare([[3.0, 4.0]], [[3.0, 0.0]])
        self.assertEqual(report["shape_candidate"], [1, 2])
        self.assertEqual(report["shape_reference"], [1, 2])
        self.assertAlmostEqual(report["relative_l2"], 4.0 / 3.0)
        self.assertEqual(report["max_abs_error"], 4.0)

    def test_grade_requires_threshold_for_a_discriminating_control(self):
        report = grade([1.0, 1.0], [1.0, 1.0], [2.0, 2.0], 0.1)
        self.assertTrue(report["pass"])
        self.assertTrue(report["control_discriminates"])

    def test_control_without_threshold_is_not_claimed_to_discriminate(self):
        report = grade([1.0], [1.0], [2.0])
        self.assertNotIn("pass", report)
        self.assertFalse(report["control_discriminates"])

    def test_compare_rejects_shape_and_non_rectangular_inputs(self):
        with self.assertRaises(ValueError):
            compare([[1.0, 2.0]], [[1.0], [2.0]])
        with self.assertRaises(ValueError):
            compare([[1.0], [2.0, 3.0]], [[1.0], [2.0, 3.0]])

    def test_non_finite_candidate_cannot_pass(self):
        report = grade([math.nan], [1.0], max_relative_l2=0.1)
        self.assertTrue(math.isinf(report["relative_l2"]))
        self.assertFalse(report["pass"])


if __name__ == "__main__":
    unittest.main()
