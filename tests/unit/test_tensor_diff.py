import math
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tensor_diff import compare, grade, main  # noqa: E402
from core.storage import WritePolicyError
import pytest


def test_cli_registers_only_report_and_refuses_overwrite(tmp_path, monkeypatch):
    values = tmp_path / "values.json"
    values.write_text("[1, 2]")
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "tensor_diff", "--candidate", str(values), "--reference", str(values),
        "--out", str(out), "--max-relative-l2", "0.01",
    ])
    assert main() == 0
    manifest = json.loads((tmp_path / "report.json.manifest.json").read_text())
    assert [item["path"] for item in manifest["artifacts"]] == ["report.json"]
    with pytest.raises(WritePolicyError, match="already owned"):
        main()


def test_cli_rejects_source_report_before_reading_inputs(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "tensor_diff", "--candidate", "missing", "--reference", "missing",
        "--out", str(ROOT / "forbidden-report.json"),
    ])
    with pytest.raises(WritePolicyError, match="source repository"):
        main()
    assert not (ROOT / "forbidden-report.json").exists()


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
