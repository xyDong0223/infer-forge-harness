"""The sequenced correctness executors, with the pod faked and the grader real.

What these tests pin: a control that cannot fail makes the grade AMBIGUOUS and
never a pass; a kernel that raises is a measured FAIL, not missing evidence;
the end-to-end report is packaged from the differential's own case data, not
re-measured; and a geometry that does not cross the selection boundary is
refused before anything expensive runs. tensor_diff itself runs as a real
subprocess on small tensor files, because the executor's promise is that the
contract's own grader produced the numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.correctness_executor import (  # noqa: E402
    run_end_to_end,
    run_kernel_grade,
    run_long_context,
)

REFERENCE = [[1.0, 2.0], [3.0, 4.0]]
DISCRIMINATING_CONTROL = [[9.0, 9.0], [9.0, 9.0]]
NON_DISCRIMINATING_CONTROL = [[1.0, 2.0], [3.0, 4.0]]
WRONG_CANDIDATE = [[5.0, 5.0], [5.0, 5.0]]


class FakeOps:
    """run_probe and fetch are faked; run_tool executes real subprocesses."""

    def __init__(self, *, probe_state="EXERCISED_PASS",
                 candidate=REFERENCE, control=DISCRIMINATING_CONTROL,
                 differential_status="ACCURACY_PASS", differential_missing=False):
        self.probe_state = probe_state
        self.tensors = {
            "candidate.json": json.dumps(candidate),
            "reference.json": json.dumps(REFERENCE),
            "control.json": json.dumps(control),
            "selected_blocks.json": json.dumps([[3, 1, 7, 0]]),
        }
        self.differential_status = differential_status
        self.differential_missing = differential_missing
        self.calls: list[str] = []

    def run_probe(self, pod, probe, files, args):
        self.calls.append(f"probe:{probe.name}:{args}")
        return {"state": self.probe_state, "geometry": {"context_len": 4096},
                "operators": ["kunlun_ops.speculative_attention"]}

    def fetch(self, pod, remote):
        name = remote.rsplit("/", 1)[-1]
        return self.tensors.get(name, "")

    def run_tool(self, command):
        self.calls.append("tool:" + command[1].rsplit("/", 1)[-1])
        if "accuracy_differential" in " ".join(command):
            if self.differential_missing:
                return SimpleNamespace(returncode=1, stdout="", stderr="server unreachable")
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "accuracy_differential.json").write_text(json.dumps({
                "status": self.differential_status, "subject": "Qwen3-8B", "revision": "abc",
                "reference": {"implementation": "transformers on CPU", "device": "cpu",
                              "dtype": "float32", "transformers_version": "4.55.0"},
                "metric": "top-1 next-token identity",
                "top1_agreement": "3/3",
                "cases": [{"prompt": "The capital of France is", "top1_match": True,
                           "top1_candidate": " Paris", "top1_reference": " Paris",
                           "top5_overlap": 5}],
            }), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


class KernelGradeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "mat-021"

    def test_a_matching_kernel_with_a_discriminating_control_passes(self):
        status = run_kernel_grade("pod-0", FakeOps(), self.out, 0.02)
        self.assertEqual(status["state"], "KERNEL_PASS")
        grade = json.loads((self.out / "kernel_grade.json").read_text())
        self.assertTrue(grade["control_discriminates"])
        self.assertTrue(grade["reference_provenance"]["independently_written"])
        self.assertTrue((self.out / "grade_raw.json").exists())
        self.assertTrue((self.out / "tensors" / "candidate.json").exists())

    def test_a_mismatching_kernel_fails(self):
        status = run_kernel_grade("pod-0", FakeOps(candidate=WRONG_CANDIDATE), self.out, 0.02)
        self.assertEqual(status["state"], "KERNEL_FAIL")

    def test_a_control_that_cannot_fail_makes_the_grade_ambiguous(self):
        status = run_kernel_grade(
            "pod-0", FakeOps(control=NON_DISCRIMINATING_CONTROL), self.out, 0.02)
        self.assertEqual(status["state"], "KERNEL_AMBIGUOUS")

    def test_a_kernel_that_raises_is_a_measured_failure(self):
        status = run_kernel_grade(
            "pod-0", FakeOps(probe_state="EVALUATION_ERROR"), self.out, 0.02)
        self.assertEqual(status["state"], "KERNEL_FAIL")
        self.assertIn("raised", status["reason"])

    def test_an_inconclusive_probe_is_ambiguous_not_a_pass(self):
        status = run_kernel_grade(
            "pod-0", FakeOps(probe_state="EVALUATION_INCONCLUSIVE"), self.out, 0.02)
        self.assertEqual(status["state"], "KERNEL_AMBIGUOUS")


class EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "mat-022"
        self.request = self.tmp / "model_request.yaml"
        self.request.write_text("model: {id: Qwen3-8B}", encoding="utf-8")

    def test_a_passing_differential_becomes_an_integrated_path_report(self):
        status = run_end_to_end("pod-0", FakeOps(), self.out, self.request,
                                "qwen3-8b", 8000)
        self.assertEqual(status["state"], "ACCURACY_PASS")
        report = json.loads((self.out / "end_to_end_accuracy.json").read_text())
        self.assertEqual(report["composition"], "integrated_serving_path")
        self.assertEqual(report["reference_provenance"]["device"], "cpu")
        self.assertNotEqual(report["reference_provenance"]["device"],
                            report["candidate"]["device"])
        self.assertTrue(all(case["evidence"] for case in report["cases"]))

    def test_a_failing_differential_fails(self):
        status = run_end_to_end("pod-0", FakeOps(differential_status="ACCURACY_FAIL"),
                                self.out, self.request, "qwen3-8b", 8000)
        self.assertEqual(status["state"], "ACCURACY_FAIL")

    def test_no_differential_report_is_needs_human_not_a_guessed_verdict(self):
        status = run_end_to_end("pod-0", FakeOps(differential_missing=True), self.out,
                                self.request, "qwen3-8b", 8000)
        self.assertEqual(status["state"], "NEEDS_HUMAN")

    def test_without_a_model_request_the_contract_is_invalid(self):
        status = run_end_to_end("pod-0", FakeOps(), self.out, None, "qwen3-8b", 8000)
        self.assertEqual(status["state"], "CONTRACT_INVALID")


class LongContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "mat-023"

    def test_geometry_below_the_boundary_is_refused_before_anything_runs(self):
        ops = FakeOps()
        status = run_long_context("pod-0", ops, self.out, context_len=512,
                                  block_size=64, topk=8, max_relative_l2=0.02)
        self.assertEqual(status["state"], "CONTRACT_INVALID")
        self.assertIn("block_size*topk", status["reason"])
        self.assertEqual(ops.calls, [])  # nothing touched the pod

    def test_a_sparse_pass_records_selection_and_crosses_the_boundary(self):
        status = run_long_context("pod-0", FakeOps(), self.out, context_len=4096,
                                  block_size=64, topk=8, max_relative_l2=0.02)
        self.assertEqual(status["state"], "LONG_CONTEXT_PASS")
        report = json.loads((self.out / "long_context_grade.json").read_text())
        self.assertEqual(report["path"], "sparse")
        self.assertTrue(report["selected_blocks"])
        self.assertGreater(report["geometry"]["context_len"], report["geometry"]["boundary"])

    def test_a_mismatch_fails_long_context(self):
        status = run_long_context("pod-0", FakeOps(candidate=WRONG_CANDIDATE), self.out,
                                  context_len=4096, block_size=64, topk=8, max_relative_l2=0.02)
        self.assertEqual(status["state"], "LONG_CONTEXT_FAIL")

    def test_an_undiscriminating_control_is_ambiguous(self):
        status = run_long_context(
            "pod-0", FakeOps(control=NON_DISCRIMINATING_CONTROL), self.out,
            context_len=4096, block_size=64, topk=8, max_relative_l2=0.02)
        self.assertEqual(status["state"], "LONG_CONTEXT_AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()
