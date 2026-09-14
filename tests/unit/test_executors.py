"""The sequenced executors for mat-006 and mat-007, with the pod faked.

What these tests pin is the sequence discipline, not the tools (their probes
have their own tests): triage always restores instrumentation even when the
middle fails; the verdict follows the isolated replay mechanically; a patch
that fails server validation or numerical agreement is removed, not recorded;
and neither executor may write its success state without the independent
validator's agreement.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.patch_executor import PatchExecutor  # noqa: E402
from runners.triage_executor import TriageExecutor  # noqa: E402

TRACE_LINE = json.dumps({
    "kernel": "speculative_attention",
    "error": "ValueError: Check 0 == ret failed",
    "head_num": 32, "kv_head_num": 8, "head_dim": 128,
    "block_size": 64, "batch_num": 1, "max_context_len": 32768,
})


class FakePodOps:
    def __init__(self, *, trace: str = TRACE_LINE + "\n",
                 replay_results: list[dict] | None = None,
                 service_state: str = "DEPLOYMENT_FAILED",
                 service_reason: str = "Check 0 == ret failed"):
        self.trace = trace
        self.replay_results = replay_results or [
            {"case": "captured_case", "ok": True},
            {"case": "batch_num=8", "ok": True},
        ]
        self.service_state = service_state
        self.service_reason = service_reason
        self.calls: list[str] = []

    def instrument(self, pod, call):
        self.calls.append(f"instrument:{call}")
        return "INSTRUMENTED"

    def restore(self, pod):
        self.calls.append("restore")
        return "RESTORED"

    def fetch_trace(self, pod):
        self.calls.append("fetch_trace")
        return self.trace

    def replay_in_pod(self, pod, call):
        self.calls.append(f"replay:{call}")
        return {"results": self.replay_results}

    def rerun_service(self, pod, contract_instance, out):
        self.calls.append("rerun_service")
        return {"state": self.service_state, "reason": self.service_reason}


class TriageSequenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "triage"

    def test_the_sequence_instruments_reruns_captures_replays_and_restores(self):
        ops = FakePodOps()
        status = TriageExecutor("pod-0", self.tmp / "instance.yaml",
                                "speculative_attention", ops, self.out).run()
        self.assertEqual(status["state"], "TRIAGE_READY")
        self.assertEqual(ops.calls, ["instrument:speculative_attention", "rerun_service",
                                     "fetch_trace", "replay:speculative_attention", "restore"])
        report = json.loads((self.out / "triage_report.json").read_text())
        self.assertEqual(report["verdict"], "RUNTIME_STATE_DEPENDENT")
        self.assertTrue((self.out / "kernel_failure_arguments.jsonl").exists())
        self.assertTrue((self.out / "kernel_unit_test_matrix.txt").exists())

    def test_an_isolated_failure_is_a_vendor_kernel_gap(self):
        ops = FakePodOps(replay_results=[
            {"case": "captured_case", "ok": False, "error": "ValueError: Check 0 == ret failed"},
        ])
        status = TriageExecutor("pod-0", self.tmp / "instance.yaml",
                                "speculative_attention", ops, self.out).run()
        self.assertEqual(status["state"], "TRIAGE_READY")
        self.assertEqual(status["verdict"], "VENDOR_KERNEL_GAP")
        self.assertEqual(status["owner"], "vendor_with_local_workaround")

    def test_restoration_happens_even_when_the_middle_fails(self):
        class ExplodingOps(FakePodOps):
            def fetch_trace(self, pod):
                self.calls.append("fetch_trace")
                raise RuntimeError("pod vanished")

        ops = ExplodingOps()
        status = TriageExecutor("pod-0", self.tmp / "instance.yaml",
                                "speculative_attention", ops, self.out).run()
        self.assertEqual(status["state"], "TRIAGE_FAILED")
        self.assertIn("restore", ops.calls)
        self.assertLess(ops.calls.index("fetch_trace"), ops.calls.index("restore"))

    def test_no_captured_arguments_is_a_failed_triage_not_a_guess(self):
        ops = FakePodOps(trace="")
        status = TriageExecutor("pod-0", self.tmp / "instance.yaml",
                                "speculative_attention", ops, self.out).run()
        self.assertEqual(status["state"], "TRIAGE_FAILED")
        self.assertIn("no captured arguments", status["reason"])


class PatchSequenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "patch"
        # The triage report the placement consumes: the captured geometry the
        # reproduction probe replays.
        self.triage = self.tmp / "triage_report.json"
        self.triage.write_text(json.dumps({
            "state": "TRIAGE_READY",
            "captured_arguments": {
                "head_num": 32, "kv_head_num": 8, "head_dim": 128,
                "block_size": 64, "batch_num": 1, "max_context_len": 32768,
                "context_lens_cpu": {"max": 128},
            },
        }), encoding="utf-8")

    def make_ops(self, *, cosine=1.0, service_state="DEPLOYMENT_READY",
                 reproduced=True):
        class FakePatchOps(FakePodOps):
            def __init__(self):
                super().__init__(service_state=service_state)
                self.removed = False
                self.geometry = None

            def apply(self, pod):
                self.calls.append("apply")
                return "INSTALLED"

            def remove(self, pod):
                self.calls.append("remove")
                self.removed = True
                return "REMOVED"

            def numerical_comparison(self, pod, geometry):
                self.calls.append("validate")
                self.geometry = geometry
                return {
                    "reproduction": {
                        "attempted": True, "reproduced": reproduced,
                        "error": "ValueError: Check 0 == ret failed",
                    },
                    "results": [
                        {"case": "b1/c128/k64", "cosine": cosine},
                        {"case": "b64/c4096/k128", "cosine": cosine},
                    ],
                }

        return FakePatchOps()

    def executor(self, ops):
        return PatchExecutor("pod-0", self.tmp / "instance.yaml", ops, self.out,
                             triage_report=self.triage)

    def test_a_valid_patch_is_placed_with_evidence(self):
        ops = self.make_ops()
        status = self.executor(ops).run()
        self.assertEqual(status["state"], "PATCH_PLACED")
        self.assertEqual(ops.calls, ["apply", "rerun_service", "validate"])
        self.assertFalse(ops.removed)
        self.assertEqual(ops.geometry["head_num"], 32)  # triage geometry reached the probe
        report = json.loads((self.out / "placement_report.json").read_text())
        self.assertEqual(report["mechanism"], "post_import_patch")
        self.assertGreaterEqual(report["numerical"]["min_cosine_similarity"], 0.9999)
        self.assertTrue(report["reversibility"]["reproduces_original_failure"])
        self.assertTrue(report["limitations"])

    def test_a_patch_that_breaks_the_server_is_removed(self):
        ops = self.make_ops(service_state="DEPLOYMENT_FAILED")
        status = self.executor(ops).run()
        self.assertEqual(status["state"], "PATCH_REJECTED")
        self.assertTrue(ops.removed)

    def test_a_patch_below_numerical_threshold_is_removed(self):
        ops = self.make_ops(cosine=0.99)
        status = self.executor(ops).run()
        self.assertEqual(status["state"], "PATCH_REJECTED")
        self.assertTrue(ops.removed)

    def test_a_patch_whose_original_failure_no_longer_reproduces_is_removed(self):
        """No reproduction means no evidence the patch is needed."""
        ops = self.make_ops(reproduced=False)
        status = self.executor(ops).run()
        self.assertEqual(status["state"], "PATCH_REJECTED")
        self.assertTrue(ops.removed)
        self.assertIn("no longer reproduces", status["reason"])

    def test_placement_without_a_triage_report_is_refused(self):
        ops = self.make_ops()
        status = PatchExecutor("pod-0", self.tmp / "instance.yaml", ops, self.out,
                               triage_report=None).run()
        self.assertEqual(status["state"], "PATCH_REJECTED")
        self.assertIn("triage report", status["reason"])

    def test_missing_numerical_evidence_rejects_rather_than_skips(self):
        class NoCompareOps(self.make_ops().__class__):
            def numerical_comparison(self, pod, geometry):
                self.calls.append("validate")
                return {"error": "validation produced no output"}

        ops = NoCompareOps()
        status = self.executor(ops).run()
        self.assertEqual(status["state"], "PATCH_REJECTED")
        self.assertTrue(ops.removed)


if __name__ == "__main__":
    unittest.main()
