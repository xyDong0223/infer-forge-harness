import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_lifecycle import (  # noqa: E402
    DEFAULT_KNOWN_OPS,
    dispatch,
    freeze_baseline,
    integration_decision,
    load_known_operators,
)


class OperatorLifecycleTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_dispatch_is_non_blocking_and_creates_one_request_per_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gaps = self.write_json(
                root / "gaps.json",
                {"gaps": [{"operator": "foo"}, {"symbol": "bar"}, {"reason": "no operator"}]},
            )
            report = dispatch(gaps, root / "dispatch", "Model", "baseline-1")
            self.assertEqual(report["state"], "DISPATCHED")
            self.assertEqual(report["request_count"], 2)
            self.assertTrue((root / "dispatch/requests/Model-op-001.json").exists())

    def test_dispatch_derives_operator_request_for_capability_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gaps = self.write_json(
                root / "gaps.json",
                {"gaps": [{"class": "CAPABILITY_MISSING", "axis": "swiglu_oai"}]},
            )
            report = dispatch(gaps, root / "dispatch", "Model")
            self.assertEqual(report["request_count"], 1)
            request = json.loads(
                (root / "dispatch/requests/Model-op-001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(request["operator"], "swiglu_oai")

    def test_dispatch_skips_operators_the_repo_already_ships(self):
        # The GLM-5.2 lesson: two of three dispatches duplicated kernels
        # XSpeedGate already had. An existing operator is recorded as a
        # duplicate, never dispatched.
        self.assertTrue(DEFAULT_KNOWN_OPS.exists(),
                        "catalog/known_operators.json must be committed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            known = self.write_json(root / "known.json", {"operators": ["existing_op"]})
            gaps = self.write_json(
                root / "gaps.json",
                {"gaps": [{"operator": "existing_op"}, {"operator": "genuinely_new"}]},
            )
            report = dispatch(gaps, root / "dispatch", "Model", known_ops=known)
            self.assertEqual(report["request_count"], 1)
            self.assertEqual(
                [dup["operator"] for dup in report["skipped_duplicates"]], ["existing_op"]
            )
            self.assertEqual(
                json.loads(
                    (root / "dispatch/requests/Model-op-002.json").read_text(encoding="utf-8")
                )["operator"],
                "genuinely_new",
            )

    def test_dispatch_without_an_index_still_dispatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gaps = self.write_json(root / "gaps.json", {"gaps": [{"operator": "any_op"}]})
            report = dispatch(gaps, root / "dispatch", "Model",
                              known_ops=root / "missing.json")
            self.assertEqual(report["state"], "DISPATCHED")
            self.assertEqual(report["request_count"], 1)
            self.assertEqual(load_known_operators(root / "missing.json"), set())

    def test_baseline_requires_both_service_and_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = self.write_json(root / "service.json", {"state": "DEPLOYMENT_READY"})
            accuracy = self.write_json(root / "accuracy.json", {"state": "ACCURACY_PASS"})
            report = freeze_baseline(service, accuracy, root / "baseline", "Model", {"hw": "P800"})
            self.assertEqual(report["state"], "BASELINE_FROZEN")
            manifest = json.loads(
                (root / "baseline/baseline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["integration_policy"]["one_candidate_at_a_time"])
            self.assertTrue(manifest["integration_policy"]["batch_integration"])
            self.assertEqual(
                manifest["integration_policy"]["attribution_on_failure"], "automated_bisect"
            )

    def test_integration_waits_without_mutating_the_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.write_json(root / "baseline.json", {"baseline_id": "b-1"})
            report = integration_decision(baseline, [], root / "integration", "Model")
            self.assertEqual(report["state"], "WAITING_FOR_CANDIDATE")

    def test_integration_rejects_any_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.write_json(root / "baseline.json", {"baseline_id": "b-1"})
            candidate = self.write_json(
                root / "candidate.json",
                {
                    "kernel_grade": "KERNEL_PASS",
                    "dispatch_report": "DISPATCH_CONFIRMED",
                    "service_regression": "PASS",
                    "accuracy_regression": "FAIL",
                },
            )
            report = integration_decision(baseline, [candidate], root / "integration", "Model")
            self.assertEqual(report["state"], "CANDIDATE_REJECTED")
            self.assertIn("accuracy_regression", report["candidates"][0]["failed_gates"])

    def _complete_candidate(self, path):
        return self.write_json(
            path,
            {
                "kernel_grade": "KERNEL_PASS",
                "dispatch_report": "DISPATCH_CONFIRMED",
                "package_swap": "PASS",
                "path_proof": "PASS",
                "service_regression": "PASS",
                "accuracy_regression": "PASS",
                "kernel_grade_report": "kernel_grade.json",
                "dispatch_report_path": "dispatch_report.json",
                "package_swap_report": "package_swap.json",
                "worker_path_log": "worker_path.log",
                "service_regression_report": "service_regression.json",
                "accuracy_regression_report": "accuracy_regression.json",
            },
        )

    def test_a_candidate_without_swap_evidence_is_rejected(self):
        # The GLM-5.2 lesson: four swaps failed on glibc, base commit, a
        # side-car module and a version gate before anyone demanded this
        # evidence. A candidate that cannot show how it was built into the
        # pod does not reach the service.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.write_json(root / "baseline.json", {"baseline_id": "b-1"})
            candidate = self.write_json(
                root / "candidate.json",
                {
                    "kernel_grade": "KERNEL_PASS",
                    "dispatch_report": "DISPATCH_CONFIRMED",
                    "service_regression": "PASS",
                    "accuracy_regression": "PASS",
                },
            )
            report = integration_decision(baseline, [candidate], root / "integration", "Model")
            self.assertEqual(report["state"], "CANDIDATE_REJECTED")
            self.assertIn("package_swap", report["candidates"][0]["failed_gates"])
            self.assertIn("path_proof", report["candidates"][0]["failed_gates"])

    def test_a_candidate_with_full_evidence_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.write_json(root / "baseline.json", {"baseline_id": "b-1"})
            candidate = self._complete_candidate(root / "candidate.json")
            report = integration_decision(baseline, [candidate], root / "integration", "Model")
            self.assertEqual(report["state"], "READY_FOR_INTEGRATION")

    def test_a_full_batch_is_ready_in_one_decision(self):
        # The policy change: N operators cost one regression run, not N. A
        # rejected member poisons only itself; the rest of the batch stays
        # eligible, and attribution is a recorded bisect schedule.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.write_json(root / "baseline.json", {"baseline_id": "b-1"})
            good = self._complete_candidate(root / "good.json")
            good_payload = json.loads(good.read_text(encoding="utf-8"))
            good_payload["candidate_id"] = "good-op"
            good.write_text(json.dumps(good_payload), encoding="utf-8")
            bad = self.write_json(
                root / "bad.json",
                {"candidate_id": "bad-op", "kernel_grade": "KERNEL_FAIL"},
            )
            mixed = integration_decision(baseline, [good, bad], root / "mixed", "Model")
            self.assertEqual(mixed["state"], "CANDIDATE_REJECTED")
            self.assertEqual(mixed["failed_candidates"], ["bad-op"])

            full_batch = [self._complete_candidate(root / f"c{index}.json")
                          for index in range(4)]
            ready = integration_decision(baseline, full_batch, root / "ready", "Model")
            self.assertEqual(ready["state"], "READY_FOR_INTEGRATION")
            self.assertEqual(ready["batch_size"], 4)
            self.assertTrue(ready["bisect_schedule"])
            final_round = ready["bisect_schedule"][-1]
            self.assertEqual(len(final_round["integrate"]), 1)

    def test_ready_without_evidence_files_is_refused(self):
        from validators.operator_lifecycle_validator import validate_integration

        report = {
            "state": "READY_FOR_INTEGRATION",
            "candidates": [{"candidate_id": "c-1", "failed_gates": {
                "evidence:package_swap": None, "evidence:path_proof": None}}],
            "bisect_schedule": [{"round": 1}],
        }
        errors = validate_integration(report)
        self.assertTrue(any("package_swap_report" in e for e in errors))
        self.assertTrue(any("worker_path_log" in e for e in errors))

    def test_rejection_must_name_its_candidates(self):
        from validators.operator_lifecycle_validator import validate_integration

        errors = validate_integration({"state": "CANDIDATE_REJECTED"})
        self.assertTrue(any("name the candidates" in e for e in errors))

    def test_ready_without_a_bisect_schedule_is_refused(self):
        from validators.operator_lifecycle_validator import validate_integration

        report = {
            "state": "READY_FOR_INTEGRATION",
            "candidates": [{"candidate_id": "c-1", "failed_gates": {}}],
        }
        errors = validate_integration(report)
        self.assertTrue(any("bisect schedule" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
