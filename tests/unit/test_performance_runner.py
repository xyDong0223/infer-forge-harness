import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from core.contracts import Artifact, Metric, Workload
from runners.performance_runner import PerformanceRunner


class FakePerformance:
    """Synthetic adapter; these tests do not run or validate device benchmarks."""

    def __init__(self, root, metrics=None, extracted=None):
        self.root = root
        self.metrics = metrics if metrics is not None else [Metric("throughput", 100.0, "tok/s")]
        self.extracted = (
            extracted if extracted is not None else [Metric("throughput", 1.0, "tok/s")]
        )

    def prepare_workload(self, workload):
        return workload

    def run_benchmark(self, workload):
        return self.metrics

    def collect_trace(self, workload):
        trace = self.root / "synthetic_trace.json"
        trace.write_text('{"traceEvents": []}\n', encoding="utf-8")
        return [Artifact(str(trace), "trace", metadata={"synthetic": True})]

    def extract_metrics(self, artifacts):
        return self.extracted


class PerformanceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("artifacts") / f"performance-tests-{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.workload = Workload("synthetic", input_lengths=(128,), output_lengths=(16,))
        self.baseline = [Metric("throughput", 100.0, "tok/s")]

    def runner(self, metrics=None, extracted=None):
        return PerformanceRunner(FakePerformance(self.root, metrics, extracted), self.root)

    def assert_persisted(self, result):
        path = Path(result["report_path"])
        self.assertEqual(path.parent, self.root.resolve())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)
        self.assertFalse(list(self.root.glob("*.pending")))
        for gate in result["gates"]:
            self.assertEqual(gate["evidence"], [str(path)])

    def test_runner_preserves_benchmark_and_trace_metrics_and_emits_evidence(self):
        with patch("runners.performance_runner.os.replace", wraps=os.replace) as replace:
            result = self.runner().run(self.workload, self.baseline)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["metrics"], result["benchmark_metrics"])
        self.assertEqual(result["benchmark_metrics"][0]["value"], 100.0)
        self.assertEqual(result["trace_metrics"][0]["value"], 1.0)
        self.assertEqual(result["baseline"][0]["value"], 100.0)
        self.assertEqual(result["workload"]["input_lengths"], [128])
        self.assertTrue(Path(result["artifacts"][0]["path"]).is_file())
        self.assertEqual(result["artifacts"][0]["kind"], "trace")
        replace.assert_called_once()
        self.assert_persisted(result)

    def test_trace_metrics_cannot_hide_benchmark_regression(self):
        result = self.runner(
            metrics=[Metric("throughput", 50.0, "tok/s")],
            extracted=[Metric("throughput", 200.0, "tok/s")],
        ).run(self.workload, self.baseline)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["benchmark_metrics"][0]["value"], 50.0)
        self.assert_persisted(result)

    def test_missing_baseline_is_unknown_not_pass_or_fail(self):
        result = self.runner().run(self.workload)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("missing baseline", result["gates"][0]["reason"])
        self.assert_persisted(result)

    def test_no_metrics_is_unknown(self):
        result = self.runner(metrics=[], extracted=[]).run(self.workload)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assert_persisted(result)

    def test_empty_gates_cannot_pass(self):
        with patch("runners.performance_runner.compare_metrics", return_value=[]):
            result = self.runner().run(self.workload, self.baseline)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assert_persisted(result)

    def test_missing_required_candidates_fail_even_when_trace_has_metrics(self):
        result = self.runner(metrics=[]).run(self.workload, self.baseline)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["benchmark_metrics"], [])
        self.assertIn("missing candidate", result["gates"][0]["reason"])
        self.assert_persisted(result)

    def test_mixed_pass_and_unknown_cannot_pass(self):
        result = self.runner(
            metrics=self.baseline + [Metric("mean_ttft_ms", 20.0, "ms")]
        ).run(self.workload, self.baseline)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual([g["verdict"] for g in result["gates"]], ["PASS", "UNKNOWN"])
        self.assert_persisted(result)

    def test_unit_mismatch_is_incomparable(self):
        result = self.runner(metrics=[Metric("throughput", 100.0, "req/s")]).run(
            self.workload, self.baseline
        )
        self.assertEqual(result["status"], "INCOMPARABLE")
        self.assert_persisted(result)

    def test_nonfinite_metrics_persist_as_explicit_strings_in_strict_json(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for side in ("baseline", "candidate"):
                with self.subTest(value=value, side=side):
                    invalid = [Metric("throughput", value, "tok/s")]
                    baseline = invalid if side == "baseline" else self.baseline
                    metrics = invalid if side == "candidate" else self.baseline
                    result = self.runner(metrics=metrics, extracted=invalid).run(
                        self.workload, baseline
                    )
                    self.assertEqual(result["status"], "INCOMPARABLE")
                    field = "baseline" if side == "baseline" else "benchmark_metrics"
                    self.assertEqual(result[field][0]["value"], str(value))
                    self.assertEqual(result["trace_metrics"][0]["value"], str(value))
                    text = Path(result["report_path"]).read_text(encoding="utf-8")
                    json.loads(text, parse_constant=self.fail)
                    self.assert_persisted(result)

    def test_benchmark_passes_without_extracted_metrics(self):
        result = self.runner(extracted=[]).run(self.workload, self.baseline)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["trace_metrics"], [])
        self.assert_persisted(result)

    def test_failed_publication_preserves_previous_report_and_cleans_pending_file(self):
        previous = self.runner().run(self.workload, self.baseline)
        with patch("runners.performance_runner.os.replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                self.runner(metrics=[]).run(self.workload, self.baseline)
        self.assert_persisted(previous)


if __name__ == "__main__":
    unittest.main()
