import unittest

from core.contracts import Artifact, Metric, Workload
from runners.performance_runner import PerformanceRunner


class FakePerformance:
    def prepare_workload(self, workload): return workload
    def run_benchmark(self, workload):
        return [Metric("throughput", 100.0, "tok/s")]
    def collect_trace(self, workload):
        return [Artifact("/tmp/trace", "trace")]
    def extract_metrics(self, artifacts):
        return [Metric("throughput", 100.0, "tok/s")]


class PerformanceRunnerTests(unittest.TestCase):
    def test_runner_uses_adapter_and_emits_evidence(self):
        result = PerformanceRunner(FakePerformance(), "/tmp/perf-run").run(
            Workload("smoke", input_lengths=(128,), output_lengths=(16,))
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["artifacts"][0]["kind"], "trace")

    def test_regression_fails_gate(self):
        result = PerformanceRunner(FakePerformance(), "/tmp/perf-run").run(
            Workload("smoke"),
            [Metric("throughput", 200.0, "tok/s")],
        )
        self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
