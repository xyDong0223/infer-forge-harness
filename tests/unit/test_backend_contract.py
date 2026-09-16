import unittest
from types import SimpleNamespace

from runners.deployment_proof import DeploymentProofRunner


class BackendContractTests(unittest.TestCase):
    def test_backend_expectation_comes_from_runtime(self):
        runner = object.__new__(DeploymentProofRunner)
        runner.contract = {
            "context": {"runtime": {"backend": "cuda"}},
            "checks": {"backend": {"reject_unexpected_fallback": False}},
        }
        runner.checks = {}
        runner.pod = "pod"
        runner.adapter = SimpleNamespace(
            exec=lambda *args, **kwargs: SimpleNamespace(stdout="backend=cuda")
        )
        runner.server_log_path = lambda: "/tmp/server.log"
        runner.records = []
        runner.write = lambda *args: None
        runner.server_log = "backend=cuda"
        runner.record = lambda *args: runner.records.append(args)
        runner.verify_backend()
        self.assertEqual(runner.checks["expected_backend"], "cuda")


if __name__ == "__main__":
    unittest.main()
