import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class GraphTargetGateTests(unittest.TestCase):
    def run_graph(self, target):
        return subprocess.run(
            [
                sys.executable,
                "runners/graph_runner.py",
                "--subject", "demo",
                "--target", str(ROOT / target),
                "--artifact-root", "/tmp/infer-forge-target-gate",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def test_planned_target_is_blocked_before_graph_walk(self):
        result = self.run_graph("config/examples/p800-sglang-kunlun.yaml")
        self.assertEqual(result.returncode, 2)
        self.assertIn("blocked:", result.stdout + result.stderr)

    def test_supported_target_reaches_plan(self):
        result = self.run_graph("config/examples/p800-vllm-kunlun.yaml")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("blocked:", result.stdout)


if __name__ == "__main__":
    unittest.main()
