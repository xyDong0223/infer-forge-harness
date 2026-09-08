import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe.cpu_reference_logits_probe import _visible_accelerator


class CpuReferenceProbeTest(unittest.TestCase):
    def test_xpu_visibility_is_rejected(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            xpu=SimpleNamespace(is_available=lambda: True),
        )
        self.assertEqual(_visible_accelerator(torch), "xpu")

    def test_cpu_only_runtime_is_allowed(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            xpu=SimpleNamespace(is_available=lambda: False),
        )
        self.assertIsNone(_visible_accelerator(torch))


if __name__ == "__main__":
    unittest.main()
