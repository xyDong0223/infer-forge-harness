import unittest
from pathlib import Path
from unittest.mock import patch

from core.facade import resolve_adapters
from core.target import load_target
from core.contracts import TargetContext


ROOT = Path(__file__).resolve().parents[2]


class AdapterFacadeTests(unittest.TestCase):
    def test_resolves_existing_p800_stack(self):
        target = load_target(ROOT / "config/examples/p800-vllm-kunlun.yaml")
        bundle = resolve_adapters(target, require_supported=True)
        self.assertEqual(bundle.compatibility["status"], "supported")
        self.assertEqual(bundle.runtime.profile.framework, "vllm-kunlun")
        self.assertEqual(bundle.hardware.__name__, "KunlunP800Adapter")

    def test_planned_stack_is_not_silently_fallbacked(self):
        target = load_target(ROOT / "config/examples/p800-sglang-kunlun.yaml")
        with self.assertRaises(ValueError):
            resolve_adapters(target, require_supported=True)

    def test_unknown_compatibility_does_not_resolve_hardware_or_runtime(self):
        target = TargetContext("demo", "unknown/device", "unknown", "unknown")
        with patch("core.facade.get_hardware") as hardware, patch("core.facade.get_runtime") as runtime:
            bundle = resolve_adapters(target)
            self.assertEqual(bundle.compatibility["status"], "unknown")
            self.assertIsNone(bundle.hardware)
            self.assertIsNone(bundle.runtime)
            with self.assertRaisesRegex(ValueError, "unknown"):
                resolve_adapters(target, require_supported=True)
        hardware.assert_not_called()
        runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
