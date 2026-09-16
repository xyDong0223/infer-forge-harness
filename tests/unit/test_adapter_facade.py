import unittest
from pathlib import Path

from core.facade import resolve_adapters
from core.target import load_target


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


if __name__ == "__main__":
    unittest.main()
