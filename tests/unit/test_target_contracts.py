import unittest
from pathlib import Path

from core.target import compatibility_status, load_target, require_supported
from core.target import canonical_hardware


ROOT = Path(__file__).resolve().parents[2]


class TargetContractTests(unittest.TestCase):
    def test_legacy_hardware_alias_is_normalized(self):
        self.assertEqual(canonical_hardware("Kunlunxin-3-P800"), "kunlun/p800")

    def test_p800_vllm_is_supported(self):
        target = load_target(ROOT / "config/examples/p800-vllm-kunlun.yaml")
        self.assertEqual(compatibility_status(target)["status"], "supported")
        require_supported(target)

    def test_p800_sglang_is_declared_planned(self):
        target = load_target(ROOT / "config/examples/p800-sglang-kunlun.yaml")
        self.assertEqual(compatibility_status(target)["status"], "planned")

    def test_b200_sglang_is_declared_planned(self):
        target = load_target(ROOT / "config/examples/b200-sglang.yaml")
        self.assertEqual(compatibility_status(target)["status"], "planned")

    def test_b200_vllm_is_rejected(self):
        target = load_target(ROOT / "config/examples/p800-vllm-kunlun.yaml")
        target = target.__class__(
            model=target.model, hardware="nvidia/b200", engine="vllm",
            backend="cuda", plugin=None,
        )
        self.assertEqual(compatibility_status(target)["status"], "unsupported")
        with self.assertRaises(ValueError):
            require_supported(target)


if __name__ == "__main__":
    unittest.main()
