import unittest
from pathlib import Path

from core.target import compatibility_status, load_target, require_supported
from core.target import canonical_hardware
from core.target import bind_subject, contract_target, target_environment, target_from_mapping


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

    def test_model_and_revisions_do_not_change_platform_compatibility(self):
        target = target_from_mapping({
            "model": "model-a", "hardware": "p800",
            "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun",
                        "revisions": {"model": " rev-a ", "plugin": "rev-b"}},
        })
        self.assertEqual(compatibility_status(target)["status"], "supported")
        self.assertEqual(target_environment(target)["model_revision"], "rev-a")
        with self.assertRaisesRegex(ValueError, "subject"):
            bind_subject(target, "model-b")

    def test_contract_revisions_cannot_override_requested_revisions(self):
        target = target_from_mapping({
            "model": "demo", "hardware": "p800",
            "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun",
                        "revisions": {"plugin": "rev-a"}},
        })
        contract = {"context": {"model": {"name": "demo"},
                                "runtime": {"revisions": {"plugin": "rev-b"}}}}
        with self.assertRaisesRegex(ValueError, "revision"):
            contract_target(contract, target)

    def test_environment_proof_can_use_a_different_base_model(self):
        target = bind_subject(load_target(ROOT / "config/examples/p800-vllm-kunlun.yaml"), "demo")
        contract = {"metadata": {"task_type": "environment_proof"},
                    "context": {"model": {"name": "base-smoke"}}}
        self.assertEqual(contract_target(contract, target).model, "base-smoke")
        contract["metadata"]["task_type"] = "service_proof"
        with self.assertRaises(ValueError):
            contract_target(contract, target)

    def test_checkpoint_revision_cannot_be_relabeled_by_target(self):
        target = target_from_mapping({
            "model": "demo", "hardware": "p800",
            "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun",
                        "revisions": {"model": "new"}},
        })
        contract = {"metadata": {"task_type": "service_proof"},
                    "context": {"model": {"name": "demo", "revision": "old"}}}
        with self.assertRaisesRegex(ValueError, "revision"):
            contract_target(contract, target)
        self.assertEqual(contract_target(contract).revisions["model"], "old")
        contract["metadata"]["task_type"] = "environment_proof"
        contract["context"]["model"]["name"] = "base-smoke"
        self.assertEqual(contract_target(contract, target).revisions["model"], "new")


if __name__ == "__main__":
    unittest.main()
