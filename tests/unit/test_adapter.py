"""Safety-guard tests for the Kunlun P800 adapter."""

from __future__ import annotations

import unittest
from pathlib import Path

from adapters.kunlun_p800 import ClusterConfig, KunlunP800Adapter, SafetyViolation

CONFIG = Path(__file__).resolve().parents[2] / "harness" / "config.yaml"


def make_adapter() -> KunlunP800Adapter:
    return KunlunP800Adapter(
        ClusterConfig(
            kubeconfig="/dev/null",
            namespace="pd-test",
            container="model-server",
            resource_prefix="dongxinyu03-",
            deployment_kind="feddeployments.eks.baidu.com",
        )
    )


class TestSafetyGuard(unittest.TestCase):
    def test_owned_resource_is_allowed(self) -> None:
        make_adapter().assert_owned("dongxinyu03-kdp001-a1")

    def test_other_operator_resource_is_refused(self) -> None:
        adapter = make_adapter()
        for name in ("gjj-vllm-deepseek-v4-decode-rb", "attention-store-agent", ""):
            with self.subTest(name=name), self.assertRaises(SafetyViolation):
                adapter.assert_owned(name)

    def test_manifest_name_is_checked(self) -> None:
        adapter = make_adapter()
        with self.assertRaises(SafetyViolation):
            adapter.assert_manifest_owned({"metadata": {"name": "someone-else-deploy"}})

    def test_delete_needs_human_gate(self) -> None:
        adapter = make_adapter()
        with self.assertRaises(SafetyViolation):
            adapter.delete("pod", "dongxinyu03-kdp001-a1")


class TestHarnessConfig(unittest.TestCase):
    def test_repository_config_loads(self) -> None:
        config = ClusterConfig.load(CONFIG)
        self.assertEqual(config.namespace, "pd-test")
        self.assertEqual(config.resource_prefix, "dongxinyu03-")
        self.assertTrue(config.deployment_kind.startswith("feddeployments"))


if __name__ == "__main__":
    unittest.main()
