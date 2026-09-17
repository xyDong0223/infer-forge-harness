"""Safety-guard tests for the Kunlun P800 adapter."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch
from pathlib import Path

from adapters.kunlun_p800 import ClusterConfig, KunlunP800Adapter, SafetyViolation, push_snippet

CONFIG = Path(__file__).resolve().parents[2] / "config" / "clusters" / "p800-cluster.yaml"


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
    def test_explicit_user_id_overrides_environment_without_truncation(self) -> None:
        with patch.dict(os.environ, {"KUBECONFIG": __file__, "USER_ID": "wrong-owner"}):
            config = ClusterConfig.load(CONFIG, user_id="team-member")
            self.assertEqual(config.resource_prefix, "team-member-")
            self.assertEqual(os.environ["USER_ID"], "wrong-owner")
            adapter = KunlunP800Adapter(config)
            adapter.assert_owned("team-member-environment-base")
            with self.assertRaises(SafetyViolation):
                adapter.assert_owned("team-environment-base")

    def test_repository_config_loads_with_env_kubeconfig(self) -> None:
        os.environ["KUBECONFIG"] = __file__  # any existing readable path
        config = ClusterConfig.load(CONFIG)
        self.assertEqual(config.namespace, "pd-test")
        self.assertEqual(config.resource_prefix, "<USER_ID>-")
        self.assertTrue(config.deployment_kind.startswith("feddeployments"))

    def test_missing_env_kubeconfig_is_refused(self) -> None:
        os.environ.pop("KUBECONFIG", None)
        with self.assertRaises(ValueError) as ctx:
            ClusterConfig.load(CONFIG)
        self.assertIn("KUBECONFIG", str(ctx.exception))

    def test_config_holds_no_credential_path(self) -> None:
        """A public repository must not point at a credential file."""
        self.assertNotIn("kconf", CONFIG.read_text(encoding="utf-8").lower())


class TestPushSnippet(unittest.TestCase):
    """The one pod-push transport every harness site must share."""

    def test_round_trips_bytes(self) -> None:
        snippet = push_snippet(b"print('hi')", "/tmp/probe.py")
        self.assertIn("base64 -d", snippet)
        self.assertTrue(snippet.startswith("echo "))
        self.assertTrue(snippet.endswith("| base64 -d > /tmp/probe.py"))

    def test_str_and_path_agree_with_bytes(self) -> None:
        by_bytes = push_snippet(b"data", "/tmp/a.py")
        by_str = push_snippet("data", "/tmp/a.py")
        self.assertEqual(by_bytes, by_str)

    def test_remote_is_quoted(self) -> None:
        snippet = push_snippet(b"x", "/tmp/with space.py")
        self.assertIn("'/tmp/with space.py'", snippet)


if __name__ == "__main__":
    unittest.main()
