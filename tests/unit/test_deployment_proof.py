"""Manifest rendering tests for the deployment-proof executor."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from runners.deployment_proof import ActionFailed, manifest_values, render_manifest

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "tasks/kdp-001-deployment-proof/instances/qwen3-8b-p800.yaml"
TEMPLATE = REPO / "tasks/kdp-001-deployment-proof/manifests/feddeployment.template.yaml"


class TestManifestRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
        self.values = manifest_values(self.contract, "20260907T000000Z", "/workspace", "")

    def test_rendered_manifest_has_no_placeholders(self) -> None:
        rendered = render_manifest(TEMPLATE, self.values)
        self.assertNotIn("${", rendered)

    def test_rendered_manifest_is_owned_and_labelled(self) -> None:
        doc = yaml.safe_load(render_manifest(TEMPLATE, self.values))
        self.assertTrue(doc["metadata"]["name"].startswith("dongxinyu03-"))
        self.assertEqual(doc["metadata"]["namespace"], "pd-test")
        self.assertEqual(
            doc["metadata"]["labels"]["infer.kunlun/attempt-id"], "20260907T000000Z"
        )
        container = doc["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["resources"]["limits"]["kunlunxin.com/xpu"], "8")

    def test_missing_value_is_reported_as_contract_invalid(self) -> None:
        incomplete = dict(self.values)
        incomplete.pop("MODEL_PVC")
        with self.assertRaises(ActionFailed) as ctx:
            render_manifest(TEMPLATE, incomplete)
        self.assertEqual(ctx.exception.state, "CONTRACT_INVALID")


if __name__ == "__main__":
    unittest.main()
