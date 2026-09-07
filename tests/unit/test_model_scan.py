"""MAT-002: what a support verdict may claim.

The Qwen3 scan below is the real 2026-09-07 output, which is the case that
matters most: it is the verdict that says "this is not a gap".
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scan_model_support import ScanFailed, main as scan_main, render_card  # noqa: E402
from validators.scan_validator import validate_support_card  # noqa: E402

CONTRACT_PATH = ROOT / "tasks" / "mat-002-model-scan" / "task.yaml"

SCAN = {
    "state": "SCAN_READY",
    "scanned_in": "dongxinyu03-kdp001-qwen3-8b-7c969df894-hsz7d-0",
    "results": [
        {
            "architecture": "Qwen3ForCausalLM",
            "vllm_version": "0.25.1",
            "kunlun_oot_archs": ["DeepseekV3ForCausalLM", "Qwen3NextForCausalLM"],
            "in_kunlun_oot": False,
            "in_installed_vllm": True,
            "verdict": "UPSTREAM_GENERIC",
            "meaning": "the installed vLLM implementation is used",
        }
    ],
}


class ScanValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        import yaml

        self.contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.scan = copy.deepcopy(SCAN)

    def entry(self) -> dict:
        return self.scan["results"][0]

    def test_the_real_qwen3_scan_passes(self):
        self.assertEqual(validate_support_card(self.scan, self.contract), [])

    def test_absent_cannot_be_claimed_when_upstream_was_unreachable(self):
        """The failure that would send someone to write an existing model."""
        self.entry().update(
            verdict="ABSENT", in_installed_vllm=False, main_lookup="UNKNOWN", pr_lookup="UNKNOWN"
        )
        errors = validate_support_card(self.scan, self.contract)
        self.assertTrue(any("must stay UNKNOWN_UPSTREAM" in error for error in errors), errors)

    def test_absent_with_both_lookups_done_is_accepted(self):
        self.entry().update(
            verdict="ABSENT", in_installed_vllm=False, main_lookup="NOT_FOUND", pr_lookup="NOT_FOUND"
        )
        self.assertEqual(validate_support_card(self.scan, self.contract), [])

    def test_pr_pending_must_cite_pull_requests(self):
        self.entry().update(verdict="PR_PENDING", in_installed_vllm=False)
        errors = validate_support_card(self.scan, self.contract)
        self.assertTrue(any("must cite the pull requests" in error for error in errors), errors)

    def test_a_verdict_must_agree_with_the_installed_registry(self):
        self.entry()["verdict"] = "KUNLUN_OOT"
        errors = validate_support_card(self.scan, self.contract)
        self.assertTrue(any("registry does not list it" in error for error in errors), errors)

    def test_a_verdict_without_its_meaning_is_rejected(self):
        self.entry().pop("meaning")
        errors = validate_support_card(self.scan, self.contract)
        self.assertTrue(any("invites the wrong next action" in error for error in errors), errors)

    def test_the_scanned_pod_must_be_recorded(self):
        self.scan.pop("scanned_in")
        errors = validate_support_card(self.scan, self.contract)
        self.assertTrue(any("scanned_in" in error for error in errors), errors)

    def test_the_installed_registry_contents_must_be_recorded(self):
        self.entry().pop("kunlun_oot_archs")
        errors = validate_support_card(self.scan, self.contract)
        self.assertTrue(any("recorded, not summarised" in error for error in errors), errors)


class CardRenderingTest(unittest.TestCase):
    def test_open_pull_requests_are_cited_with_numbers(self):
        scan = copy.deepcopy(SCAN)
        scan["results"][0].update(
            verdict="PR_PENDING",
            pull_requests=[{"number": 12345, "title": "Add FooForCausalLM", "url": "https://x/1"}],
        )
        card = render_card(scan, {"model": {"id": "Foo"}, "target": {}}, "pod-1")
        self.assertIn("#12345", card)
        self.assertIn("https://x/1", card)


class EnvironmentGateTest(unittest.TestCase):
    def test_a_scan_refuses_an_unproven_environment(self):
        """Registry contents from a runtime nobody validated describe nothing."""
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "model_request.yaml"
            request.write_text(
                yaml.safe_dump({"model": {"id": "X"}, "identity": {"architectures": ["XForCausalLM"]}}),
                encoding="utf-8",
            )
            status = Path(tmp) / "status.json"
            status.write_text(json.dumps({"state": "INSTALL_FAILED", "pod": "p"}), encoding="utf-8")
            argv = sys.argv
            sys.argv = [
                "scan", "--model-request", str(request), "--env-status", str(status),
                "--out", str(Path(tmp) / "out"),
            ]
            try:
                with self.assertRaises(ScanFailed) as ctx:
                    scan_main()
            finally:
                sys.argv = argv
            self.assertEqual(ctx.exception.state, "NEEDS_HUMAN")
            self.assertIn("INSTALL_FAILED", ctx.exception.reason)


if __name__ == "__main__":
    unittest.main()
