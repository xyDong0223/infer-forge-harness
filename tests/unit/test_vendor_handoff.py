"""MAT-020 acceptance: the package form, and the M3 findings it is built from.

The checks that matter here are the ones that reject. A handoff is the artifact that
leaves this project, so each rule below is paired with a case that must fail it —
otherwise the rule is decoration.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vendor_handoff import build_package, render_package  # noqa: E402
from validators.handoff_validator import validate_handoff_package  # noqa: E402

CONTRACT = yaml.safe_load(
    (REPO_ROOT / "tasks" / "mat-020-vendor-handoff" / "task.yaml").read_text(encoding="utf-8")
)
FINDINGS = yaml.safe_load(
    (REPO_ROOT / "tasks" / "mat-020-vendor-handoff" / "minimax-m3-findings.yaml").read_text(
        encoding="utf-8"
    )
)


class M3PackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.package = build_package(copy.deepcopy(FINDINGS))

    def test_the_real_package_passes(self) -> None:
        self.assertEqual(validate_handoff_package(self.package, CONTRACT), [])
        self.assertEqual(self.package["state"], "HANDOFF_READY")

    def test_owners_are_routed_not_lumped(self) -> None:
        # The reason the package form exists: these items go to different readers.
        self.assertIn("upstream", self.package["routing"])
        self.assertIn("us", self.package["routing"])
        self.assertEqual(self.package["routing"]["upstream"], ["M3-03"])

    def test_measured_and_predicted_stay_separable(self) -> None:
        self.assertEqual(self.package["observed"], 4)
        self.assertEqual(self.package["unreached"], 2)
        for finding in self.package["findings"]:
            if finding["kind"] == "unreached_dependency":
                self.assertNotIn("error_text", finding)
                self.assertTrue(finding["blocked_by"])

    def test_every_site_names_a_line(self) -> None:
        for finding in self.package["findings"]:
            self.assertIn(":", finding["site"], finding["id"])

    def test_stand_ins_are_labelled(self) -> None:
        for finding in self.package["findings"]:
            if finding.get("stand_in"):
                self.assertTrue(finding["stand_in_is_not_a_fix"], finding["id"])
        self.assertIn("a probe, not a fix", render_package(self.package))

    def test_the_markdown_leads_with_not_serving(self) -> None:
        rendered = render_package(self.package)
        self.assertIn("serves requests: **False**", rendered)
        self.assertIn("stopped at:", rendered)


class RejectionTest(unittest.TestCase):
    """Each rule, with the case it has to reject."""

    def package(self, mutate) -> dict:
        doc = copy.deepcopy(FINDINGS)
        mutate(doc)
        return build_package(doc)

    def assertRejects(self, mutate, fragment: str) -> None:
        errors = validate_handoff_package(self.package(mutate), CONTRACT)
        self.assertTrue(errors, f"nothing rejected, expected {fragment!r}")
        self.assertTrue(
            any(fragment in error for error in errors),
            f"expected {fragment!r} among {errors}",
        )

    def test_rejects_claiming_the_subject_serves(self) -> None:
        def mutate(doc):
            doc["subject_status"]["serves_requests"] = True

        self.assertRejects(mutate, "not a handoff but a deployment proof")

    def test_rejects_a_missing_stopping_point(self) -> None:
        def mutate(doc):
            doc["subject_status"].pop("stopped_at")

        self.assertRejects(mutate, "stopped_at is required")

    def test_rejects_an_error_text_on_something_never_reached(self) -> None:
        def mutate(doc):
            unreached = next(f for f in doc["findings"] if f["kind"] == "unreached_dependency")
            unreached["error_text"] = "RuntimeError: msa_sparse_attention failed"

        self.assertRejects(mutate, "cannot carry an error_text")

    def test_rejects_an_observed_failure_without_the_error(self) -> None:
        def mutate(doc):
            doc["findings"][0].pop("error_text")

        self.assertRejects(mutate, "must carry the error verbatim")

    def test_rejects_an_upstream_filing_without_generality(self) -> None:
        def mutate(doc):
            upstream = next(f for f in doc["findings"] if f["owner"] == "upstream")
            upstream.pop("generality")

        self.assertRejects(mutate, "must state who else it hits")

    def test_rejects_blaming_the_vendor_for_a_layer_we_can_edit(self) -> None:
        def mutate(doc):
            doc["findings"][0]["owner"] = "vendor"

        self.assertRejects(mutate, "the vendor cannot own it")

    def test_rejects_claiming_a_binary_layer_as_our_work(self) -> None:
        def mutate(doc):
            doc["findings"][0]["layer"] = "kunlun_ops_vendor"

        self.assertRejects(mutate, "not ours")

    def test_rejects_a_fix_with_nothing_to_check_it_against(self) -> None:
        def mutate(doc):
            doc["findings"][0].pop("verified_against")

        self.assertRejects(mutate, "verified_against is required")

    def test_rejects_a_site_without_a_line(self) -> None:
        def mutate(doc):
            doc["findings"][0]["site"] = "vllm/models/minimax_m3/nvidia/model.py"

        self.assertRejects(mutate, "must name a file and a line")

    def test_rejects_an_unlabelled_stand_in(self) -> None:
        def mutate(doc):
            doc["findings"][0].pop("stand_in_is_not_a_fix")

        self.assertRejects(mutate, "must be labelled as one")

    def test_rejects_an_empty_package(self) -> None:
        def mutate(doc):
            doc["findings"] = []

        self.assertRejects(mutate, "no findings is not a handoff")


if __name__ == "__main__":
    unittest.main()
