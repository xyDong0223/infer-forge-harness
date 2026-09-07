"""MAT-007: what a replacement kernel must prove before it may stay."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validators.patch_validator import validate_patch_placement  # noqa: E402

CONTRACT_PATH = ROOT / "tasks" / "mat-007-patch-placement" / "task.yaml"

# The real 2026-09-07 placement of patches/torch_paged_decode.py.
REPORT = {
    "state": "PATCH_PLACED",
    "mechanism": "post_import_patch",
    "scope": {"routes_only": "qlen == 1"},
    "numerical": {
        "reference": "kunlun_ops.speculative_attention",
        "cases": 8,
        "min_cosine_similarity": 0.999998,
        "max_relative_error": 0.0074,
    },
    "server_validation": {"state": "DEPLOYMENT_READY", "health": 200},
    "reversibility": {
        "backup_path": "/opt/vllm_kunlun/lib/python3.10/site-packages/vllm_kunlun/__init__.py.kdp_backup",
        "runtime_switch": "KDP_DECODE_KERNEL",
        "reproduces_original_failure": True,
    },
    "limitations": ["shape-dynamic span requires --enforce-eager"],
}


class PatchPlacementValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        import yaml

        self.contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.report = copy.deepcopy(REPORT)

    def check(self) -> list[str]:
        return validate_patch_placement(self.report, self.contract)

    def test_the_real_placement_passes(self):
        self.assertEqual(self.check(), [])

    def test_the_contract_chooses_exactly_one_mechanism_with_reasons(self):
        mechanisms = self.contract["checks"]["mechanisms"]
        self.assertEqual(sum(1 for m in mechanisms if m.get("chosen")), 1)
        self.assertTrue(all(m.get("why") for m in mechanisms))

    def test_a_patch_placed_by_another_mechanism_is_refused(self):
        self.report["mechanism"] = "install_time_file_edit"
        errors = self.check()
        self.assertTrue(any("the contract chose" in error for error in errors), errors)

    def test_replacing_a_kernel_without_comparing_it_is_refused(self):
        self.report["numerical"].pop("min_cosine_similarity")
        errors = self.check()
        self.assertTrue(any("a rewrite, not a patch" in error for error in errors), errors)

    def test_disagreement_with_the_reference_kernel_is_refused(self):
        self.report["numerical"]["min_cosine_similarity"] = 0.98
        errors = self.check()
        self.assertTrue(any("below the contract" in error for error in errors), errors)

    def test_isolation_alone_cannot_validate_the_patch(self):
        self.report["server_validation"] = {"state": "ISOLATED_PASS"}
        errors = self.check()
        self.assertTrue(any("service proof" in error for error in errors), errors)

    def test_a_patch_that_cannot_be_turned_off_is_refused(self):
        self.report["reversibility"]["reproduces_original_failure"] = False
        errors = self.check()
        self.assertTrue(any("reproduce the original failure" in error for error in errors), errors)

    def test_widening_the_routing_scope_is_refused(self):
        """qlen > 1 has no fallback implementation, so it must keep the vendor kernel."""
        self.report["scope"]["routes_only"] = "all decode"
        errors = self.check()
        self.assertTrue(any("silently changes results" in error for error in errors), errors)

    def test_a_patch_without_a_stated_limitation_is_refused(self):
        self.report["limitations"] = []
        errors = self.check()
        self.assertTrue(any("nobody can check" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
