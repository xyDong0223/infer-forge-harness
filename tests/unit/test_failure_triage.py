"""MAT-006: what an attribution must have before it names a layer."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validators.triage_validator import validate_triage  # noqa: E402

CONTRACT_PATH = ROOT / "tasks" / "mat-006-failure-triage" / "task.yaml"

# The real Qwen3-8B triage: two vendor kernels fail in the server and accept the
# identical arguments outside it.
REPORT = {
    "state": "TRIAGE_READY",
    "failing_symbol": "kunlun_ops.speculative_attention",
    "error_text": "ValueError: Check 0 == ret failed, left operand=0, speculative_attention failed",
    "layer": "kunlun_ops_vendor",
    "verdict": "RUNTIME_STATE_DEPENDENT",
    "owner": "vendor_with_local_workaround",
    "captured_arguments": {
        "batch_num": 1024,
        "block_size": 16,
        "max_num_blocks_per_seq": 2048,
        "k_cache": {"shape": [31561, 8, 16, 128], "dtype": "torch.bfloat16"},
        "context_lens_cpu": {"min": 3, "max": 3},
    },
    "isolated_reproduction": {
        "attempted": True,
        "reproduced": False,
        "cases": 49,
        "passed": 47,
        "hard_constraints": [
            "context_lens must be int32",
            # float32 is not uniformly unimplemented: it failed at block_size 64
            # and passed at block_size 16, so the constraint is geometry-dependent.
            "dtype=float32 rejected at block_size 64, accepted at block_size 16",
        ],
    },
    "workaround": {
        "description": "route the qlen==1 decode path through patches/torch_paged_decode.py",
        "validated_in_server": True,
    },
}


class TriageValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        import yaml

        self.contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.report = copy.deepcopy(REPORT)

    def check(self) -> list[str]:
        return validate_triage(self.report, self.contract)

    def test_the_real_qwen3_triage_passes(self):
        self.assertEqual(self.check(), [])

    def test_an_attribution_without_a_reproduction_attempt_is_refused(self):
        self.report.pop("isolated_reproduction")
        errors = self.check()
        self.assertTrue(any("is a guess" in error for error in errors), errors)

    def test_a_vendor_gap_needs_the_isolated_call_to_fail_too(self):
        """Otherwise a ticket goes out that the vendor cannot reproduce."""
        self.report["verdict"] = "VENDOR_KERNEL_GAP"
        errors = self.check()
        self.assertTrue(any("RUNTIME_STATE_DEPENDENT" in error for error in errors), errors)

    def test_runtime_state_dependent_requires_isolation_to_pass(self):
        self.report["isolated_reproduction"]["reproduced"] = True
        errors = self.check()
        self.assertTrue(any("plain vendor gap" in error for error in errors), errors)

    def test_captured_arguments_are_mandatory(self):
        self.report.pop("captured_arguments")
        errors = self.check()
        self.assertTrue(any("cannot be reproduced by anyone else" in error for error in errors), errors)

    def test_a_workaround_validated_only_in_isolation_is_refused(self):
        """Isolation already passed here, so it proves nothing about the fix."""
        self.report["workaround"]["validated_in_server"] = False
        errors = self.check()
        self.assertTrue(any("validated in the server" in error for error in errors), errors)

    def test_a_binary_layer_cannot_be_owned_by_us(self):
        self.report["owner"] = "us"
        errors = self.check()
        self.assertTrue(any("ships as a binary" in error for error in errors), errors)

    def test_a_modifiable_layer_cannot_be_blamed_on_the_vendor(self):
        self.report.update(layer="vllm_kunlun_plugin", owner="vendor", verdict="PLUGIN_DEFECT")
        errors = self.check()
        self.assertTrue(any("modifiable here" in error for error in errors), errors)

    def test_the_error_text_must_be_quoted(self):
        self.report.pop("error_text")
        errors = self.check()
        self.assertTrue(any("what the runtime actually said" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
