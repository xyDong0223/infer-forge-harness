"""MAT-027 / MAT-028: the two gates that run before anything expensive.

The fixtures are the real GLM-5.2 answers. The drift report is the measured output
of importing every `vllm_kunlun` module against the installed engine -- the check
that, run at the start instead of at the end, would have replaced six full 707 GiB
loads. The bring-up report is the shape the same six defects present in when the
weights are dummy and the model is four layers deep.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tools.probe.toy_bringup_probe import STAGES  # noqa: E402
from tools.scan_runtime_drift import render_card as render_drift_card  # noqa: E402
from tools.toy_bringup import render_card as render_bringup_card  # noqa: E402
from validators.bringup_validator import validate_bringup_report  # noqa: E402
from validators.drift_validator import validate_drift_report  # noqa: E402

DRIFT_CONTRACT = yaml.safe_load(
    (ROOT / "tasks" / "mat-027-runtime-drift" / "task.yaml").read_text(encoding="utf-8")
)
BRINGUP_CONTRACT = yaml.safe_load(
    (ROOT / "tasks" / "mat-028-toy-bringup" / "task.yaml").read_text(encoding="utf-8")
)

# Measured in the P800 dev pod, 2026-09-09.
DRIFT = {
    "state": "DRIFT_FOUND",
    "engine": {"name": "vllm", "version": "0.25.1", "path": "/opt/.../site-packages/vllm"},
    "plugin": {"name": "vllm_kunlun", "version": "0.25.1.dev0", "modules_scanned": 214},
    "failures": [
        {
            "module": "vllm_kunlun.models.gpt_oss",
            "error_type": "ImportError",
            "message": "cannot import name 'cdiv' from 'vllm.utils'",
            "resolution": {"kind": "MISSING_SYMBOL", "symbol": "cdiv",
                           "old_module": "vllm.utils", "found_in_package": "vllm",
                           "candidates": ["vllm.utils.math_utils"],
                           "gone_upstream": False},
        },
        {
            "module": "vllm_kunlun.v1.sample.spec_decode.eagle",
            "error_type": "ModuleNotFoundError",
            "message": "No module named 'vllm.v1.attention.backends.tree_attn'",
            "resolution": {"kind": "MISSING_MODULE",
                           "module": "vllm.v1.attention.backends.tree_attn",
                           "candidates": [], "gone_upstream": True},
        },
    ],
}

BRINGUP = {
    "state": "BRINGUP_PASS",
    "stage": "DECODE_OK",
    "stages_passed": list(STAGES),
    "complete": True,
    "decode": {"token_ids": [1, 2, 3, 4], "count": 4},
    "config": {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": {"real": 78, "toy": 5},
        "n_routed_experts": {"real": 256, "toy": 8},
        "quantization": "compressed-tensors",
        "kept_dimensions": {"hidden_size": 6144, "qk_nope_head_dim": 192,
                            "qk_rope_head_dim": 64, "v_head_dim": 256,
                            "kv_lora_rank": 512, "index_topk": 2048},
    },
}


class DriftReportTest(unittest.TestCase):
    def test_the_measured_report_passes(self):
        self.assertEqual(validate_drift_report(DRIFT, DRIFT_CONTRACT), [])

    def test_a_failure_without_a_resolution_is_rejected(self):
        """A bare failure list makes the next Task guess where the symbol went."""
        report = copy.deepcopy(DRIFT)
        del report["failures"][0]["resolution"]
        errors = validate_drift_report(report, DRIFT_CONTRACT)
        self.assertTrue(any("no resolution" in error for error in errors), errors)

    def test_an_unresolvable_symbol_must_say_it_is_gone(self):
        """Empty candidates and no `gone_upstream` leaves an unanswered question.

        The probe indexed the whole engine, so it can distinguish "repoint this
        import" from "reimplement this behaviour". Leaving the list empty without
        saying which one it is pushes that judgement onto the reader.
        """
        report = copy.deepcopy(DRIFT)
        del report["failures"][1]["resolution"]["gone_upstream"]
        errors = validate_drift_report(report, DRIFT_CONTRACT)
        self.assertTrue(any("gone_upstream" in error for error in errors), errors)

    def test_an_empty_scan_proves_nothing(self):
        report = copy.deepcopy(DRIFT)
        report["plugin"]["modules_scanned"] = 0
        errors = validate_drift_report(report, DRIFT_CONTRACT)
        self.assertTrue(any("modules_scanned" in error for error in errors), errors)

    def test_state_and_failures_may_not_disagree(self):
        report = copy.deepcopy(DRIFT)
        report["state"] = "DRIFT_CLEAR"
        errors = validate_drift_report(report, DRIFT_CONTRACT)
        self.assertTrue(any("DRIFT_CLEAR" in error for error in errors), errors)

    def test_the_card_names_where_each_symbol_went(self):
        card = render_drift_card(DRIFT, "pod-0")
        self.assertIn("vllm.utils.math_utils", card)
        self.assertIn("DRIFT_FOUND", card)
        # The limitation that sends the reader to the next gate.
        self.assertIn("Import success is not correctness", card)


class DriftResolutionTest(unittest.TestCase):
    """Where a missing name is looked for, checked without a pod.

    Both directions were wrong once. Searching every package at once matched a
    missing `vllm_kunlun.ops.quantization` to `vllm.config.quantization` on the tail
    name alone; searching only the named package turned a correct answer for
    `ParallelLMHead` into "gone", because the plugin re-exports what the engine
    defines. Nearest first, then widen.
    """

    INDEXES = {
        "vllm": {"cdiv": ["vllm.utils.math_utils"],
                 "ParallelLMHead": ["vllm.model_executor.layers.vocab_parallel_embedding"]},
        "vllm_kunlun": {},
    }
    MODULES = {
        "vllm": {"vllm.config.quantization", "vllm.utils.math_utils"},
        "vllm_kunlun": {"vllm_kunlun.quantization", "vllm_kunlun.ops.linear"},
    }

    def _resolve(self, message: str) -> dict:
        from tools.probe.runtime_drift_probe import _resolve

        return _resolve(message, self.INDEXES, self.MODULES)

    def test_a_moved_engine_symbol_is_located(self):
        resolution = self._resolve("cannot import name 'cdiv' from 'vllm.utils'")
        self.assertEqual(resolution["candidates"], ["vllm.utils.math_utils"])
        self.assertFalse(resolution["gone_upstream"])

    def test_a_symbol_the_plugin_only_re_exports_falls_back_to_the_engine(self):
        resolution = self._resolve(
            "cannot import name 'ParallelLMHead' from 'vllm_kunlun.ops.vocab_parallel_embedding'"
        )
        self.assertEqual(resolution["found_in_package"], "vllm")
        self.assertFalse(resolution["gone_upstream"])

    def test_a_missing_plugin_module_is_not_matched_to_the_engine(self):
        resolution = self._resolve("No module named 'vllm_kunlun.ops.quantization'")
        self.assertEqual(resolution["found_in_package"], "vllm_kunlun")
        self.assertNotIn("vllm.config.quantization", resolution["candidates"])

    def test_a_name_nothing_defines_is_reported_as_gone(self):
        resolution = self._resolve("cannot import name 'skip_code' from 'torch._C'")
        self.assertTrue(resolution["gone_upstream"])
        self.assertEqual(resolution["candidates"], [])

    def test_a_failure_that_is_not_drift_is_left_unclassified(self):
        """Repointing an import would be the wrong response to this one."""
        resolution = self._resolve("there's already a kernel registered from python")
        self.assertEqual(resolution["kind"], "UNCLASSIFIED")


class ToyBringupReportTest(unittest.TestCase):
    def test_the_passing_report_passes(self):
        self.assertEqual(validate_bringup_report(BRINGUP, BRINGUP_CONTRACT), [])

    def test_one_decoded_token_is_not_a_decode(self):
        """One token proves prefill; the indexer and paged attention differ later."""
        report = copy.deepcopy(BRINGUP)
        report["decode"] = {"token_ids": [1], "count": 1}
        errors = validate_bringup_report(report, BRINGUP_CONTRACT)
        self.assertTrue(any("more than one decoded token" in error for error in errors), errors)

    def test_shrinking_a_kernel_selecting_dimension_is_rejected(self):
        """A toy run on different head dims describes a different code path."""
        report = copy.deepcopy(BRINGUP)
        report["config"]["kept_dimensions"]["qk_rope_head_dim"] = 32
        errors = validate_bringup_report(
            report, BRINGUP_CONTRACT, {"qk_rope_head_dim": 64, "hidden_size": 6144}
        )
        self.assertTrue(any("qk_rope_head_dim" in error for error in errors), errors)

    def test_an_incomplete_run_must_carry_its_error(self):
        report = copy.deepcopy(BRINGUP)
        report.update({"state": "BRINGUP_BLOCKED", "complete": False,
                       "stage": "ENGINE_CONSTRUCTED",
                       "stages_passed": ["CONFIG_DERIVED", "ENGINE_CONSTRUCTED"]})
        report.pop("decode")
        errors = validate_bringup_report(report, BRINGUP_CONTRACT)
        self.assertTrue(any("error that stopped it" in error for error in errors), errors)

    def test_stages_must_be_a_prefix(self):
        """Skipping a stage would let a later pass hide an earlier failure."""
        report = copy.deepcopy(BRINGUP)
        report["stages_passed"] = ["CONFIG_DERIVED", "PREFILL_OK"]
        report["complete"] = False
        report["error"] = {"type": "X", "message": "y"}
        errors = validate_bringup_report(report, BRINGUP_CONTRACT)
        self.assertTrue(any("prefix" in error for error in errors), errors)

    def test_the_card_states_what_a_pass_does_not_mean(self):
        card = render_bringup_card(BRINGUP, "GLM-5.2-W8A8-INT8-Dynamic", "pod-0")
        self.assertIn("5 of 78 layers", card)
        self.assertIn("Does not mean the numbers are right", card)


class ToyConfigDerivationTest(unittest.TestCase):
    """The config rules, checked without an engine: depth floor and list truncation."""

    def _derive(self, source_config: dict) -> dict:
        import json
        import tempfile

        from tools.probe.toy_bringup_probe import derive_config

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "real"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
            target = Path(tmp) / "toy"
            derive_config(str(source), str(target), layers=4, experts=8)
            return json.loads((target / "config.json").read_text(encoding="utf-8"))

    def test_per_layer_lists_are_truncated_to_the_new_depth(self):
        """78 layer types on a 5-layer model fails in the config loader."""
        derived = self._derive({
            "num_hidden_layers": 78,
            "first_k_dense_replace": 3,
            "indexer_types": ["full"] * 78,
            "mlp_layer_types": ["dense"] * 78,
        })
        self.assertEqual(derived["num_hidden_layers"], 5)
        self.assertEqual(len(derived["indexer_types"]), 5)
        self.assertEqual(len(derived["mlp_layer_types"]), 5)

    def test_depth_clears_the_dense_prefix_so_one_moe_layer_runs(self):
        """The MoE layer is where most of the contracts live."""
        derived = self._derive({"num_hidden_layers": 78, "first_k_dense_replace": 3})
        self.assertGreater(derived["num_hidden_layers"], 3)

    def test_kernel_selecting_dimensions_are_untouched(self):
        derived = self._derive({
            "num_hidden_layers": 78,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "index_topk": 2048,
            "quantization_config": {"quant_method": "compressed-tensors"},
        })
        self.assertEqual(derived["qk_rope_head_dim"], 64)
        self.assertEqual(derived["kv_lora_rank"], 512)
        self.assertEqual(derived["index_topk"], 2048)
        self.assertEqual(derived["quantization_config"]["quant_method"], "compressed-tensors")

    def test_mtp_is_excluded_so_one_report_carries_one_failure(self):
        derived = self._derive({"num_hidden_layers": 78, "num_nextn_predict_layers": 1})
        self.assertEqual(derived["num_nextn_predict_layers"], 0)

    def test_expert_count_stays_meaningful_for_grouped_topk(self):
        derived = self._derive({
            "num_hidden_layers": 78, "n_routed_experts": 256,
            "num_experts_per_tok": 8, "n_group": 1,
        })
        self.assertGreaterEqual(derived["n_routed_experts"], 8)
        self.assertLess(derived["n_routed_experts"], 256)


if __name__ == "__main__":
    unittest.main()
