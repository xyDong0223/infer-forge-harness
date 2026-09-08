"""MAT-008: dynamic fan-out in the executor, and what an EXERCISED claim requires."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from runners.graph_runner import NODES, Unresolved, fan_out_plan  # noqa: E402
from tools.evaluate_capability import PROBES, aggregate, dimensions_for, probe_argv  # noqa: E402
from tools.journal import record  # noqa: E402
from validators.evaluation_validator import validate_evaluation  # noqa: E402

CONTRACT = yaml.safe_load(
    (ROOT / "tasks" / "mat-008-capability-evaluation" / "task.yaml").read_text(encoding="utf-8")
)
ENVIRONMENT = {"hardware": "P800", "stack_commit": "3ced109a"}


def match_payload(axes: list[dict]) -> dict:
    return {"state": "MATCH_READY", "runtime_verified": False, "axes": axes}


class DimensionSelectionTest(unittest.TestCase):
    def test_only_demanded_dimensions_are_selected(self):
        axes = [
            {"axis": "attention", "required": "gqa", "verdict": "PROVIDED_MODULE_ONLY"},
            {"axis": "quantization", "required": None, "verdict": "NOT_REQUIRED"},
            {"axis": "moe", "required": None, "verdict": "NOT_REQUIRED"},
            {"axis": "multimodal", "required": False, "verdict": "NOT_REQUIRED"},
        ]
        self.assertEqual(dimensions_for(match_payload(axes)), [])

    def test_a_quantized_moe_model_fans_out_over_both(self):
        axes = [
            {"axis": "quantization", "required": "compressed-tensors", "verdict": "PROVIDED"},
            {"axis": "moe", "required": 256, "verdict": "PROVIDED_MODULE_ONLY"},
        ]
        self.assertEqual(dimensions_for(match_payload(axes)), ["quantization", "moe"])

    def test_sliding_window_is_read_off_the_attention_axis(self):
        """There is no window operator on P800; it is a parameter of the attention
        kernels, so the dimension has to come from what attention requires."""
        axes = [{"axis": "attention", "required": "sliding_window",
                 "verdict": "PROVIDED_MODULE_ONLY"}]
        self.assertEqual(dimensions_for(match_payload(axes)), ["msa"])

    def test_block_sparse_is_a_different_dimension_from_a_window(self):
        """Both live on the attention axis and they are not the same capability: a
        window is a parameter on a dense kernel, block-sparse selection is three
        kernels and an index cache of its own. The cache-write op comes with it —
        selecting the right blocks out of a wrongly written index cache is worse than
        failing, so the read side and the write side are selected together."""
        axes = [{"axis": "attention", "required": "block_sparse",
                 "verdict": "PROVIDED_MODULE_ONLY"}]
        self.assertEqual(dimensions_for(match_payload(axes)),
                         ["block_sparse", "fused_qknorm_rope_insert"])
        self.assertIn("block_sparse", PROBES)
        self.assertIn("fused_qknorm_rope_insert", PROBES)


class BlockSparseReferenceTest(unittest.TestCase):
    """The probe's own reference arithmetic, on the CPU, with no accelerator."""

    def geometry(self):
        import torch

        from tools.probe.block_sparse_attention_probe import reference_block_scores

        torch.manual_seed(3)
        batch, heads, length, dim, page = 1, 2, 8, 4, 4
        case = {
            "batch": batch, "heads": heads, "kv_heads": 1, "dim": dim, "page": page,
            "context_len": length, "scale": 0.5,
            "q": torch.randn(batch, heads, dim),
            "k_cache": torch.randn(length // page, 1, page, dim),
            "v_cache": torch.randn(length // page, 1, page, dim),
            "block_tables": torch.arange(length // page, dtype=torch.int32).reshape(batch, -1),
        }
        return case, reference_block_scores

    def test_a_block_score_is_the_max_of_its_token_scores(self):
        import torch

        case, reference_block_scores = self.geometry()
        from tools.probe.block_sparse_attention_probe import token_scores

        per_token = token_scores(case)
        scores = reference_block_scores(case, 4, "max")
        self.assertEqual(tuple(scores.shape), (1, 2, 2))
        for head in range(2):
            for block in range(2):
                expected = per_token[0, head, block * 4:(block + 1) * 4].max()
                self.assertTrue(torch.allclose(scores[0, head, block], expected))

    def test_the_unscaled_variant_differs_by_the_scale(self):
        case, reference_block_scores = self.geometry()
        scaled = reference_block_scores(case, 4, "max")
        unscaled = reference_block_scores(case, 4, "max_unscaled")
        # Same ordering, different magnitude: which is why relative L2 can tell them
        # apart and cosine cannot.
        self.assertTrue(bool((scaled - unscaled * case["scale"]).abs().max() < 1e-5))

    def test_reserving_the_local_block_changes_the_selection(self):
        import torch

        from tools.probe.block_sparse_attention_probe import reference_topk, set_agreement

        # The last block scores worst, so a plain top-k drops it and a reserved local
        # block keeps it. If these two agreed, the probe could not tell them apart.
        scores = torch.tensor([[[5.0, 4.0, 3.0, 0.0]]])
        plain = reference_topk(scores, 2, 4, False)
        reserved = reference_topk(scores, 2, 4, True)
        self.assertEqual(sorted(plain[0, 0].tolist()), [0, 1])
        self.assertEqual(sorted(reserved[0, 0].tolist()), [0, 3])
        self.assertEqual(set_agreement(plain, reserved)["exact_set_match_fraction"], 0.0)
        self.assertEqual(set_agreement(plain, plain)["exact_set_match_fraction"], 1.0)


class FusedInsertContractTest(unittest.TestCase):
    """The write side of block_sparse: what its contract entry has to carry."""

    def setUp(self) -> None:
        from tools.evaluate_capability import SIDECARS

        self.entry = CONTRACT["checks"]["dimensions"]["fused_qknorm_rope_insert"]
        self.sidecars = SIDECARS

    def test_it_grades_the_file_that_would_be_loaded(self):
        # A copy of the stand-in can drift from the one a launch actually uses, so the
        # probe is handed the real path — the same reason the msa dimension does it.
        self.assertEqual(self.sidecars["fused_qknorm_rope_insert"]["--implementation"],
                         "patches/m3_fused_qknorm_rope_probe.py")
        self.assertTrue((ROOT / "patches" / "m3_fused_qknorm_rope_probe.py").exists())

    def test_it_is_a_sparse_layer_with_a_scattered_slot_mapping(self):
        geometry = self.entry["geometry"]
        self.assertGreater(geometry["index_heads"], 0,
                           "num_index_heads == 0 is the dense branch and writes no index cache")
        self.assertNotEqual(geometry["tokens"] % geometry["block_size"], 0,
                            "a token count that fills whole blocks hides offset mistakes")

    def test_the_cache_write_is_verified_by_reading_it_back(self):
        conventions = self.entry["conventions"]
        self.assertEqual(conventions["verified_by"], "readback_at_the_written_slots")
        self.assertEqual(conventions["cache_layout"],
                         "two_num_blocks_kv_heads_block_size_head_size")

    def test_probe_arguments_come_from_the_contract(self):
        class Args:
            model_path = None
            tensor = None
            tokens = None

        argv = probe_argv("fused_qknorm_rope_insert", Args(), self.entry)
        for flag, key in (("--index-heads", "index_heads"), ("--rotary-dim", "rotary_dim"),
                          ("--block-size", "block_size"), ("--tokens", "tokens")):
            self.assertEqual(argv[argv.index(flag) + 1], str(self.entry["geometry"][key]))


class BlockSparseContractTest(unittest.TestCase):
    """Invariants the geometry has to keep, or the dimension measures nothing."""

    def setUp(self) -> None:
        self.entry = CONTRACT["checks"]["dimensions"]["block_sparse"]
        self.geometry = self.entry["geometry"]

    def test_topk_leaves_blocks_unselected(self):
        blocks = self.geometry["context_len"] // self.geometry["block_size"]
        self.assertLess(
            self.geometry["topk"], blocks,
            "with topk >= blocks sparse attention is dense attention and the gate passes for free",
        )

    def test_index_heads_divide_the_main_heads(self):
        self.assertEqual(self.geometry["heads"] % self.geometry["index_heads"], 0)

    def test_the_measured_conventions_are_recorded(self):
        conventions = self.entry["conventions"]
        self.assertEqual(conventions["block_score_reduction"], "max_over_block_of_scaled_qk")
        # Measured: a plain top-k matched 0.375 of rows, reserving the local block
        # matched all of them. Assuming the wrong one silently changes the selection.
        self.assertTrue(conventions["topk_reserves_local_block"])
        self.assertEqual(conventions["index_cache_kv_heads"], 1)

    def test_the_gate_is_not_cosine_only(self):
        self.assertEqual(CONTRACT["checks"]["method"]["gate_metric"], "relative_l2")
        self.assertIn("max_relative_l2", self.entry)

    def test_blockers_say_what_breaks_and_where(self):
        for blocker in self.entry["known_blockers"]:
            self.assertTrue(len(blocker) > 40, blocker)

    def test_probe_arguments_come_from_the_contract(self):
        class Args:
            model_path = None
            tensor = None
            tokens = None

        argv = probe_argv("block_sparse", Args(), self.entry)
        for flag, key in (("--index-heads", "index_heads"), ("--heads", "heads"),
                          ("--topk", "topk"), ("--context-len", "context_len")):
            self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index(flag) + 1], str(self.geometry[key]))



class FanOutTest(unittest.TestCase):
    def context(self, artifacts: Path) -> dict:
        return {"subject": "MiniMax-M2.5", "artifacts": str(artifacts), "attempt": "graph",
                "pod": "dongxinyu03-pod", "weights": "/mnt/cluster/MiniMax-M2.5-W8A8-INT8-Dynamic",
                "environment_text": "hardware=P800"}

    def prepared(self, tmp: Path, axes: list[dict]) -> tuple[Path, Path]:
        journal = tmp / "journal.jsonl"
        bundle = tmp / "mat-003"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "capability_match.json").write_text(json.dumps(match_payload(axes)),
                                                      encoding="utf-8")
        record(journal, "CapabilityMatch", "MiniMax-M2.5", "MATCH_READY", bundle, ENVIRONMENT)
        return journal, bundle

    def test_one_child_per_dimension_plus_a_fan_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            journal, _ = self.prepared(tmp, [
                {"axis": "quantization", "required": "compressed-tensors", "verdict": "PROVIDED"},
                {"axis": "moe", "required": 256, "verdict": "PROVIDED_MODULE_ONLY"},
            ])
            artifacts = tmp / "mat-008"
            plan = fan_out_plan(NODES["capability_evaluation"], self.context(artifacts),
                                journal, ENVIRONMENT, artifacts)
            self.assertEqual(len(plan), 3)
            self.assertEqual([target.name for target, _ in plan[:2]], ["quantization", "moe"])
            for target, command in plan[:2]:
                self.assertIn("--dimension", command)
                self.assertIn(target.name, command)
                # Each child owns a directory, so the aggregate can cite the evidence
                # rather than replace it.
                self.assertIn(str(target), command)

    def test_the_fan_in_cites_every_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            journal, _ = self.prepared(tmp, [
                {"axis": "quantization", "required": "compressed-tensors", "verdict": "PROVIDED"},
                {"axis": "moe", "required": 256, "verdict": "PROVIDED_MODULE_ONLY"},
            ])
            artifacts = tmp / "mat-008"
            plan = fan_out_plan(NODES["capability_evaluation"], self.context(artifacts),
                                journal, ENVIRONMENT, artifacts)
            _, aggregate_command = plan[-1]
            self.assertIn("--aggregate", aggregate_command)
            self.assertEqual(aggregate_command.count("--child"), 2)

    def test_a_model_demanding_nothing_stops_instead_of_reporting_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            journal, _ = self.prepared(tmp, [
                {"axis": "quantization", "required": None, "verdict": "NOT_REQUIRED"},
            ])
            artifacts = tmp / "mat-008"
            with self.assertRaises(Unresolved):
                fan_out_plan(NODES["capability_evaluation"], self.context(artifacts),
                             journal, ENVIRONMENT, artifacts)


class AggregateTest(unittest.TestCase):
    def child(self, root: Path, dimension: str, state: str) -> Path:
        path = root / dimension
        path.mkdir(parents=True, exist_ok=True)
        (path / "capability_evaluation.json").write_text(
            json.dumps({"dimension": dimension, "state": state}), encoding="utf-8"
        )
        return path

    def test_all_exercised_is_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            children = [self.child(tmp, "quantization", "EXERCISED_PASS")]
            self.assertEqual(aggregate(children)["state"], "EVALUATION_PASS")

    def test_an_unimplemented_dimension_caps_the_node_at_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            children = [self.child(tmp, "quantization", "EXERCISED_PASS"),
                        self.child(tmp, "moe", "EVALUATION_UNIMPLEMENTED")]
            self.assertEqual(aggregate(children)["state"], "EVALUATION_PARTIAL")

    def test_one_failing_dimension_fails_the_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            children = [self.child(tmp, "quantization", "EXERCISED_FAIL"),
                        self.child(tmp, "moe", "EVALUATION_UNIMPLEMENTED")]
            self.assertEqual(aggregate(children)["state"], "EVALUATION_FAIL")

    def test_a_missing_child_report_is_an_error_not_an_omission(self):
        with tempfile.TemporaryDirectory() as tmp:
            children = [Path(tmp) / "quantization"]
            self.assertEqual(aggregate(children)["state"], "EVALUATION_FAIL")


def passing_report(**overrides) -> dict:
    report = {
        "dimension": "quantization",
        "state": "EXERCISED_PASS",
        "operators": ["_C::scaled_int8_quant", "_C::matmul"],
        "exercised_in": "dongxinyu03-pod",
        "threshold_source": "tasks/mat-008-capability-evaluation/task.yaml",
        "thresholds": {"min_cosine": 0.9999, "max_relative_l2": 0.01},
        "cases": [{"case": "kernel_vs_dequantized_reference", "reference": "float32 dequantized",
                   "cosine": 1.0, "relative_l2": 0.00166}],
        "control": {"case": "omit_scale_to_max_conversion", "relative_l2": 0.992,
                    "cosine": 0.9999998, "discriminates": True},
    }
    report.update(overrides)
    return report


class EvaluationValidatorTest(unittest.TestCase):
    def test_a_well_formed_pass_is_accepted(self):
        self.assertEqual(validate_evaluation(passing_report(), CONTRACT), [])

    def test_a_control_that_also_passed_blocks_the_claim(self):
        """The real bug this catches: the first version gated on cosine, and cosine is
        invariant to the uniform per-channel factor the 127 conversion supplies, so
        omitting it still scored 0.9999999."""
        report = passing_report()
        report["control"] = {**report["control"], "discriminates": False}
        errors = validate_evaluation(report, CONTRACT)
        self.assertTrue(any("blind to what it claims to check" in error for error in errors))

    def test_a_pass_without_a_control_is_rejected(self):
        report = passing_report()
        report.pop("control")
        self.assertTrue(any("negative control is required" in e
                            for e in validate_evaluation(report, CONTRACT)))

    def test_a_metric_outside_the_gate_cannot_be_a_pass(self):
        report = passing_report()
        report["cases"][0]["relative_l2"] = 0.5
        self.assertTrue(any("exceeds 0.01" in e for e in validate_evaluation(report, CONTRACT)))

    def test_the_gate_must_be_the_contract_s_own(self):
        report = passing_report(thresholds={"min_cosine": 0.9999, "max_relative_l2": 0.9})
        self.assertTrue(any("not the declared one" in e
                            for e in validate_evaluation(report, CONTRACT)))

    def test_an_unimplemented_dimension_is_valid_but_carries_no_cases(self):
        report = {"dimension": "moe", "state": "EVALUATION_UNIMPLEMENTED",
                  "reason": "no probe is registered for this dimension yet", "cases": []}
        self.assertEqual(validate_evaluation(report, CONTRACT), [])
        report["cases"] = [{"case": "x"}]
        self.assertTrue(any("contradicts its state" in e
                            for e in validate_evaluation(report, CONTRACT)))

    def test_a_blocker_must_name_who_it_affects_and_where_it_lives(self):
        """A path that cannot run must stay actionable rather than sit behind a pass for
        the paths that did."""
        report = passing_report(blockers=[{"path": "fused_moe(sigmoid)",
                                          "error": "TypeError: ..."}])
        errors = validate_evaluation(report, CONTRACT)
        self.assertTrue(any("missing affected" in e for e in errors))
        self.assertTrue(any("missing call_sites" in e for e in errors))
        report = passing_report(blockers=[{"path": "fused_moe(sigmoid)",
                                          "error": "TypeError: ...",
                                          "affected": ["sigmoid MoE"],
                                          "call_sites": ["_kunlun_ops.py:426"]}])
        self.assertEqual(validate_evaluation(report, CONTRACT), [])


class ContractTest(unittest.TestCase):
    def test_the_contract_forbids_a_cosine_only_gate(self):
        method = CONTRACT["checks"]["method"]
        self.assertTrue(method["forbid_cosine_only_gate"])
        self.assertEqual(method["gate_metric"], "relative_l2")

    def test_every_declared_dimension_has_thresholds(self):
        declared = CONTRACT["context"]["dimensions_declared"]
        for dimension in declared:
            with self.subTest(dimension=dimension):
                self.assertIn(dimension, CONTRACT["checks"]["dimensions"])

    def test_the_quantization_dimension_names_the_line_under_test(self):
        text = (ROOT / "tasks" / "mat-008-capability-evaluation" / "task.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("scale_mm.py", text)
        self.assertIn("127", text)

    def test_the_msa_geometry_makes_the_window_observable(self):
        """context_len <= window makes windowed and unwindowed decode the same
        computation, so the probe would pass while measuring nothing."""
        geometry = CONTRACT["checks"]["dimensions"]["msa"]["geometry"]
        self.assertGreater(geometry["context_len"], geometry["window"])

    def test_every_registered_probe_has_an_argument_mapping(self):
        class Args:
            model_path = "/mnt/cluster/whatever"
            tensor = None
            tokens = None

        for dimension in PROBES:
            with self.subTest(dimension=dimension):
                argv = probe_argv(dimension, Args(), CONTRACT["checks"]["dimensions"][dimension])
                self.assertIn("--max-relative-l2", argv)

    def test_the_msa_probe_takes_its_geometry_from_the_contract(self):
        class Args:
            model_path = None
            tensor = None
            tokens = None

        argv = probe_argv("msa", Args(), CONTRACT["checks"]["dimensions"]["msa"])
        geometry = CONTRACT["checks"]["dimensions"]["msa"]["geometry"]
        self.assertIn(str(geometry["window"]), argv)
        # No weights: the window lives in the kernel and the mask, not in a checkpoint.
        self.assertNotIn("--model-path", argv)

    def test_the_moe_geometry_straddles_the_preprocessing_switch(self):
        """fused_moe switches preprocessing at M*top_k > 768. If both token counts land
        on the same side, the two cases exercise one implementation twice."""
        geometry = CONTRACT["checks"]["dimensions"]["moe"]["geometry"]
        top_k = geometry["top_k"]
        self.assertLessEqual(geometry["tokens_below"] * top_k, 768)
        self.assertGreater(geometry["tokens_above"] * top_k, 768)

    def test_the_quantization_tensor_is_discovered_not_hardcoded(self):
        """A fixed name fails silently on the next model: M2.5 has
        model.layers.0.self_attn.q_proj, M3 nests it under language_model."""
        self.assertEqual(CONTRACT["checks"]["dimensions"]["quantization"]["tensor"], "auto")

    def test_the_moe_dimension_records_the_sigmoid_blocker(self):
        moe = CONTRACT["checks"]["dimensions"]["moe"]
        self.assertTrue(moe["known_blockers"])
        self.assertIn("block_static", " ".join(moe["known_blockers"]))

    def test_the_moe_dimension_is_declared_tensor_parallel_only(self):
        text = (ROOT / "tasks" / "mat-008-capability-evaluation" / "task.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("TP first", text)


class WindowMaskTest(unittest.TestCase):
    """The mask is the whole sliding-window implementation, so it is worth pinning."""

    def mask(self, lengths, window):
        import torch

        from patches.torch_paged_decode import window_mask

        span = max(lengths)
        return window_mask(torch.arange(span), torch.tensor(lengths), window)

    def test_without_a_window_only_the_context_length_masks(self):
        mask = self.mask([3, 5], -1)
        self.assertEqual(mask[0].tolist(), [False, False, False, True, True])
        self.assertEqual(mask[1].tolist(), [False] * 5)

    def test_a_window_keeps_the_last_window_positions_inclusive(self):
        mask = self.mask([5], 2)
        # Positions 3 and 4 stay: the current token plus one before it.
        self.assertEqual(mask[0].tolist(), [True, True, True, False, False])

    def test_a_window_at_least_as_long_as_the_context_masks_nothing_extra(self):
        self.assertEqual(self.mask([4], 4)[0].tolist(), [False] * 4)
        self.assertEqual(self.mask([4], 99)[0].tolist(), [False] * 4)

    def test_the_current_position_is_never_masked_out(self):
        for length in (1, 2, 7):
            for window in (1, 2, 3):
                with self.subTest(length=length, window=window):
                    self.assertFalse(bool(self.mask([length], window)[0, length - 1]))

    def test_rows_are_masked_independently(self):
        mask = self.mask([6, 2], 2)
        self.assertEqual(mask[0].tolist(), [True, True, True, True, False, False])
        self.assertEqual(mask[1].tolist(), [False, False, True, True, True, True])


class FallbackRefusalTest(unittest.TestCase):
    def test_sinks_are_still_refused(self):
        """A per-head sink logit joins the softmax denominator and there is no sink
        model here to check an implementation against."""
        import torch

        from patches.torch_paged_decode import UnsupportedDecode, torch_paged_decode

        with self.assertRaises(UnsupportedDecode):
            torch_paged_decode(sink=torch.zeros(4), max_window_size=-1, qlen=1)

    def test_a_zero_window_is_refused_rather_than_guessed(self):
        from patches.torch_paged_decode import UnsupportedDecode, torch_paged_decode

        with self.assertRaises(UnsupportedDecode):
            torch_paged_decode(max_window_size=0, qlen=1)


class MoeReferenceTest(unittest.TestCase):
    """The reference is CPU-only, so its routing logic can be tested here."""

    def payload(self, scores_row, top_k):
        import torch

        from tools.probe.moe_layer_probe import torch_reference

        experts = len(scores_row)
        # One token, identity-ish experts: expert e scales the input by (e + 1), so the
        # output says which experts were selected.
        x = torch.ones(1, 2)
        w13 = torch.zeros(experts, 4, 2)
        w2 = torch.zeros(experts, 2, 2)
        for expert in range(experts):
            w13[expert, 0, 0] = 1.0
            w13[expert, 2, 0] = float(expert + 1)
            w2[expert, 0, 0] = 1.0
        logits = torch.log(torch.tensor([scores_row], dtype=torch.float32))
        return torch_reference(x, w13, w2, logits, top_k, True)

    def test_an_exact_tie_at_the_cut_is_reported(self):
        _, tied = self.payload([0.4, 0.2, 0.2, 0.2], 2)
        self.assertTrue(bool(tied[0]))

    def test_a_clear_ordering_is_not_reported_as_tied(self):
        _, tied = self.payload([0.4, 0.3, 0.2, 0.1], 2)
        self.assertFalse(bool(tied[0]))

    def test_a_tie_below_the_cut_does_not_count(self):
        """Ties only matter when they decide which experts are selected."""
        _, tied = self.payload([0.4, 0.3, 0.15, 0.15], 2)
        self.assertFalse(bool(tied[0]))

    def test_the_control_selects_the_least_scored_experts(self):
        import torch

        from tools.probe.moe_layer_probe import torch_reference

        x = torch.ones(1, 2)
        experts = 4
        w13 = torch.zeros(experts, 4, 2)
        w2 = torch.zeros(experts, 2, 2)
        for expert in range(experts):
            w13[expert, 0, 0] = 1.0
            w13[expert, 2, 0] = float(expert + 1)
            w2[expert, 0, 0] = 1.0
        logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32))
        top, _ = torch_reference(x, w13, w2, logits, 2, True, select="top")
        bottom, _ = torch_reference(x, w13, w2, logits, 2, True, select="bottom")
        self.assertFalse(torch.allclose(top, bottom))


if __name__ == "__main__":
    unittest.main()
