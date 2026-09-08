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


if __name__ == "__main__":
    unittest.main()
