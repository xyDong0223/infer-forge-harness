"""Memory budget Tool and Validator checks.

The samples are the real `xpu_smi -m` and capacity-planner output captured from
the MiniMax-M2.5 deployment on 2026-09-07, so a change in either parser shows up
as a diff against observed hardware rather than against invented numbers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.kunlun_p800.adapter import KunlunP800Adapter  # noqa: E402
from tools.memory_budget import (  # noqa: E402
    BudgetError,
    detect_active_ranks,
    load_device_spec,
    reconcile,
    reconcile_deployment,
    renumber_cards,
    resolve_analyzer,
)
from validators.memory_validator import validate_memory_budget  # noqa: E402

XPU_SMI_SAMPLE = (
    '00000000:03:00.0 0 0 02K15K624CV00304 36 0 0 0 86 1450 1450 1450 1450 1450 1450 '
    '2 96 90440 98304 0 1.0:2.6:1.39.1.7 "P800 OAM" 0 0 0 0 0 0 0 0 0 0\n'
    '00000000:05:00.0 1 1 02K15K624CV0030V 39 0 0 0 86 1450 1450 1450 1450 1450 1450 '
    '2 96 90440 98304 0 1.0:2.6:1.39.1.7 "P800 OAM" 0 0 0 0 0 0 0 0 0 0\n'
)

# The Qwen3.8 TP=1 pod on 2026-09-09: the service runs on device index 2, the
# other seven cards are idle after the previous deployment was stopped.
TP1_XPU_SMI_SAMPLE = "".join(
    (
        f'00000000:0{3 + index * 32:X}:00.0 {index} {index} 02K15K624CV0030{index:X} '
        '36 0 0 0 86 1450 1450 1450 1450 1450 1450 '
        f'2 96 {88986 if index == 2 else 0} 98304 0 1.0:2.6:1.39.1.7 "P800 OAM" '
        '0 0 0 0 0 0 0 0 0 0\n'
    )
    for index in range(8)
)

PLANNER_REPORT = {
    "framework": "vllm",
    "memory_breakdown": {
        "rank": 0,
        "model_weights_gib": 27.18,
        "kv_pool_gib": 58.02,
        "cuda_graph_gib": 0.06,
        "framework_overhead_gib": 0.0,
        "other_gib": 3.06,
    },
    "vllm": {"gpu_kv_cache_tokens": 1962496, "max_model_len": 196608, "maximum_concurrency": 9.98},
}


class XpuSmiParsingTest(unittest.TestCase):
    def test_memory_columns_are_read_by_documented_position(self):
        cards = KunlunP800Adapter.parse_xpu_smi(XPU_SMI_SAMPLE)
        self.assertEqual([card["index"] for card in cards], [0, 1])
        self.assertEqual(cards[0]["used_mib"], 90440)
        self.assertEqual(cards[0]["total_mib"], 98304)
        self.assertEqual(cards[0]["free_mib"], 7864)

    def test_l3_pair_is_not_mistaken_for_device_memory(self):
        """Columns 15/16 are L3 (2 MiB / 96 MiB) and sit right before memory."""
        card = KunlunP800Adapter.parse_xpu_smi(XPU_SMI_SAMPLE)[0]
        self.assertNotEqual(card["total_mib"], 96)
        self.assertNotEqual(card["used_mib"], 2)

    def test_non_data_lines_are_ignored(self):
        self.assertEqual(KunlunP800Adapter.parse_xpu_smi("XPU-SMI\n\nnot a row\n"), [])

    def test_csv_matches_the_nvidia_smi_shape_the_analyzer_parses(self):
        cards = KunlunP800Adapter.parse_xpu_smi(XPU_SMI_SAMPLE)
        self.assertEqual(
            KunlunP800Adapter.as_nvidia_smi_csv(cards),
            "0, 90440 MiB, 7864 MiB\n1, 90440 MiB, 7864 MiB\n",
        )


class ReconciliationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cards = KunlunP800Adapter.parse_xpu_smi(XPU_SMI_SAMPLE)
        self.spec = load_device_spec("p800")

    def test_catalog_spec_matches_the_cards(self):
        self.assertEqual(self.spec["hbm_mib"], self.cards[0]["total_mib"])

    def test_unknown_device_is_refused(self):
        with self.assertRaises(BudgetError):
            load_device_spec("h100")

    def test_remainder_is_recomputed_from_the_device_counters(self):
        budget = reconcile(PLANNER_REPORT, self.cards, self.spec)
        self.assertEqual(budget["measured_used_mib"], 90440)
        self.assertEqual(budget["unattributed_mib"], 90440 - 87306)
        self.assertTrue(budget["reconciled"])
        self.assertEqual(budget["card_spread_mib"], 0)
        self.assertEqual(budget["utilization_pct"], 92.0)

    def test_a_log_from_another_process_fails_to_reconcile(self):
        stale = {**PLANNER_REPORT, "memory_breakdown": {**PLANNER_REPORT["memory_breakdown"], "other_gib": 20.0}}
        budget = reconcile(stale, self.cards, self.spec)
        self.assertFalse(budget["reconciled"])

    def test_missing_rank_is_an_error(self):
        with self.assertRaises(BudgetError):
            reconcile(PLANNER_REPORT, self.cards, self.spec, rank=7)

    def test_analyzer_must_be_located_explicitly(self):
        with self.assertRaises(BudgetError):
            resolve_analyzer("/nonexistent/skills/clone")


class MemoryValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        cards = KunlunP800Adapter.parse_xpu_smi(XPU_SMI_SAMPLE)
        self.budget = reconcile(PLANNER_REPORT, cards, load_device_spec("p800"))
        self.budget["evidence"] = {"xpu_smi": "memory/xpu_smi.csv"}

    def contract_thresholds(self) -> dict:
        import yaml

        contract = yaml.safe_load(
            (ROOT / "tasks" / "mem-001-memory-budget" / "task.yaml").read_text(encoding="utf-8")
        )
        return (contract.get("checks") or {}).get("thresholds") or {}

    def test_the_contract_thresholds_accept_the_observed_minimax_budget(self):
        """The Task's own thresholds must pass the deployment they were written for."""
        self.assertEqual(validate_memory_budget(self.budget, self.contract_thresholds()), [])

    def test_tightening_the_contract_tightens_the_verdict(self):
        thresholds = {**self.contract_thresholds(), "max_utilization_pct": 90}
        errors = validate_memory_budget(self.budget, thresholds)
        self.assertTrue(any("utilization" in error for error in errors), errors)

    def test_observed_minimax_budget_passes(self):
        self.assertEqual(validate_memory_budget(self.budget), [])

    def test_snapshot_evidence_is_required(self):
        self.budget.pop("evidence")
        self.assertIn(
            "evidence.xpu_smi must record the device snapshot path",
            validate_memory_budget(self.budget),
        )

    def test_zero_remainder_is_rejected_as_implausible(self):
        self.budget["unattributed_mib"] = 0
        errors = validate_memory_budget(self.budget)
        self.assertTrue(any("different processes" in error for error in errors))

    def test_thresholds_are_enforced(self):
        errors = validate_memory_budget(
            self.budget,
            {
                "min_free_mib": 16384,
                "max_utilization_pct": 90,
                "min_kv_cache_tokens": 4_000_000,
                "max_card_spread_mib": 0,
            },
        )
        self.assertEqual(len(errors), 3, errors)
        self.assertTrue(any("headroom" in error for error in errors))
        self.assertTrue(any("utilization" in error for error in errors))
        self.assertTrue(any("KV pool holds" in error for error in errors))


class ActiveRankScopeTest(unittest.TestCase):
    """The Qwen3.8 TP=1 incident: scope by usage, not by the rank argument.

    Reconciling card 0 of that pod read an idle card, produced a negative
    remainder, and failed closed on a healthy deployment. These tests pin the
    corrected behaviour: idle cards are context, active ranks are scope.
    """

    def setUp(self) -> None:
        self.cards = KunlunP800Adapter.parse_xpu_smi(TP1_XPU_SMI_SAMPLE)
        self.spec = load_device_spec("p800")
        self.tp1_log = (
            "INFO [vllm] Initializing engine (tp=1)\n"
            "tensor_parallel_size=1\n"
            "Model loading took 28.5 GiB\n"
            "Available KV cache memory: 49.61 GiB\n"
        )

    def test_active_rank_is_detected_by_usage_not_by_position(self):
        scope = detect_active_ranks(self.cards, self.tp1_log)
        self.assertEqual([card["index"] for card in scope["ranks"]], [2])
        self.assertEqual(scope["tp_size"], 1)
        self.assertEqual(scope["warnings"], [])

    def test_an_all_idle_snapshot_is_an_error_not_a_zero_budget(self):
        idle = [dict(card, used_mib=0, free_mib=card["total_mib"]) for card in self.cards]
        with self.assertRaises(BudgetError):
            detect_active_ranks(idle, self.tp1_log)

    def test_active_count_disagreeing_with_the_log_declares_a_warning(self):
        two_busy = [dict(card, used_mib=88986, free_mib=9318) for card in self.cards[:2]]
        scope = detect_active_ranks(two_busy, self.tp1_log)
        self.assertEqual(len(scope["ranks"]), 2)
        self.assertTrue(any("co-tenant" in warning for warning in scope["warnings"]))

    def test_reconciled_budget_uses_the_active_card_not_card_zero(self):
        # 88986 MiB used, log attributes 80118 MiB: the remainder is real memory
        # held by driver/runtime/allocator, not the -80118 of the incident.
        report = {
            "memory_breakdown": {
                "rank": 0,
                "model_weights_gib": 28.51,
                "kv_pool_gib": 49.61,
                "cuda_graph_gib": 0.12,
                "framework_overhead_gib": 0.0,
                "other_gib": 8.66,
            },
            "vllm": {"gpu_kv_cache_tokens": 754392, "max_model_len": 32768},
        }
        scope = detect_active_ranks(self.cards, self.tp1_log)
        budget = reconcile_deployment(report, self.cards, self.spec, scope)
        self.assertEqual(budget["ranks"], [2])
        self.assertEqual(budget["measured_used_mib"], 88986)
        self.assertEqual(budget["unattributed_mib"], 88986 - 80118)
        self.assertTrue(budget["reconciled"])
        # Idle cards stay in the report as context but never enter the spread.
        self.assertEqual(budget["cards"], 8)
        self.assertEqual(budget["card_spread_mib"], 0)
        self.assertEqual(budget["per_rank"][0]["rank"], 2)

    def test_analyzer_only_sees_the_active_cards_renumbered(self):
        scope = detect_active_ranks(self.cards, self.tp1_log)
        active_csv = KunlunP800Adapter.as_nvidia_smi_csv(renumber_cards(scope["ranks"]))
        self.assertEqual(active_csv, "0, 88986 MiB, 9318 MiB\n")

    def test_worst_rank_fails_the_aggregate(self):
        report = {
            "memory_breakdown": {
                "rank": 0,
                "model_weights_gib": 28.51,
                "kv_pool_gib": 49.61,
                "cuda_graph_gib": 0.12,
                "framework_overhead_gib": 0.0,
                "other_gib": 30.0,
            },
        }
        busy = [dict(card, used_mib=88986, free_mib=9318) for card in self.cards[:2]]
        scope = detect_active_ranks(busy, self.tp1_log)
        budget = reconcile_deployment(report, busy, self.spec, scope)
        self.assertFalse(budget["reconciled"])
        self.assertEqual(budget["unattributed_mib"], min(r["unattributed_mib"] for r in budget["per_rank"]))


if __name__ == "__main__":
    unittest.main()
