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
from tools.memory_budget import BudgetError, load_device_spec, reconcile, resolve_analyzer  # noqa: E402
from validators.memory_validator import validate_memory_budget  # noqa: E402

XPU_SMI_SAMPLE = (
    '00000000:03:00.0 0 0 02K15K624CV00304 36 0 0 0 86 1450 1450 1450 1450 1450 1450 '
    '2 96 90440 98304 0 1.0:2.6:1.39.1.7 "P800 OAM" 0 0 0 0 0 0 0 0 0 0\n'
    '00000000:05:00.0 1 1 02K15K624CV0030V 39 0 0 0 86 1450 1450 1450 1450 1450 1450 '
    '2 96 90440 98304 0 1.0:2.6:1.39.1.7 "P800 OAM" 0 0 0 0 0 0 0 0 0 0\n'
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


if __name__ == "__main__":
    unittest.main()
