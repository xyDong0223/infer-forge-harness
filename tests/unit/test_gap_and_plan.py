"""MAT-004, MAT-005 and the Journal: routing, provenance, and fact reuse."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tools.classify_gaps import classify  # noqa: E402
from tools.journal import fingerprint, latest, query, record  # noqa: E402
from tools.plan_deployment import plan  # noqa: E402
from validators.gap_validator import validate_gap_classification  # noqa: E402
from validators.plan_validator import validate_deployment_plan  # noqa: E402

GAP_CONTRACT = yaml.safe_load(
    (ROOT / "tasks" / "mat-004-gap-classification" / "task.yaml").read_text(encoding="utf-8")
)
PLAN_CONTRACT = yaml.safe_load(
    (ROOT / "tasks" / "mat-005-deployment-plan" / "task.yaml").read_text(encoding="utf-8")
)

# The real Qwen3-8B chain, as produced on 2026-09-07.
SUPPORT = {
    "state": "SCAN_READY",
    "scanned_in": "dongxinyu03-kdp001-qwen3-8b-7c969df894-hsz7d-0",
    "results": [{"architecture": "Qwen3ForCausalLM", "verdict": "UPSTREAM_GENERIC",
                 "meaning": "the installed vLLM implementation is used"}],
}
MATCH = {
    "state": "MATCH_READY",
    "matched_in": "dongxinyu03-kdp001-qwen3-8b-7c969df894-hsz7d-0",
    "verdict": "MATCHED",
    "model": {"id": "Qwen3-8B"},
    "axes": [{"axis": "attention", "required": "gqa", "evidence": "MODULE",
              "verdict": "PROVIDED_MODULE_ONLY"}],
}
REQUEST = {
    "model": {"id": "Qwen3-8B", "source": "/mnt/cluster/aiak-inference-test/Qwen3-8B",
              "pvc": "rapidfs-baige-v3-pvc", "revision": "e962c91b" + "0" * 56,
              "total_weight_bytes": 16381516776, "trust_remote_code_required": False},
    "identity": {"torch_dtype": "bfloat16", "max_position_embeddings": 40960,
                 "num_attention_heads": 32, "num_key_value_heads": 8},
    "target": {"hardware": "Kunlunxin-3-P800", "vllm_kunlun_commit": "3ced109a" + "f" * 32},
}
SPEC = {"hbm_mib": 98304}
PATCH = {"limitations": ["shape-dynamic span requires --enforce-eager"]}


class GapClassificationTest(unittest.TestCase):
    def test_qwen3_has_no_static_gap_and_routes_to_triage(self):
        report = classify(SUPPORT, MATCH)
        self.assertEqual(report["classification"], "NO_STATIC_GAP")
        self.assertIn("triage", report["next_action"])
        self.assertEqual(validate_gap_classification(report, GAP_CONTRACT), [])

    def test_an_absent_architecture_asks_for_a_model_implementation(self):
        support = copy.deepcopy(SUPPORT)
        support["results"][0].update(verdict="ABSENT")
        report = classify(support, MATCH)
        self.assertEqual(report["classification"], "REGISTRATION_MISSING")
        self.assertEqual(validate_gap_classification(report, GAP_CONTRACT), [])

    def test_a_version_lag_does_not_ask_for_a_model_implementation(self):
        """The mistake this class exists to prevent."""
        for verdict in ("MAIN_ONLY", "PR_PENDING"):
            support = copy.deepcopy(SUPPORT)
            support["results"][0].update(verdict=verdict)
            report = classify(support, MATCH)
            with self.subTest(verdict=verdict):
                self.assertEqual(report["classification"], "VERSION_LAG")
                self.assertIn("cherry-pick", report["next_action"])

    def test_a_missing_capability_blocks(self):
        match = copy.deepcopy(MATCH)
        match["axes"][0].update(evidence="ABSENT", verdict="NOT_PROVIDED")
        report = classify(SUPPORT, match)
        self.assertEqual(report["classification"], "CAPABILITY_MISSING")
        self.assertEqual(report["blocking"], ["attention"])
        self.assertEqual(validate_gap_classification(report, GAP_CONTRACT), [])

    def test_no_static_gap_cannot_hide_a_gap(self):
        report = classify(SUPPORT, MATCH)
        report["gaps"] = [{"class": "UNVERIFIED", "axis": "moe", "detail": "d",
                           "next_action": "a", "evidence": "e"}]
        errors = validate_gap_classification(report, GAP_CONTRACT)
        self.assertTrue(any("would hide them" in error for error in errors), errors)

    def test_a_classification_may_not_claim_a_runtime_result(self):
        report = classify(SUPPORT, MATCH)
        report["runtime_verified"] = True
        errors = validate_gap_classification(report, GAP_CONTRACT)
        self.assertTrue(any("executes\nnothing" in error or "executes" in error for error in errors), errors)


class DeploymentPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.classification = classify(SUPPORT, MATCH)

    def test_the_real_qwen3_plan_is_derived_and_valid(self):
        report = plan(REQUEST, self.classification, SPEC, PATCH)
        values = {item["parameter"]: item["value"] for item in report["parameters"]}
        self.assertEqual(values["dtype"], "bfloat16")
        self.assertEqual(values["tensor_parallel_size"], 1)
        self.assertEqual(values["max_model_len"], 40960)
        self.assertEqual(values["block_size"], 16)
        self.assertTrue(values["enforce_eager"])
        self.assertEqual(validate_deployment_plan(report, PLAN_CONTRACT, REQUEST), [])

    def test_eager_is_the_patch_s_doing_not_a_habit(self):
        report = plan(REQUEST, self.classification, SPEC, patch=None)
        values = {item["parameter"]: item["value"] for item in report["parameters"]}
        self.assertFalse(values["enforce_eager"])

    def test_a_large_checkpoint_is_sharded_across_cards(self):
        request = copy.deepcopy(REQUEST)
        request["model"]["total_weight_bytes"] = 230_274_936_728  # MiniMax-M2.5 W8A8
        report = plan(request, self.classification, SPEC, None)
        values = {item["parameter"]: item["value"] for item in report["parameters"]}
        self.assertGreaterEqual(values["tensor_parallel_size"], 4)
        self.assertEqual(validate_deployment_plan(report, PLAN_CONTRACT, request), [])

    def test_an_unsourced_parameter_is_refused(self):
        report = plan(REQUEST, self.classification, SPEC, PATCH)
        report["parameters"][0]["source"] = ""
        errors = validate_deployment_plan(report, PLAN_CONTRACT, REQUEST)
        self.assertTrue(any("guess in disguise" in error for error in errors), errors)

    def test_a_plan_contradicting_the_checkpoint_is_refused(self):
        report = plan(REQUEST, self.classification, SPEC, PATCH)
        for item in report["parameters"]:
            if item["parameter"] == "max_model_len":
                item["value"] = 131072
        errors = validate_deployment_plan(report, PLAN_CONTRACT, REQUEST)
        self.assertTrue(any("exceeds the checkpoint's limit" in error for error in errors), errors)

    def test_assumptions_must_be_declared(self):
        report = plan(REQUEST, self.classification, SPEC, PATCH)
        self.assertIn("gpu_memory_utilization", report["assumptions"])
        report["assumptions"] = []
        errors = validate_deployment_plan(report, PLAN_CONTRACT, REQUEST)
        self.assertTrue(any("assumptions disagree" in error for error in errors), errors)


class JournalTest(unittest.TestCase):
    ENVIRONMENT = {"hardware": "P800", "stack_commit": "3ced109a"}

    def test_a_fact_is_found_under_its_own_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            record(path, "ModelRequest", "Qwen3-8B", "INTAKE_READY", Path(tmp), self.ENVIRONMENT)
            hit = latest(path, "ModelRequest", subject="Qwen3-8B", environment=self.ENVIRONMENT)
            self.assertIsNotNone(hit)
            self.assertEqual(hit["state"], "INTAKE_READY")

    def test_a_fact_from_another_stack_commit_is_not_reused(self):
        """Otherwise a cached fact answers for an environment nobody validated."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            record(path, "ModelRequest", "Qwen3-8B", "INTAKE_READY", Path(tmp), self.ENVIRONMENT)
            other = {**self.ENVIRONMENT, "stack_commit": "deadbeef"}
            self.assertEqual(query(path, "ModelRequest", "Qwen3-8B", other), [])

    def test_the_most_recent_fact_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            record(path, "DeploymentProof", "Qwen3-8B", "SERVER_START_FAILED", Path(tmp), self.ENVIRONMENT)
            record(path, "DeploymentProof", "Qwen3-8B", "DEPLOYMENT_READY", Path(tmp), self.ENVIRONMENT)
            self.assertEqual(
                latest(path, "DeploymentProof", subject="Qwen3-8B")["state"], "DEPLOYMENT_READY"
            )

    def test_the_fingerprint_ignores_key_order(self):
        self.assertEqual(
            fingerprint({"a": "1", "b": "2"}), fingerprint({"b": "2", "a": "1"})
        )


if __name__ == "__main__":
    unittest.main()
