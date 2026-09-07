"""The graph executor: input resolution, edges, and refusing to guess."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.graph_runner import (  # noqa: E402
    MANUAL,
    NODES,
    Unresolved,
    load_workflow,
    node_task_type,
    resolve,
)
from tools.journal import record  # noqa: E402

WORKFLOW = ROOT / "workflows" / "model_adaptation.yaml"
ENVIRONMENT = {"hardware": "P800", "stack_commit": "3ced109a"}


class WorkflowShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = load_workflow(WORKFLOW)
        self.ids = {node["id"] for node in self.nodes}

    def test_every_edge_points_somewhere_meaningful(self):
        terminal = {"NEEDS_HUMAN", "REWORK", "DELIVERED", "PLANNED"}
        for node in self.nodes:
            for edge in ("on_success", "on_failure"):
                target = node.get(edge)
                if target is None:
                    continue
                with self.subTest(node=node["id"], edge=edge):
                    self.assertIn(target, self.ids | terminal)

    def test_a_service_proof_failure_leads_to_triage_not_to_a_dead_end(self):
        service = next(node for node in self.nodes if node["id"] == "kdp-001b-service-proof")
        self.assertEqual(service["on_failure"], "mat-006-failure-triage")

    def test_a_binary_layer_verdict_ends_in_a_delivery(self):
        triage = next(node for node in self.nodes if node["id"] == "mat-006-failure-triage")
        self.assertEqual(triage["on_failure"], "mat-020-vendor-handoff")
        handoff = next(node for node in self.nodes if node["id"] == "mat-020-vendor-handoff")
        self.assertEqual(handoff["on_success"], "DELIVERED")

    def test_every_wired_node_is_either_executable_or_declared_manual(self):
        """A node the walk cannot run must say so, not fail obscurely mid-graph."""
        for node in self.nodes:
            task_type = node_task_type(node)
            if task_type is None:
                continue
            with self.subTest(node=node["id"], task_type=task_type):
                self.assertTrue(task_type in NODES or task_type in MANUAL)

    def test_multi_step_nodes_explain_themselves(self):
        for task_type, guidance in MANUAL.items():
            with self.subTest(task_type=task_type):
                self.assertIn("runs_with", guidance)


class InputResolutionTest(unittest.TestCase):
    def context(self, artifacts: Path) -> dict:
        return {"subject": "Qwen3-8B", "artifacts": str(artifacts), "attempt": "graph",
                "environment_text": "hardware=P800"}

    def test_inputs_come_from_recorded_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            bundle = Path(tmp) / "mat-001"
            record(path, "ModelRequest", "Qwen3-8B", "INTAKE_READY", bundle, ENVIRONMENT)
            record(path, "EnvironmentProof", "Qwen3-8B", "ENVIRONMENT_READY", bundle, ENVIRONMENT)
            command = resolve(NODES["model_scan"], self.context(Path(tmp)), path, ENVIRONMENT)
            self.assertIn(str(bundle / "model_request.yaml"), command)
            self.assertIn(str(bundle / "status.json"), command)

    def test_a_missing_fact_stops_the_walk_instead_of_guessing_a_path(self):
        """The failure mode this replaces: a stale path answering for another model."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            with self.assertRaises(Unresolved):
                resolve(NODES["model_scan"], self.context(Path(tmp)), path, ENVIRONMENT)

    def test_a_fact_from_another_environment_does_not_satisfy_a_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            record(path, "ModelRequest", "Qwen3-8B", "INTAKE_READY", Path(tmp),
                   {**ENVIRONMENT, "stack_commit": "deadbeef"})
            record(path, "EnvironmentProof", "Qwen3-8B", "ENVIRONMENT_READY", Path(tmp),
                   {**ENVIRONMENT, "stack_commit": "deadbeef"})
            with self.assertRaises(Unresolved):
                resolve(NODES["model_scan"], self.context(Path(tmp)), path, ENVIRONMENT)

    def test_the_latest_fact_is_the_one_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            old, new = Path(tmp) / "old", Path(tmp) / "new"
            for bundle in (old, new):
                record(path, "ModelSupportCard", "Qwen3-8B", "SCAN_READY", bundle, ENVIRONMENT)
                record(path, "CapabilityMatch", "Qwen3-8B", "MATCH_READY", bundle, ENVIRONMENT)
            command = resolve(NODES["gap_classification"], self.context(Path(tmp)), path, ENVIRONMENT)
            self.assertIn(str(new / "model_support.json"), command)
            self.assertNotIn(str(old / "model_support.json"), command)


if __name__ == "__main__":
    unittest.main()
