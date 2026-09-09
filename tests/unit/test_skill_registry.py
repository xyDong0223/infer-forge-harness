import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.skill_registry import (  # noqa: E402
    SkillResolutionError,
    index_by_task_type,
    resolve,
    resolve_for_context,
    validate_tool_references,
)


class SkillRegistryTest(unittest.TestCase):
    def test_every_workflow_task_type_resolves_to_one_skill(self):
        index = index_by_task_type()
        expected = {
            "model_intake", "deployment_proof", "environment_proof", "service_proof",
            "model_scan", "capability_match", "gap_classification",
            "failure_triage", "capability_evaluation", "deployment_plan",
            "patch_placement", "accuracy_differential", "memory_budget",
            "api_conformance", "support_matrix", "vendor_handoff",
            "platform_kernel_correctness", "end_to_end_accuracy",
            "long_context_sparse_correctness",
            "model_bringup_loop",
        }
        self.assertEqual(set(index), expected)

    def test_skill_contains_agent_execution_contract(self):
        skill = resolve("capability_evaluation")
        self.assertEqual(skill["verification"], "evaluation_validator")
        self.assertIn("relative L2", " ".join(skill["rules"]))
        self.assertIn("CapabilityMatch", skill["preconditions"])

    def test_duplicate_or_missing_registration_is_refused(self):
        with self.assertRaises(SkillResolutionError):
            resolve("unknown_task")

    def test_m3_issue_context_selects_a_narrow_skill(self):
        skill = resolve_for_context("failure_triage", {"issue": "cache_layout"})
        self.assertEqual(skill["id"], "cache-layout-validation")
        skill = resolve_for_context("failure_triage", {"issue": "runtime_state"})
        self.assertEqual(skill["id"], "runtime-state-triage")

    def test_all_skill_tools_exist_in_tool_catalog(self):
        self.assertEqual(validate_tool_references(), [])

    def test_new_correctness_tasks_use_kernel_grade(self):
        for task_type in (
            "platform_kernel_correctness",
            "end_to_end_accuracy",
            "long_context_sparse_correctness",
        ):
            with self.subTest(task_type=task_type):
                self.assertEqual(resolve(task_type)["id"], "kernel-grade")

    def test_bringup_loop_has_a_default_skill(self):
        self.assertEqual(resolve("model_bringup_loop")["id"], "model-bringup-loop")


if __name__ == "__main__":
    unittest.main()
