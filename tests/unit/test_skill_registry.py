import sys
import unittest
import hashlib
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.skill_registry import (
    SkillResolutionError,
    execution_contract,
    index_by_task_type,
    load_catalog,
    load_packages,
    resolve,
    resolve_for_context,
    validate_method_references,
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
            "operator_task_dispatch", "baseline_freeze",
            "operator_candidate_integration",
            # The two pre-flight gates. Both run before anything expensive, so both
            # need a Skill for the graph to resolve a method before executing them.
            "runtime_drift_scan", "toy_bringup",
            # The gate that turns torch shims into operator requests.
            "torch_shim_handoff",
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

    def test_all_method_packages_are_connected_and_valid(self):
        self.assertEqual(validate_method_references(), [])

    def test_execution_contract_contains_the_exact_method_snapshot(self):
        selected = resolve_for_context("model_scan")
        contract = execution_contract(selected, "model_scan")
        method = contract["method"]
        document = ROOT / method["document"]
        content = document.read_text(encoding="utf-8")
        self.assertEqual(method["package_id"], "model-scanner")
        self.assertEqual(method["content"], content)
        self.assertEqual(
            method["sha256"],
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    def test_contextual_fallback_routes_to_its_packaged_method(self):
        selected = resolve_for_context(
            "capability_evaluation", {"strategy": "torch_fallback"}
        )
        contract = execution_contract(selected, "capability_evaluation")
        self.assertEqual(contract["id"], "fallback-validation")
        self.assertEqual(contract["method"]["package_id"], "fallback-validation")

    def test_non_mapping_package_descriptor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "broken"
            package.mkdir()
            (package / "skill.yaml").write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(SkillResolutionError, "YAML mapping"):
                load_packages(Path(tmp))

    def test_invalid_nested_package_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "broken"
            package.mkdir()
            (package / "skill.yaml").write_text(
                "\n".join([
                    "kind: Skill",
                    "id: broken",
                    "task_types: 7",
                    "catalog: catalog/skill_catalog.yaml",
                    "method_document: skills/broken/SKILL.md",
                    "",
                ]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SkillResolutionError, "task_types"):
                load_packages(Path(tmp))

    def test_non_mapping_catalog_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.yaml"
            catalog.write_text(
                "kind: SkillCatalog\nentries: [broken]\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SkillResolutionError, "must be a mapping"):
                load_catalog(catalog)

    def test_non_mapping_method_front_matter_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "broken"
            package.mkdir()
            document = package / "SKILL.md"
            document.write_text("---\n[]\n---\n# Broken\n", encoding="utf-8")
            (package / "skill.yaml").write_text(
                "\n".join([
                    "api_version: infer.kunlun/v1alpha1",
                    "kind: Skill",
                    "id: broken",
                    "task_types: [model_scan]",
                    "catalog: catalog/skill_catalog.yaml",
                    f"method_document: {document}",
                    "",
                ]),
                encoding="utf-8",
            )
            selected = {
                "id": "broken-route",
                "task_types": ["model_scan"],
                "tools": ["model_scan"],
                "verification": "scan_validator",
                "exit_conditions": ["SCAN_READY"],
                "method_package": "broken",
            }
            with self.assertRaisesRegex(SkillResolutionError, "front matter"):
                execution_contract(selected, "model_scan", skills_root=root)

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
