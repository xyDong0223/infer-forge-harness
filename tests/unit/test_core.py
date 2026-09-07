import unittest

from runners.task_runner import build_plan
from validators.contract_validator import find_placeholders, validate_task_contract
from validators.deployment_validator import validate_deployment_status


class CoreContractTests(unittest.TestCase):
    def test_valid_contract_shape(self):
        contract = {
            "api_version": "infer.kunlun/v1alpha1",
            "kind": "Task",
            "metadata": {"name": "demo", "task_type": "deployment_proof"},
            "context": {"model": {"name": "Qwen3-8B"}, "target": {"hardware": "P800"}},
            "actions": ["preflight"],
            "acceptance": {"pod_ready": True},
        }
        self.assertEqual(validate_task_contract(contract), [])

    def test_placeholders_are_visible(self):
        paths = find_placeholders({"context": {"model": {"path": "${MODEL_PATH}"}}})
        self.assertEqual(paths, ["$.context.model.path"])

    def test_plan_is_safe_by_default(self):
        contract = {
            "metadata": {"name": "demo", "task_type": "deployment_proof"},
            "actions": ["preflight"],
        }
        plan = build_plan(contract)
        self.assertEqual(plan["mode"], "PLAN_ONLY")
        self.assertTrue(plan["requires"]["explicit_execute"])

    def test_deployment_ready_requires_real_checks(self):
        status = {
            "state": "DEPLOYMENT_READY",
            "checks": {
                "pod_ready": True,
                "health_check": 200,
                "chat_completion": "non_empty",
                "expected_backend": "kunlun",
                "unexpected_fallback": False,
            },
            "artifacts": ["status.json"],
        }
        self.assertEqual(validate_deployment_status(status), [])

    def test_running_pod_is_not_deployment_ready(self):
        errors = validate_deployment_status({"state": "RUNNING", "checks": {}, "artifacts": []})
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
