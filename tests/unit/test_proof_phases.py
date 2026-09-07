"""The two halves of the deployment proof, and what each may claim."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.deployment_proof import ActionFailed, DeploymentProofRunner  # noqa: E402
from validators.deployment_validator import (  # noqa: E402
    validate_deployment_status,
    validate_environment_status,
)

ENVIRONMENT_STATUS = {
    "task_id": "kdp-001a-environment-proof",
    "state": "ENVIRONMENT_READY",
    "pod": "dongxinyu03-kdp001-qwen3-8b-7c969df894-hsz7d-0",
    "phase": "environment",
    "checks": {"pod_ready": True, "runtime_importable": True},
    "artifacts": ["environment_fingerprint.txt", "runtime_import.txt", "status.json"],
}


class PhaseSelectionTest(unittest.TestCase):
    class StubAdapter:
        """Only what collect_artifacts touches; no cluster is reachable here."""

        class config:  # noqa: N801 - mirrors the adapter attribute name
            kubeconfig = "/path/to/kubeconfig"

    def runner(self, **kwargs):
        return DeploymentProofRunner(
            contract={"metadata": {"name": "t"}, "execution": {}},
            adapter=self.StubAdapter(),
            repo_root=ROOT,
            artifact_dir=Path("/tmp/kdp-phase-test"),
            **kwargs,
        )

    def test_unknown_phase_is_refused(self):
        with self.assertRaises(ActionFailed):
            self.runner(phase="halfway")

    def test_service_phase_without_a_pod_fails_before_creating_one(self):
        """Creating a pod here would reinstall the runtime and void the proof."""
        status = self.runner(phase="service").run()
        self.assertEqual(status["state"], "CONTRACT_INVALID")
        self.assertIn("--attach-pod", status["reason"])
        self.assertIsNone(status["pod"])


class EnvironmentAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.status = {**ENVIRONMENT_STATUS, "checks": dict(ENVIRONMENT_STATUS["checks"])}

    def test_a_proven_environment_passes(self):
        self.assertEqual(validate_environment_status(self.status), [])

    def test_an_installed_but_unimportable_runtime_fails(self):
        """The 2026-09-03 case: package present, import broken by gcc 9."""
        self.status["checks"]["runtime_importable"] = False
        self.assertIn("checks.runtime_importable must be True", validate_environment_status(self.status))

    def test_the_pod_must_be_recorded_for_the_next_phase(self):
        self.status["pod"] = None
        errors = validate_environment_status(self.status)
        self.assertTrue(any("import it" in error for error in errors), errors)

    def test_the_fingerprint_is_part_of_the_deliverable(self):
        self.status["artifacts"] = ["runtime_import.txt"]
        errors = validate_environment_status(self.status)
        self.assertTrue(any("environment_fingerprint.txt" in error for error in errors), errors)

    def test_environment_readiness_is_not_deployment_readiness(self):
        """A proven environment must not pass as a served model."""
        self.assertNotEqual(validate_deployment_status(self.status), [])


if __name__ == "__main__":
    unittest.main()
