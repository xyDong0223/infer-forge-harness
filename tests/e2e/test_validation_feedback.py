"""Offline handoff uses production CLI; observed skips never certify hardware."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from core.paths import REPO_ROOT
from core.storage import ArtifactStore
from tests.e2e.test_model_adaptation import source_snapshot
from tests.e2e.test_managed_worker import ManagedScenario


pytestmark = pytest.mark.local_e2e


@pytest.fixture
def feedback_scenario(tmp_path, request):
    before = source_snapshot()
    case = ManagedScenario(tmp_path)
    # These are private plan inputs, not an environment proof or success fixture.
    # The actual hardware test below must SKIP before touching this configuration.
    private = case.store.write_json("private-scenario.json", {
        "run_id": case.run_id, "state": str(case.state),
        "artifact_root": str(tmp_path / "run"), "subject": "PRIVATE_MODEL_SENTINEL",
        "pod": "PRIVATE_POD_SENTINEL", "namespace": "PRIVATE_NAMESPACE_SENTINEL",
        "image_digest": "sha256:" + "a" * 64, "hardware": "kunlun-p800",
        "model_revision": "fixture-model-v1", "plugin_revision": "fixture-plugin-v1",
        "cleanup_policy": "retain_prepared_pod", "environment": {},
        "context": {"user_id": "PRIVATE_USER_SENTINEL"},
    })
    request.node.user_properties.append(("artifact_root", str(tmp_path)))
    yield case, private
    case.store.register(identity={"scenario": "validation_feedback", "case": request.node.name},
                        outcome="RECORDED")
    assert source_snapshot() == before


def observed_skip(case):
    """Run the real optional test without authorization; do not write a fake report."""
    env = {key: value for key, value in case.env.items()
           if key not in {"INFER_FORGE_RUN_DEVICE_SMOKE", "INFER_FORGE_RUN_REAL_MODEL",
                          "INFER_FORGE_HARDWARE_SCENARIO", "INFER_FORGE_VALIDATION_REQUEST"}}
    report = case.root / "actual-skipped-junit.xml"
    result = subprocess.run([
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/e2e/test_model_adaptation_hardware.py::test_prepared_device_smoke",
        "--basetemp", str(case.root / "hardware-pytest"), "--junitxml", str(report),
    ], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False, timeout=45)
    case.store.write_text("hardware-test-driver.log", result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout
    return report, result.returncode


def prepare_feedback(case, private):
    planned = case.cli("validation-plan", "--scenario", str(private),
                       "--tier", "device_smoke", "--out", str(case.root / "private-plan"))
    request = Path(planned["request_path"])
    junit, exit_code = observed_skip(case)
    shared = case.root / "share"
    exported = case.cli("feedback-export", "--request", str(request),
                        "--junit", str(junit), "--exit-code", str(exit_code), "--out", str(shared))
    assert exported["status"] == "FEEDBACK_EXPORTED"
    return request, shared


def durable_status(case):
    current = case.status()
    return {key: current[key] for key in ("run", "tasks", "events")}


def test_feedback_skip_export_restart_preserves_scheduler(feedback_scenario):
    case, private = feedback_scenario
    original = durable_status(case)
    db_hash = hashlib.sha256(case.state.read_bytes()).hexdigest()
    request, shared = prepare_feedback(case, private)
    # Each CLI call is a new process. Read/check has no path to mutate a task.
    first = case.cli("feedback-check", "--request", str(request), "--feedback-dir", str(shared))
    assert first["status"] == "FEEDBACK_VALID"
    assert first["promoted"] is False and first["observations_verified"] is False
    assert case.cli("feedback-check", "--request", str(request), "--feedback-dir", str(shared)) == first
    assert hashlib.sha256(case.state.read_bytes()).hexdigest() == db_hash
    assert durable_status(case) == original
    assert {path.name for path in shared.iterdir()} == {"feedback.json", "manifest.json"}
    public = request.read_text() + "".join(path.read_text() for path in shared.iterdir())
    for secret in (str(case.root), "PRIVATE_MODEL_SENTINEL", "PRIVATE_POD_SENTINEL",
                   "PRIVATE_NAMESPACE_SENTINEL", "PRIVATE_USER_SENTINEL"):
        assert secret not in public
    # Source/request hash binding is not authentication of another machine.
    assert not any(task["status"] == "succeeded" for task in case.status()["tasks"])


def test_feedback_tampering_and_wrong_request_rejected(feedback_scenario):
    case, private = feedback_scenario
    original = durable_status(case)
    request, shared = prepare_feedback(case, private)
    other = case.cli("validation-plan", "--scenario", str(private), "--tier", "device_smoke",
                     "--out", str(case.root / "other-plan"))
    mismatch = case.cli("feedback-check", "--request", other["request_path"],
                        "--feedback-dir", str(shared), expected=2)
    assert mismatch["status"] == "REJECTED"
    feedback = shared / "feedback.json"
    value = json.loads(feedback.read_text())
    value["untrusted_claim"] = "FUNCTIONAL_READY"
    feedback.write_text(json.dumps(value))
    rejected = case.cli("feedback-check", "--request", str(request),
                        "--feedback-dir", str(shared), expected=2)
    assert rejected["status"] == "REJECTED"
    assert durable_status(case) == original


def test_feedback_commands_never_create_a_scheduler_database(feedback_scenario):
    case, private = feedback_scenario
    absent = case.root / "must-not-exist.sqlite"
    command = [sys.executable, str(REPO_ROOT / "cli/adaptation.py"), "--state", str(absent)]
    result = case.process([*command, "validation-plan", "--scenario", str(private),
                           "--tier", "device_smoke", "--out", str(case.root / "offline-plan")])
    assert json.loads(result.stdout)["status"] == "PLANNED"
    assert not absent.exists()
