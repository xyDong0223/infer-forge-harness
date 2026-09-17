"""Optional real tiers. Never collect credentials or provision a replacement Pod."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from core.paths import REPO_ROOT
from core.storage import ArtifactStore, RunPaths, ensure_external
from core.target import canonical_hardware
from engine.scheduler import TaskScheduler


@pytest.fixture
def hardware_scenario(request):
    from adapters import ClusterConfig

    value = os.environ.get("INFER_FORGE_HARDWARE_SCENARIO")
    assert value, "explicit authorization also requires INFER_FORGE_HARDWARE_SCENARIO"
    config = json.loads(ensure_external(value).read_text())
    required = {
        "run_id", "state", "artifact_root", "subject", "pod",
        "namespace", "image_digest", "hardware", "model_revision", "plugin_revision",
        "cleanup_policy", "environment", "context",
    }
    assert required <= config.keys(), f"missing configuration: {sorted(required - config.keys())}"
    assert config["cleanup_policy"] == "retain_prepared_pod"
    assert str(config["image_digest"]).startswith("sha256:")
    assert config["context"].get("user_id"), "hardware scenarios require the user-supplied owner ID"
    assert ClusterConfig.load(user_id=config["context"]["user_id"]).namespace == config["namespace"]
    state = ensure_external(config["state"])
    assert state.is_file(), "optional tiers require an existing run"
    scheduler = TaskScheduler(state)
    try:
        run = scheduler.store.run(config["run_id"])
        assert run is not None
        assert run.metadata.get("evidence_mode", "real") == "real"
        assert run.model_id == config["subject"]
        assert run.model_revision == config["model_revision"] != "unknown"
        assert run.plugin_revision == config["plugin_revision"] != "unknown"
        assert ensure_external(run.metadata["artifact_root"]) == ensure_external(config["artifact_root"])
        assert run.environment["environment_proof"]["pod"] == config["pod"]
        assert run.environment["image_digest"] == config["image_digest"]
        assert canonical_hardware(run.environment["hardware"]) == canonical_hardware(config["hardware"])
    finally:
        scheduler.store.close()
    attempt = RunPaths(config["artifact_root"], config["run_id"]).allocate_attempt(request.node.name)
    request.node.user_properties.append(("artifact_root", str(attempt.root)))
    yield config, attempt
    ArtifactStore(attempt.root).register(identity=attempt.identity, outcome="RECORDED")


def run_real(config, attempt, entrypoint, arguments):
    command = [sys.executable, str(REPO_ROOT / entrypoint), *arguments]
    env = {key: value for key, value in os.environ.items() if not key.startswith("INFER_FORGE_E2E")}
    # Do not inherit a test bootstrap from a prior local scenario invocation.
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        check=False, timeout=7200,
    )
    logs = ArtifactStore(attempt.logs)
    logs.write_text("stdout.log", result.stdout)
    logs.write_text("stderr.log", result.stderr)
    ArtifactStore(attempt.input).write_json("command.json", command)
    assert result.returncode == 0, f"real tier did not complete; inspect {attempt.logs}"
    return result


@pytest.mark.device_smoke
@pytest.mark.skipif(
    os.environ.get("INFER_FORGE_RUN_DEVICE_SMOKE") != "1",
    reason="real device smoke requires explicit authorization",
)
def test_prepared_device_smoke(hardware_scenario):
    config, attempt = hardware_scenario
    result = run_real(config, attempt, "cli/adaptation.py", [
        "--state", config["state"], "environment", "--run-id", config["run_id"],
        "--user-id", config["context"]["user_id"], "--attach-pod", config["pod"],
        "--artifact-dir", str(attempt.output / "environment"),
    ])
    proof = json.loads(result.stdout)["run"]["environment"]["environment_proof"]
    assert proof["pod"] == config["pod"]
    assert proof["checks"]["base_prefill"] is True
    assert proof["checks"]["base_decode"] is True
    assert proof["checks"]["unexpected_fallback"] is False


@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("INFER_FORGE_RUN_REAL_MODEL") != "1",
    reason="real-model regression requires explicit authorization",
)
def test_real_model_service_regression(hardware_scenario):
    config, attempt = hardware_scenario
    arguments = [
        "--workflow", str(REPO_ROOT / "workflows/model_adaptation.yaml"),
        "--scheduler-state", config["state"], "--run-id", config["run_id"],
        "--artifact-root", config["artifact_root"], "--subject", config["subject"],
        "--from-node", "kdp-001b-service-proof", "--execute", "--json",
    ]
    context = {**config["context"], "pod": config["pod"]}
    for key, value in config["environment"].items():
        arguments += ["--env", f"{key}={value}"]
    for key, value in context.items():
        arguments += ["--set", f"{key}={value}"]
    result = run_real(config, attempt, "cli/workflow/graph.py", arguments)
    decisions = [
        json.loads(line) for line in result.stdout.splitlines()
        if line.startswith('{"status":') and '"reason_code":' in line
    ]
    delivery = next(item for item in decisions if item["reason_code"] == "DELIVERY_RECORDED")
    assert delivery["evidence_mode"] == "real"
    assert delivery["state"] == "FUNCTIONAL_READY"
