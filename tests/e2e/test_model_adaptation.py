"""The original model graph and the scheduler, across real process boundaries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from core.paths import REPO_ROOT
from core.storage import ArtifactStore
from tests.e2e.external import prepare_environment


pytestmark = pytest.mark.local_e2e


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot() -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    )
    return {
        name: digest(REPO_ROOT / name)
        for name in result.stdout.decode().split("\0")
        if name and (REPO_ROOT / name).is_file()
    }


class Scenario:
    def __init__(self, root: Path):
        self.root = root
        self.run_id = "local-model-adaptation"
        self.run_root = root / "run"
        self.state = root / "state.sqlite"
        self.fixture = prepare_environment(root / "external")
        self.env = {**os.environ, **self.fixture["env"], "PYTHONDONTWRITEBYTECODE": "1"}
        self.logs = ArtifactStore(root / "driver")
        self.step = 0
        metadata = ArtifactStore(root).write_json(
            "metadata.json", {"evidence_mode": "simulation", "environment_required": False},
        )
        self.adaptation(
            "create-run", "--run-id", self.run_id, "--model", self.fixture["subject"],
            "--model-revision", self.fixture["model_revision"],
            "--plugin-revision", self.fixture["plugin_revision"],
            "--backend", self.fixture["backend"], "--metadata", str(metadata),
            "--artifact-root", str(self.run_root),
        )

    def process(self, entrypoint: str, arguments: list[str], expected: int | None = 0):
        self.step += 1
        command = [sys.executable, str(REPO_ROOT / entrypoint), *arguments]
        result = subprocess.run(
            command, cwd=self.root, env=self.env, capture_output=True,
            text=True, check=False, timeout=120,
        )
        log = self.logs.write_text(
            f"{self.step:03d}.log",
            json.dumps(command) + f"\nexit={result.returncode}\n"
            + result.stdout + "\n" + result.stderr,
        )
        if expected is not None:
            assert result.returncode == expected, (
                f"expected exit {expected}, got {result.returncode}; {log}\n"
                + result.stdout[-12000:] + result.stderr[-3000:]
            )
        return result

    def adaptation(self, *arguments: str, expected: int | None = 0):
        return self.process(
            "cli/adaptation.py", ["--state", str(self.state), *arguments], expected,
        )

    def graph(self, *arguments: str, expected: int | None = 0, scheduled: bool = True):
        return self.process("cli/workflow/graph.py", [
            *self.fixture["graph_args"],
            "--workflow", str(REPO_ROOT / "workflows/model_adaptation.yaml"),
            *(["--scheduler-state", str(self.state), "--run-id", self.run_id] if scheduled else []),
            "--artifact-root", str(self.run_root),
            *(["--operator-report", str(self.fixture["operator_report"])] if scheduled else []),
            "--execute", "--json", "--watch-interval", "0", *arguments,
        ], expected)

    def worker(self, stage: str, *arguments: str, expected: int | None = 0):
        return self.process("tests/e2e/worker.py", [
            "--state", str(self.state), "--run-id", self.run_id,
            "--stage", stage, *arguments,
        ], expected)

    def status(self, run_id: str | None = None) -> dict:
        return json.loads(self.adaptation(
            "status", "--run-id", run_id or self.run_id, "--events",
        ).stdout)

    def facts(self) -> list[dict]:
        return [json.loads(line) for line in self.run_root.joinpath("journal.jsonl").read_text().splitlines()]

    def delivery(self, result) -> dict:
        decisions = [
            json.loads(line) for line in result.stdout.splitlines()
            if line.startswith('{"status":') and '"reason_code":' in line
        ]
        delivery = next(item for item in decisions if item["reason_code"] == "DELIVERY_RECORDED")
        assert delivery["state"] == "SIMULATION_PASS"
        assert delivery["evidence_mode"] == "simulation"
        receipt = json.loads(Path(delivery["receipt_path"]).read_text())
        assert receipt["evidence_mode"] == "simulation"
        assert Path(delivery["manifest_path"]).is_file()
        return delivery


@pytest.fixture
def scenario(tmp_path, request):
    source_before = source_snapshot()
    case = Scenario(tmp_path)
    case.logs.write_json("source_snapshot.json", source_before)
    request.node.user_properties.append(("artifact_root", str(tmp_path)))
    yield case
    case.logs.register(
        identity={"scenario": "model_adaptation", "case": request.node.name},
        outcome="RECORDED",
    )
    assert source_snapshot() == source_before


def test_environment_user_id_input_and_restart(scenario):
    scenario.env.pop("USER_ID", None)
    arguments = ["environment", "--run-id", scenario.run_id,
                 "--health-interval-seconds", "0"]
    missing = json.loads(scenario.adaptation(*arguments, expected=6).stdout)
    assert missing["proof"]["status"] == "INPUT_REQUIRED"
    assert "INPUT_REQUIRED" in missing["error"]
    assert "--user-id" in missing["proof"]["message"]
    missing_root = Path(missing["proof"]["artifact_root"])
    assert (missing_root / "status.json").is_file()
    assert json.loads((scenario.fixture["root"] / "cluster.json").read_text())["pods"] == {}

    ready = json.loads(scenario.adaptation(*arguments, "--user-id", "simulation").stdout)
    assert ready["proof"]["validator"]["passed"] is True
    assert ready["proof"]["evidence_mode"] == "simulation"
    assert ready["run"]["environment"]["environment_proof"]["user_id"] == "simulation"
    first_root = Path(ready["proof"]["artifact_root"])
    contract = json.loads((first_root.parent / "input" / "task_contract.json").read_text())
    assert contract["execution"]["user_id"] == "simulation"
    assert contract["metadata"]["generated_by"] == "harness.environment_contract"
    assert contract["context"]["model"]["name"] == "MiniMax-M2.5-Int8-W8A8"

    # A new CLI process resumes the same run with no ID in argv or environment.
    resumed = json.loads(scenario.adaptation(*arguments).stdout)
    assert resumed["proof"]["validator"]["passed"] is True
    assert resumed["proof"]["pod"] == ready["proof"]["pod"]
    assert resumed["proof"]["user_id"] == "simulation"
    assert Path(resumed["proof"]["artifact_root"]) not in {first_root, missing_root}
    assert (first_root / "status.json").is_file()


def test_graph_rejects_manual_deployment_contract(scenario):
    rejected = scenario.graph("--set", f"contract_instance={scenario.fixture['contract_instance']}", expected=2)
    assert "MAT-005 supplies the service contract" in rejected.stdout
    assert json.loads((scenario.fixture["root"] / "cluster.json").read_text())["pods"] == {}


@pytest.mark.parametrize("scheduled", [True, False])
def test_environment_drift_blocks_before_intake_and_resumes_same_pod(scenario, scheduled):
    settings = json.loads(scenario.fixture["settings"].read_text())
    settings["environment_drift"] = True
    scenario.fixture["settings"].write_text(json.dumps(settings))
    failed = scenario.graph(expected=2, scheduled=scheduled)
    assert '"reason_code": "ENVIRONMENT_FAILED"' in failed.stdout
    assert "INPUT_UNRESOLVED" not in failed.stdout
    assert not (scenario.run_root / "tasks/mat-001-model-intake").exists()
    assert not (scenario.run_root / "tasks/mat-006-failure-triage").exists()
    proof = next(scenario.run_root.glob("tasks/kdp-001a-environment-proof/attempts/*/output/status.json"))
    original = proof.read_bytes()
    status = json.loads(original)
    assert status["state"] == "RUNTIME_DRIFT"
    assert status["pod"] == scenario.fixture["pod"]
    assert (proof.parent / "engine_core_drift_precheck.json").is_file()
    assert (proof.parent / "diagnosis.json").is_file()
    if scheduled:
        snapshot = scenario.status()
        assert snapshot["run"]["status"] == "ENVIRONMENT_FAILED"
    settings.pop("environment_drift")
    scenario.fixture["settings"].write_text(json.dumps(settings))
    scenario.graph("--resume", "--until-node", "mat-001-model-intake", scheduled=scheduled)
    assert proof.read_bytes() == original
    intake = next(fact for fact in scenario.facts() if fact["kind"] == "ModelRequest")
    probe = json.loads((Path(intake["artifacts"]) / "resolved_revision.json").read_text())
    assert probe["probed_in"] == "imported context: " + scenario.fixture["pod"]
    events = [json.loads(line) for line in scenario.fixture["events"].read_text().splitlines()]
    applies = [event for event in events if event["operation"] == "cluster"
               and event["args"][:2] == ["apply", "-f"]]
    assert len(applies) == 1


def test_model_adaptation_delivers_after_validated_workers(scenario):
    scenario.env.pop("USER_ID", None)
    scenario.fixture["graph_args"].extend(["--set", "user_id=simulation"])
    first = scenario.graph(expected=3)
    assert '"reason_code": "WAITING_FOR_OPERATORS"' in first.stdout
    before = scenario.status()
    assert before["progress"]["reason_code"] == "WORKER_UNCLAIMED"
    assert before["progress"]["next_action"]["owner"] == "torch_worker"
    assert before["run"]["metadata"]["graph_progress"]["node"] == "mat-026-operator-candidate-integration"
    assert [(task["stage"], task["status"]) for task in before["tasks"]] == [("torch", "pending")]
    events = [event["event_type"] for event in before["events"]]
    assert events.index("environment_bound") < events.index("operator_discovered")
    kinds = [fact["kind"] for fact in scenario.facts()]
    ordered = ["EnvironmentProof", "ModelRequest", "ModelSupportCard", "CapabilityMatch",
               "GapClassification", "ToyBringupReport", "TorchShimRegistry", "DeploymentProof"]
    assert [kinds.index(kind) for kind in ordered] == sorted(kinds.index(kind) for kind in ordered)
    external = [json.loads(line) for line in scenario.fixture["events"].read_text().splitlines()]
    scripts = [event["script"] for event in external if event["operation"] == "exec"]
    base_launch = next(i for i, script in enumerate(scripts)
                       if "echo started $!" in script and "minimax-base-smoke" in script)
    intake = next(i for i, script in enumerate(scripts) if "python3 /tmp/mat001_probe.py" in script)
    toy = next(i for i, script in enumerate(scripts) if "python3 /tmp/mat028_probe.py" in script)
    target_launch = next(i for i, script in enumerate(scripts)
                         if "echo started $!" in script and "minimax-base-smoke" not in script)
    assert base_launch < intake < toy < target_launch
    # The target proof consumes the real MAT-005 output, never the legacy
    # boundary fixture prepared for separate replay tests.
    import yaml
    facts = scenario.facts()
    plan_fact = next(fact for fact in facts if fact["kind"] == "DeploymentPlan")
    service_fact = next(fact for fact in facts if fact["kind"] == "DeploymentProof")
    planned = yaml.safe_load((Path(plan_fact["artifacts"]) / "kdp_instance.yaml").read_text())
    executed = yaml.safe_load((Path(service_fact["artifacts"]) / "task_contract.yaml").read_text())
    assert executed["metadata"]["generated_by"] == "mat-005-deployment-plan"
    assert executed["execution"]["commands"]["serve"] == planned["execution"]["commands"]["serve"]


    intake_events = [event for event in external if event["operation"] == "exec"
                     and "python3 /tmp/mat001_probe.py" in event["script"]]
    assert {event["pod"] for event in intake_events} == {scenario.fixture["pod"]}
    intake_fact = next(fact for fact in scenario.facts() if fact["kind"] == "ModelRequest")
    intake_probe = json.loads((Path(intake_fact["artifacts"]) / "resolved_revision.json").read_text())
    assert intake_probe["probed_in"] == "imported context: " + scenario.fixture["pod"]
    memory = json.loads((scenario.run_root / "task_memory.json").read_text())
    assert memory["next_loop_block"]["sub_target"] == "mat-026-operator-candidate-integration"
    assert memory["completed_loop_blocks"][-1]["state"] == "WAITING_FOR_OPERATORS"
    scan_skill_path = next(
        scenario.run_root.glob("tasks/mat-002-model-scan/attempts/*/input/skill.json")
    )
    scan_skill = json.loads(scan_skill_path.read_text(encoding="utf-8"))
    assert scan_skill["id"] == "model-scanner"
    assert scan_skill["method"]["package_id"] == "model-scanner"
    assert scan_skill["method"]["content"].startswith("---\nname: model-scanner\n")
    scan_fact = next(fact for fact in scenario.facts() if fact["kind"] == "ModelSupportCard")
    assert scan_fact["detail"]["skill"] == {
        "id": "model-scanner",
        "method_sha256": scan_skill["method"]["sha256"],
    }
    scan_block = next(
        block for block in memory["completed_loop_blocks"]
        if block["sub_target"] == "mat-002-model-scan"
    )
    assert scan_block["routing"]["method_sha256"] == scan_skill["method"]["sha256"]
    original_service = next(
        fact["artifacts"] for fact in scenario.facts() if fact["kind"] == "DeploymentProof"
    )

    for stage in ("torch", "xpu", "integration"):
        scenario.worker(stage)
    ready = scenario.status()["progress"]
    assert ready["reason_code"] == "RESUME_GRAPH"
    assert ready["state"] == "READY"
    assert "--resume" in ready["next_action"]["command"]
    delivery = scenario.delivery(scenario.graph("--resume"))
    snapshot = scenario.status()
    assert snapshot["progress"]["state"] == "COMPLETED"
    assert snapshot["progress"]["details"]["raw_status"] == "SIMULATION_PASS"
    assert len(snapshot["tasks"]) == 3
    fingerprint = snapshot["run"]["environment"]["environment_proof"]["fingerprint"]
    for task in snapshot["tasks"]:
        assert task["status"] == "succeeded"
        assert task["attempt"] == 1
        assert task["output"]["evidence_mode"] == "simulation"
        assert task["output"]["environment_fingerprint"] == fingerprint
        assert task["lease_token"] is None
        validation = json.loads(
            Path(task["output"]["evidence"]["independent_validation"]).read_text(),
        )
        assert validation["validator"] != f"local-scenario-{task['stage']}"
        assert all(check["passed"] is True for check in validation["checks"])
        workspace = Path(task["input"]["workspace"]["output"]).parent
        manifest = json.loads((workspace / "manifest.json").read_text())
        assert manifest["identity"]["run_id"] == scenario.run_id
        for artifact in manifest["artifacts"]:
            assert digest(workspace / artifact["path"]) == artifact["sha256"]
    reference_task = next(task for task in snapshot["tasks"] if task["stage"] == "torch")
    reference = json.loads(Path(reference_task["output"]["evidence"]["reference_artifact"]).read_text())
    assert reference["outputs"] == [2 * value for value in reference["inputs"]]
    service = next(
        fact["artifacts"] for fact in reversed(scenario.facts()) if fact["kind"] == "DeploymentProof"
    )
    assert service != original_service
    before_service = json.loads(
        (Path(service).parent / "input/scheduler_snapshot.json").read_text(),
    )
    assert before_service["state"] == "OPERATORS_READY"
    assert set(before_service["task_ids"]) == {task["task_id"] for task in snapshot["tasks"]}
    assert {
        "ModelRequest", "EnvironmentProof", "RuntimeDriftReport", "ModelSupportCard",
        "CapabilityMatch", "GapClassification", "OperatorTaskDispatch",
        "CapabilityEvaluation", "DeploymentPlan", "ToyBringupReport",
        "TorchShimRegistry", "DeploymentProof", "AccuracyDifferential",
        "ServingBaseline", "OperatorIntegration", "MemoryBudget", "ApiConformance",
        "SupportMatrixEntry",
    } <= {fact["kind"] for fact in scenario.facts()}
    memory = json.loads((scenario.run_root / "task_memory.json").read_text())
    assert memory["status"] == "COMPLETED"
    assert delivery["receipt_path"] in memory["completed_loop_blocks"][-1]["artifacts"]


def test_missing_worker_evidence_blocks_delivery(scenario):
    scenario.graph(expected=3)
    scenario.worker("torch", "--fault", "missing-evidence", expected=6)
    snapshot = scenario.status()
    assert snapshot["progress"]["reason_code"] == "DIAGNOSIS_PENDING"
    source_failure = next(item for item in snapshot["task_progress"] if item["details"]["stage"] == "torch")
    assert source_failure["details"]["validation_errors"]
    assert {(task["stage"], task["status"]) for task in snapshot["tasks"]} == {
        ("torch", "failed"), ("diagnosis", "pending"),
    }
    blocked = scenario.graph("--resume", expected=2)
    assert "DELIVERY_RECORDED" not in blocked.stdout
    assert not list(scenario.run_root.glob("tasks/model-adaptation-delivery/attempts/*/output/*.json"))
    assert not any(task["stage"] == "xpu" for task in scenario.status()["tasks"])


def test_pre_operator_service_cannot_certify_delivery(scenario):
    scenario.graph(expected=3)
    for stage in ("torch", "xpu", "integration"):
        scenario.worker(stage)
    blocked = scenario.graph(
        "--resume", "--from-node", "mat-026-operator-candidate-integration", expected=2,
    )
    assert '"reason_code": "DELIVERY_GATE"' in blocked.stdout
    assert "final service must start after the operator gate" in blocked.stdout.lower()
    assert "DELIVERY_RECORDED" not in blocked.stdout
    assert all(task["status"] == "succeeded" for task in scenario.status()["tasks"])
    scenario.delivery(scenario.graph("--resume"))


def test_fresh_service_cannot_reuse_pre_operator_accuracy(scenario):
    scenario.graph(expected=3)
    for stage in ("torch", "xpu", "integration"):
        scenario.worker(stage)
    scenario.graph(
        "--resume", "--from-node", "kdp-001b-service-proof",
        "--until-node", "kdp-001b-service-proof",
    )
    blocked = scenario.graph(
        "--resume", "--from-node", "mat-026-operator-candidate-integration", expected=2,
    )
    assert '"reason_code": "DELIVERY_GATE"' in blocked.stdout
    assert "final accuracy must start after the operator gate" in blocked.stdout.lower()
    assert "DELIVERY_RECORDED" not in blocked.stdout
    scenario.delivery(scenario.graph("--resume"))


def test_resume_preserves_attempts_and_rejects_old_lease(scenario):
    scenario.graph(expected=3)
    preserved = {path: digest(path) for path in (scenario.run_root / "tasks").rglob("*") if path.is_file()}
    abandoned = json.loads(scenario.worker(
        "torch", "--fault", "abandon", "--lease-seconds", "0.15", expected=75,
    ).stdout)["abandoned"]
    original_output = Path(abandoned["input"]["workspace"]["output"]) / "abandoned.json"
    original_hash = digest(original_output)
    assert scenario.status()["progress"]["reason_code"] in {"WORKER_RUNNING", "LEASE_EXPIRED"}
    time.sleep(max(0, abandoned["lease_expires"] - time.time()) + 0.03)
    expired = scenario.status()
    assert expired["progress"]["reason_code"] == "LEASE_EXPIRED"
    assert expired["tasks"][0]["status"] == "running"
    scenario.worker("torch")
    task = next(item for item in scenario.status()["tasks"] if item["stage"] == "torch")
    assert task["attempt"] == 2
    late = scenario.adaptation(
        "complete", "--task-id", abandoned["task_id"],
        "--worker", "local-scenario-torch", f"--lease-token={abandoned['lease_token']}",
        "--result", str(Path(task["input"]["workspace"]["output"]) / "submission.json"),
        expected=None,
    )
    assert late.returncode != 0
    for stage in ("xpu", "integration"):
        scenario.worker(stage)
    scenario.delivery(scenario.graph("--resume"))
    assert len(scenario.status()["tasks"]) == 3
    assert digest(original_output) == original_hash
    assert all(digest(path) == old for path, old in preserved.items())


def test_environment_tampering_blocks_resume(scenario):
    scenario.graph(expected=3)
    proof = scenario.status()["run"]["environment"]["environment_proof"]
    fingerprint = Path(proof["artifact_root"]) / "environment_fingerprint.txt"
    fingerprint.write_text(fingerprint.read_text() + "\ntampered-fixture\n")
    blocked = scenario.graph("--resume", expected=2)
    assert "DELIVERY_RECORDED" not in blocked.stdout
    snapshot = scenario.status()
    assert snapshot["run"]["status"] == "ENVIRONMENT_FAILED"
    assert snapshot["progress"]["reason_code"] == "ENVIRONMENT_FAILED"
    assert snapshot["progress"]["next_action"]["action"] == "PROVE_ENVIRONMENT"
    assert snapshot["run"]["environment"]["environment_proof"]["fingerprint"] == proof["fingerprint"]
    claimed = json.loads(scenario.adaptation(
        "claim", "--worker", "must-not-run", "--stage", "torch",
    ).stdout)
    assert claimed["tasks"] == []


def test_incomplete_operator_report_is_blocked(scenario):
    path = Path(scenario.fixture["operator_report"])
    report = json.loads(path.read_text())
    entries = report.get("entries", report.get("gaps"))
    assert entries
    del entries[0]["inputs"][0]["shape"]
    path.write_text(json.dumps(report))
    blocked = scenario.graph(expected=2)
    assert "DISPATCH_BLOCKED" in blocked.stdout
    dispatch = next(
        fact for fact in reversed(scenario.facts()) if fact["kind"] == "OperatorTaskDispatch"
    )
    assert "shape" in (Path(dispatch["artifacts"]) / "dispatch_status.json").read_text()
    assert scenario.status()["tasks"] == []
    progress = scenario.status()["progress"]
    assert progress["reason_code"] == "DISPATCH_BLOCKED"
    assert "shape" in progress["summary"]
    assert progress["next_action"]["action"] == "COMPLETE_OPERATOR_CONTRACT"
    assert "DELIVERY_RECORDED" not in blocked.stdout


def test_simulation_proof_cannot_unlock_real_run(scenario):
    scenario.graph("--until-node", "kdp-001a-environment-proof")
    proof = scenario.status()["run"]["environment"]["environment_proof"]
    real_id = "must-not-accept-simulation"
    scenario.adaptation(
        "create-run", "--run-id", real_id, "--model", scenario.fixture["subject"],
        "--backend", scenario.fixture["backend"], "--artifact-root", str(scenario.root / "real-run"),
    )
    rejected = scenario.adaptation(
        "environment", "--run-id", real_id,
        "--status", str(Path(proof["artifact_root"]) / "status.json"), expected=None,
    )
    assert rejected.returncode != 0
    assert "simulation" in rejected.stdout.lower()
    snapshot = scenario.status(real_id)
    assert snapshot["run"]["status"] == "ENVIRONMENT_FAILED"
    assert "environment_proof" not in snapshot["run"]["environment"]
    assert snapshot["tasks"] == []
