"""CLI contract tests for the unified adaptation entry point."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from engine.contracts import OperatorTask
from tests.scheduler_helpers import simulation_result, stage_result

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "cli" / "adaptation.py"


def _run(state: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--state", str(state), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_external_state_and_run_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("INFER_FORGE_STATE_ROOT", str(tmp_path / "state"))
    command = [sys.executable, str(SCRIPT), "create-run",
               "--run-id", "owned", "--model", "model", "--backend", "device"]
    created = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert created.returncode == 0, created.stdout + created.stderr
    run = json.loads(created.stdout)["run"]
    assert run["metadata"]["artifact_root"] == str(tmp_path / "state/runs/owned")
    assert (tmp_path / "state/state.sqlite").is_file()
    assert json.loads((tmp_path / "state/runs/owned/run.json").read_text())["run_id"] == "owned"
    resumed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert json.loads(resumed.stdout)["run"] == run


def test_existing_run_cannot_be_redirected_to_another_output(tmp_path):
    state = tmp_path / "state.sqlite"
    args = ("create-run", "--run-id", "owned", "--model", "model", "--backend", "device")
    created = _run(state, *args, "--artifact-root", str(tmp_path / "one"))
    assert created.returncode == 0, created.stdout
    redirected = _run(state, *args, "--artifact-root", str(tmp_path / "two"))
    assert redirected.returncode != 0
    assert "different artifact_root" in redirected.stdout
    assert not (tmp_path / "two").exists()


def test_in_repository_runtime_output_is_rejected(tmp_path):
    created = _run(
        tmp_path / "state.sqlite", "create-run", "--run-id", "owned",
        "--model", "model", "--backend", "device",
        "--artifact-root", str(ROOT / "artifacts/forbidden-run"),
    )
    assert created.returncode != 0
    assert "overlaps the source repository" in created.stdout
    assert not (ROOT / "artifacts/forbidden-run").exists()


def test_create_discover_claim_and_status_emit_json(tmp_path: Path) -> None:
    state = tmp_path / "adaptation.db"
    report = tmp_path / "gaps.json"
    report.write_text(
        json.dumps(
            {
                "plugin_revision": "kunlun-1",
                "entries": [
                    {
                        "operator_id": "missing_op",
                        "inputs": [
                            {"name": "x", "dtype": "float32", "shape": [1], "layout": "contiguous"}
                        ],
                        "outputs": [
                            {"name": "y", "dtype": "float32", "shape": [1], "layout": "contiguous"}
                        ],
                        "semantics": {"reference": "torch.add"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    created = _run(
        state,
        "create",
        "--run-id",
        "run-1",
        "--model",
        "DeepSeek-V4.1",
        "--backend",
        "kunlun-p800",
    )
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["run"]["run_id"] == "run-1"

    proof = tmp_path / "environment-status.json"
    artifact_names = [
        "environment_fingerprint.txt", "runtime_import.txt", "code_readiness.json",
        "device_readiness.json", "base_model_identity.json", "base_health_result.txt",
        "base_chat_result.json", "base_server_log.txt",
    ]
    for name in artifact_names:
        (tmp_path / name).write_text("local test fixture: " + name, encoding="utf-8")
    from tests.scheduler_helpers import write_base_model_identity
    write_base_model_identity(tmp_path)
    proof.write_text(
        json.dumps(
            {
                "state": "ENVIRONMENT_READY",
                "pod": "prepared-pod",
                "user_id": "fixture-owner",
                "checks": {
                    "pod_ready": True,
                    "runtime_importable": True,
                    "code_ready": True,
                    "device_ready": True,
                    "base_model_loaded": True,
                    "base_prefill": True,
                    "base_decode": True,
                    "base_health_check": 200,
                    "base_chat_completion": "non_empty",
                    "unexpected_fallback": False,
                },
                "artifacts": artifact_names,
            }
        ),
        encoding="utf-8",
    )
    environment = _run(
        state,
        "environment",
        "--run-id",
        "run-1",
        "--status",
        str(proof),
    )
    assert environment.returncode == 0, environment.stdout
    assert json.loads(environment.stdout)["run"]["status"] == "ENVIRONMENT_READY"

    discovered = _run(state, "discover", "--run-id", "run-1", "--report", str(report))
    assert discovered.returncode == 0, discovered.stdout
    payload = json.loads(discovered.stdout)
    assert len(payload["specs"]) == 1
    assert payload["tasks"][0]["stage"] == "torch"

    claimed = _run(state, "claim", "--worker", "torch-agent", "--stage", "torch")
    assert claimed.returncode == 0, claimed.stdout
    assert json.loads(claimed.stdout)["tasks"][0]["status"] == "running"

    status = _run(state, "status", "--run-id", "run-1", "--events")
    assert status.returncode == 0, status.stdout
    status_payload = json.loads(status.stdout)
    assert status_payload["run"]["model_id"] == "DeepSeek-V4.1"
    assert status_payload["tasks"][0]["status"] == "running"
    assert status_payload["tasks"][0]["lease_token"]
    assert [event["event_type"] for event in status_payload["events"]] == [
        "run_created",
        "environment_bound",
        "operator_discovered",
        "task_created",
        "task_claimed",
    ]

    result = tmp_path / "torch-result.json"
    claim = OperatorTask.from_dict(json.loads(claimed.stdout)["tasks"][0])
    fingerprint = json.loads(environment.stdout)["run"]["environment"]["environment_proof"]["fingerprint"]
    result.write_text(json.dumps(stage_result(
        tmp_path / "evidence", claim, worker="torch-agent",
        environment_fingerprint=fingerprint, evidence_mode="real",
    )), encoding="utf-8")
    completed = _run(
        state,
        "complete",
        "--task-id",
        payload["tasks"][0]["task_id"],
        "--worker",
        "torch-agent",
        "--lease-token",
        claim.lease_token,
        "--result",
        str(result),
    )
    assert completed.returncode == 0, completed.stdout
    assert json.loads(completed.stdout)["task"]["status"] == "succeeded"


def test_cli_requires_token_and_renews_current_lease(tmp_path: Path) -> None:
    from engine import IOSpec, OperatorSpec, TaskScheduler

    state = tmp_path / "state.db"
    scheduler = TaskScheduler(state)
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    spec = OperatorSpec(
        operator_id="op", model_id="m", model_revision="1", plugin_revision="1", backend="device",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")], semantics={"op": "identity"},
    )
    scheduler.discover_operator("r", spec)
    task = scheduler.claim_ready("worker")[0]
    renewed = _run(
        state, "renew-lease", "--task-id", task.task_id, "--worker", "worker",
        "--lease-token", task.lease_token, "--lease-seconds", "600",
    )
    assert renewed.returncode == 0, renewed.stdout
    assert json.loads(renewed.stdout)["task"]["lease_expires"] > task.lease_expires
    result = tmp_path / "result.json"
    result.write_text('{"status": "PASS"}', encoding="utf-8")
    missing = _run(state, "complete", "--task-id", task.task_id, "--worker", "worker", "--result", str(result))
    assert missing.returncode == 2
    rejected = _run(
        state, "complete", "--task-id", task.task_id, "--worker", "worker",
        "--lease-token", task.lease_token, "--result", str(result),
    )
    assert rejected.returncode == 6
    assert json.loads(rejected.stdout)["task"]["status"] == "failed"


def test_cli_applies_retry_diagnosis_and_exposes_new_attempt(tmp_path: Path) -> None:
    from engine import IOSpec, OperatorSpec, TaskScheduler

    state = tmp_path / "state.db"
    scheduler = TaskScheduler(state)
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    source = scheduler.discover_operator("r", OperatorSpec(
        operator_id="op", model_id="m", model_revision="1", plugin_revision="1",
        backend="device",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")],
        semantics={"op": "identity"},
    ))
    first = scheduler.claim_ready("worker", stage="torch")[0]
    scheduler.fail(source.task_id, "worker", "transient", first.lease_token)
    diagnosis = scheduler.claim_ready("diagnoser", stage="diagnosis")[0]
    result = tmp_path / "diagnosis.json"
    result.write_text(
        json.dumps(simulation_result(
            diagnosis, tmp_path / "evidence", worker_id="diagnoser",
            next_action="RETRY",
        )),
        encoding="utf-8",
    )
    resolved = _run(
        state, "resolve-diagnosis", "--task-id", diagnosis.task_id,
        "--worker", "diagnoser", "--lease-token", diagnosis.lease_token,
        "--result", str(result),
    )
    assert resolved.returncode == 0, resolved.stdout
    applied = _run(state, "apply-diagnosis", "--task-id", diagnosis.task_id)
    assert applied.returncode == 0, applied.stdout
    assert json.loads(applied.stdout)["task"]["status"] == "pending"
    retried = _run(state, "claim", "--worker", "worker-2", "--stage", "torch")
    assert retried.returncode == 0, retried.stdout
    retry_task = json.loads(retried.stdout)["tasks"][0]
    assert retry_task["task_id"] == source.task_id
    assert retry_task["attempt"] == first.attempt + 1


def test_contract_binding_uses_the_runners_absolute_artifact_root(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from cli.adaptation import _parser, _run as run_command

    expected = tmp_path / "artifacts" / "relative-proof" / "tasks" / "proof" / "attempts" / "000001" / "output"
    proof = {"state": "ENVIRONMENT_READY", "artifact_root": str(expected)}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "cli.adaptation.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(proof), stderr="",
        ),
    )
    observed = {}

    class Scheduler:
        store = SimpleNamespace(run=lambda run_id: SimpleNamespace(metadata={}, environment={}))

        def bind_environment(self, run_id, status, artifact_root):
            observed.update(root=artifact_root, status=status)
            return SimpleNamespace(to_dict=lambda: {"run_id": run_id})

    args = _parser().parse_args([
        "--state", "state.db", "environment", "--run-id", "r",
        "--contract", "contract.yaml", "--artifact-dir", "artifacts/relative-proof",
    ])
    run_command(args, Scheduler())
    assert Path(observed["root"]) == expected


@pytest.mark.parametrize("command", ["status", "list"])
def test_unknown_run_is_machine_readable_error(tmp_path: Path, command: str) -> None:
    from engine.scheduler import EventStore

    state = tmp_path / "state.db"
    EventStore(state).close()
    result = _run(state, command, "--run-id", "missing")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["error"] == "unknown run: missing"
    assert payload["command"] == command
    assert payload["progress"]["next_action"]["action"] == "FIX_COMMAND"


@pytest.mark.parametrize("arguments", [("status", "--run-id", "missing"), ("list",),
                                       ("list", "--run-id", "missing")])
def test_query_missing_database_is_read_only(tmp_path: Path, arguments) -> None:
    state = tmp_path / "missing/state.db"
    result = _run(state, *arguments)
    assert result.returncode == 2
    assert "existing state database" in json.loads(result.stdout)["error"]
    assert not state.parent.exists()


def test_list_reads_all_or_one_run_without_recovering_or_relabelling(tmp_path: Path) -> None:
    from engine import IOSpec, OperatorSpec, TaskScheduler

    state = tmp_path / "state.db"
    scheduler = TaskScheduler(state)
    for run_id, protocol in (("legacy", "legacy-v1"), ("managed", "managed-v2")):
        scheduler.create_run(run_id=run_id, model_id=run_id, model_revision="model-v1",
                             plugin_revision="plugin-v1", backend="simulation", metadata={
            "evidence_mode": "simulation", "worker_protocol": protocol,
            "artifact_root": str(tmp_path / run_id),
        })
        scheduler.discover_operator(run_id, OperatorSpec(
            operator_id="op", model_id=run_id, model_revision="model-v1",
            plugin_revision="plugin-v1", backend="simulation",
            inputs=[IOSpec("x", "float64", [1], "contiguous")],
            outputs=[IOSpec("y", "float64", [1], "contiguous")],
            semantics={"formula": "y = x"},
        ))
    expired = scheduler.claim_ready("worker", run_id="legacy", lease_seconds=0.001)[0]
    time.sleep(0.01)
    assert expired.lease_expires < time.time()
    expected = [task.to_dict() for task in scheduler.store.tasks()]
    scheduler.store.close()
    def durable_files():
        # SQLite mode=ro can create transient WAL/SHM coordination files; compare
        # the database and owned run artifacts, not SQLite's lock bookkeeping.
        paths = [state, *(tmp_path / "legacy").rglob("*"), *(tmp_path / "managed").rglob("*")]
        return {str(path.relative_to(tmp_path)): path.read_bytes() for path in paths if path.is_file()}

    before = durable_files()

    for arguments, run_id in (((), None), (("--run-id", "legacy"), "legacy"),
                              (("--run-id", "managed"), "managed")):
        listed = _run(state, "list", *arguments)
        assert listed.returncode == 0, listed.stdout + listed.stderr
        payload = json.loads(listed.stdout)
        assert payload == {"command": "list", "run_id": run_id,
                           "tasks": [task for task in expected if run_id is None or task["run_id"] == run_id]}

    assert durable_files() == before
    assert expected[0]["status"] == "running" and expected[0]["attempt"] == 1


def test_removed_scheduler_entry_has_no_compatibility_wrapper() -> None:
    assert not (ROOT / "cli" / "scheduler.py").exists()


def test_discover_requires_environment_proof(tmp_path: Path) -> None:
    state = tmp_path / "adaptation.db"
    report = tmp_path / "gaps.json"
    report.write_text(json.dumps({"entries": []}), encoding="utf-8")
    created = _run(
        state,
        "create",
        "--run-id",
        "run-guard",
        "--model",
        "DeepSeek-V4.1",
        "--backend",
        "kunlun-p800",
    )
    assert created.returncode == 0
    discovered = _run(
        state,
        "discover",
        "--run-id",
        "run-guard",
        "--report",
        str(report),
    )
    assert discovered.returncode == 2
    assert "environment proof is required" in json.loads(discovered.stdout)["error"]


def test_create_run_compatibility_alias(tmp_path: Path) -> None:
    result = _run(
        tmp_path / "state.db",
        "create-run",
        "--run-id",
        "compat",
        "--model",
        "m",
        "--backend",
        "xpu",
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["run"]["run_id"] == "compat"
