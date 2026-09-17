"""Agent context is a projection, never an implicit claim or recovery."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from core.paths import REPO_ROOT
from engine import IOSpec, OperatorSpec, TaskScheduler
from engine.context import run_context
from engine.result_validation import STAGE_EVIDENCE
from tests.scheduler_helpers import simulation_result


@pytest.fixture
def scheduler(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.sqlite")
    for run_id in ("r", "other"):
        scheduler.create_run(
            run_id=run_id, model_id="model", model_revision="m1", plugin_revision="p1",
            backend="xpu", metadata={"evidence_mode": "simulation", "artifact_root": str(tmp_path / run_id)},
        )
        scheduler.discover_operator(run_id, OperatorSpec(
            operator_id="identity", model_id="model", model_revision="m1", plugin_revision="p1",
            backend="xpu", inputs=[IOSpec("x", "float32", [1], "contiguous")],
            outputs=[IOSpec("y", "float32", [1], "contiguous")], semantics={"op": "identity"},
        ))
    yield scheduler
    scheduler.store.close()


def invoke(state, *args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli/adaptation.py"), "--state", str(state), *args],
        capture_output=True, text=True, check=False, timeout=30,
    )


def snapshot(scheduler, tmp_path):
    return {
        "db": list(scheduler.store.db.iterdump()),
        "files": {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                  for run_id in ("r", "other") for p in (tmp_path / run_id).rglob("*") if p.is_file()},
    }


def test_pending_context_is_readonly_and_uses_existing_requirements(scheduler, tmp_path):
    task = scheduler.store.tasks("r")[0]
    before = snapshot(scheduler, tmp_path)
    response = invoke(scheduler.store.path, "context", "--run-id", "r", "--task-id", task.task_id)
    assert response.returncode == 0, response.stderr + response.stdout
    view = json.loads(response.stdout)
    packet = view["task_context"]
    assert packet["operator_spec"] == task.input["operator_spec"]
    assert packet["run_identity"]["model_revision"] == "m1"
    assert packet["acceptance"]["required_evidence"] == list(STAGE_EVIDENCE["torch"])
    assert packet["acceptance"]["output_directory"] is None
    assert packet["attempt"] == 0
    assert view["claimed_packet"] is None
    assert view["restart"] == {"execution_context": None, "resume_command": None}
    assert "lease_token" not in response.stdout
    assert snapshot(scheduler, tmp_path) == before
    schema = yaml.safe_load((REPO_ROOT / "contracts/agent_context.schema.yaml").read_text())
    assert set(schema["required"]) <= view.keys()
    assert set(schema["$defs"]["task_packet"]["required"]) <= packet.keys()


def test_claim_saves_packet_and_query_keeps_claim_snapshot_frozen(scheduler, tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("engine.scheduler.time.time", lambda: now[0])
    first = scheduler.claim_ready("worker", run_id="r", lease_seconds=1)[0]
    path = Path(first.input["workspace"]["input"]) / "task.json"
    original = path.read_bytes()
    packet = json.loads(original)
    assert packet["input"] == first.input
    assert packet["task_id"] == first.task_id
    assert packet["stage"] == "torch" and packet["attempt"] == 1
    assert "lease_token" not in packet
    before = snapshot(scheduler, tmp_path)
    response = invoke(scheduler.store.path, "context", "--run-id", "r", "--task-id", first.task_id)
    assert response.returncode == 0
    view = json.loads(response.stdout)
    assert view["progress"]["reason_code"] == "LEASE_EXPIRED"
    assert view["claimed_packet"] == {"path": str(path), "sha256": hashlib.sha256(original).hexdigest()}
    assert view["task_context"]["evidence_revalidated"] is False
    assert snapshot(scheduler, tmp_path) == before
    now[0] += 2
    second = scheduler.claim_ready("next", run_id="r", task_id=first.task_id)[0]
    assert second.attempt == 2
    assert second.input["workspace"] != first.input["workspace"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("stage,upstream", [("xpu", ["torch"]), ("integration", ["torch", "xpu"])])
def test_upstream_results_are_bound_to_run_and_operator(scheduler, tmp_path, stage, upstream):
    for completed_stage in upstream:
        claimed = scheduler.claim_ready("worker", run_id="r", stage=completed_stage)[0]
        result = simulation_result(claimed, tmp_path)
        scheduler.complete(claimed.task_id, "worker", result, claimed.lease_token)
    task = next(t for t in scheduler.store.tasks("r") if t.stage == stage)
    packet = run_context(scheduler.store, "r", task.task_id)["task_context"]
    assert [t["stage"] for t in packet["upstream_tasks"]] == upstream
    assert all(t["run_id"] == "r" and t["output"]["evidence_sha256"] for t in packet["upstream_tasks"])
    assert packet["acceptance"]["required_evidence"] == list(STAGE_EVIDENCE[stage])


def test_diagnosis_context_has_source_failure_and_canonical_spec(scheduler):
    claimed = scheduler.claim_ready("worker", run_id="r")[0]
    scheduler.fail(claimed.task_id, "worker", "observed operator failure", claimed.lease_token)
    diagnosis = scheduler.claim_ready("diagnoser", run_id="r", stage="diagnosis")[0]
    response = invoke(scheduler.store.path, "context", "--run-id", "r", "--task-id", diagnosis.task_id)
    assert response.returncode == 0
    packet = json.loads(response.stdout)["task_context"]
    assert packet["operator_spec"] == claimed.input["operator_spec"]
    assert packet["input"]["bug_report"]["message"] == "observed operator failure"
    assert packet["upstream_tasks"][0]["task_id"] == claimed.task_id
    assert packet["upstream_tasks"][0]["status"] == "failed"
    assert packet["upstream_tasks"][0]["input"]["workspace"] == claimed.input["workspace"]
    assert "lease_token" not in response.stdout
    failed = invoke(scheduler.store.path, "context", "--run-id", "r", "--task-id", claimed.task_id)
    assert failed.returncode == 0  # A failed task is still a successful read.


def test_query_returns_saved_restart_context_without_revalidation(scheduler):
    execution = {"schema_version": 1, "run_id": "r", "workflow": "/recorded/workflow.yaml"}
    scheduler.record_graph_transition("r", "graph_execution_context", execution)
    scheduler.record_graph_transition("r", "graph_progress", {"resume_command": ["recorded", "--resume"]})
    response = invoke(scheduler.store.path, "context", "--run-id", "r")
    assert response.returncode == 0
    view = json.loads(response.stdout)
    assert view["restart"] == {"execution_context": execution, "resume_command": ["recorded", "--resume"]}
    assert view["task_context"] is None and view["evidence_revalidated"] is False
    assert {t["run_id"] for t in view["tasks"]} == {"r"}


@pytest.mark.parametrize("bad_scope", ["unknown-run", "unknown-task", "foreign-task"])
def test_rejected_query_does_not_mutate(scheduler, tmp_path, bad_scope):
    args = ["context", "--run-id", "missing" if bad_scope == "unknown-run" else "r"]
    if bad_scope != "unknown-run":
        args += ["--task-id", "missing" if bad_scope == "unknown-task" else scheduler.store.tasks("other")[0].task_id]
    before = snapshot(scheduler, tmp_path)
    result = invoke(scheduler.store.path, *args)
    assert result.returncode == 2
    assert json.loads(result.stdout)["progress"]["reason_code"] == "COMMAND_REJECTED"
    assert snapshot(scheduler, tmp_path) == before


def test_missing_database_is_not_created(tmp_path):
    state = tmp_path / "absent" / "state.sqlite"
    result = invoke(state, "context", "--run-id", "r")
    assert result.returncode == 2
    assert not state.parent.exists()
