"""Pre-descriptor pending decisions cannot authorize a changed current command."""

import json
from pathlib import Path

import pytest

from core.storage import locate_attempt
from engine.brain import DecisionRequest, FailureEvidence
from engine.interaction import canonical_digest, create_graph_handoff, current_graph_handoff
from runners import graph_runner
from runners.codex_interaction import submit_graph_decision
from tests.unit.test_graph_interaction import interactive, invoke  # noqa: F401


def _legacy_create(args, *, node, spec, context, artifacts, environment, state,
                   task_type, skill, bridge, remaining_budget=None, history=None):
    """The old producer boundary: bind original evidence, without P3 snapshots."""
    attempt = locate_attempt(artifacts)
    paths = [args.workflow, args.journal, args.loop_state,
             attempt.root / ".attempt.json", attempt.root / "manifest.json"]
    for directory in (attempt.input, attempt.output, attempt.logs):
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    for command in json.loads((attempt.input / "commands.json").read_text()):
        paths.extend(Path(value) for value in command[2:]
                     if Path(value).is_absolute() and Path(value).is_file())
    run = bridge.scheduler.store.run(args.run_id)
    source = {
        "node": node, "task_type": task_type, "attempt_id": attempt.identity["attempt_id"],
        "artifacts": str(artifacts), "state": state, "state_file": spec["state_file"],
        "source_files": {str(path.resolve()): graph_runner.file_digest(path)
                         for path in paths if path.is_file()},
        "execution_context_sha256": canonical_digest(run.metadata["graph_execution_context"]),
        "environment_sha256": canonical_digest(run.environment),
    }
    return create_graph_handoff(bridge.scheduler, args.run_id, source, DecisionRequest(
        model=args.subject, backend="p800",
        failure=FailureEvidence(node, state, "observed original failure", [str(artifacts)], environment),
        context=dict(context), skill=skill, history=list(history or []),
        available_actions=["RETRY", "RETRY_WITH_PARAMS", "BLOCKED"],
        attempts_remaining=args.recovery_budget if remaining_budget is None else remaining_budget,
    ))


@pytest.fixture
def legacy(interactive, monkeypatch):
    monkeypatch.setattr(graph_runner, "create_interactive_handoff", _legacy_create)
    assert invoke(monkeypatch, interactive.argv) == 4
    item = current_graph_handoff(interactive.scheduler, "r")
    attempt = locate_attempt(item["source"]["artifacts"])
    assert not (attempt.input / "task_execution.json").exists()
    assert not (attempt.input / "task_memory_snapshot.json").exists()
    assert not any(path.endswith(("task.yaml", "task_execution.json", "task_memory_snapshot.json"))
                   for path in item["source"]["source_files"])
    interactive.handoff = item
    interactive.decision = {
        "next_action": "RETRY", "diagnosis": "Retry the observed transient readiness failure",
        "evidence_refs": [str(attempt.output / "status.json")],
    }
    return interactive


def _submit(case, decision_id="legacy-decision"):
    return submit_graph_decision(case.scheduler, "r", case.handoff["handoff_id"], decision_id,
                                 case.handoff["source_version"], case.decision)


def test_same_legacy_argv_can_retry_and_replay_after_descriptor_drift(legacy, monkeypatch):
    legacy.outcomes[:] = ["ENVIRONMENT_READY"]
    result = _submit(legacy)
    assert result["receipt"]["execution_status"] == "FINISHED"
    assert result["receipt"]["remaining_budget"] == 2
    assert len(legacy.calls) == 2
    monkeypatch.setitem(graph_runner.NODES, "environment_proof", {
        **graph_runner.NODES["environment_proof"], "command": ["python3", "changed.py"],
    })
    before = list(legacy.scheduler.store.db.iterdump())
    replay = _submit(legacy)
    assert replay["replayed"] and replay["receipt"] == result["receipt"]
    assert len(legacy.calls) == 2
    assert list(legacy.scheduler.store.db.iterdump()) == before


@pytest.mark.parametrize("change", ["argv", "fanout"])
def test_legacy_drift_rejected_before_acceptance_without_mutation(legacy, monkeypatch, change):
    spec = dict(graph_runner.NODES["environment_proof"])
    if change == "argv":
        spec["command"] = [*spec["command"], "--changed-policy"]
    else:
        spec["fan_out"] = {"list": ["must-not-execute"]}
        monkeypatch.setattr(graph_runner, "fan_out_items", lambda *a, **k: pytest.fail("must not list"))
    monkeypatch.setitem(graph_runner.NODES, "environment_proof", spec)
    before = list(legacy.scheduler.store.db.iterdump())
    root = locate_attempt(legacy.handoff["source"]["artifacts"]).root.parents[3]
    paths = sorted(str(path) for path in root.rglob("*"))
    with pytest.raises(ValueError, match="retry command changed|legacy fan-out"):
        _submit(legacy)
    assert list(legacy.scheduler.store.db.iterdump()) == before
    assert sorted(str(path) for path in root.rglob("*")) == paths
    assert current_graph_handoff(legacy.scheduler, "r") == legacy.handoff
    assert len(legacy.calls) == 1


def test_params_apply_only_after_original_argv_check(legacy):
    legacy.outcomes[:] = ["ENVIRONMENT_READY"]
    legacy.decision.update(next_action="RETRY_WITH_PARAMS", params={"proof_health_interval": 0.5})
    result = _submit(legacy)
    assert result["receipt"]["execution_status"] == "FINISHED"
    assert len(legacy.calls) == 2
    assert legacy.calls[0][legacy.calls[0].index("--health-interval-seconds") + 1] == "1"
    assert legacy.calls[1][legacy.calls[1].index("--health-interval-seconds") + 1] == "0.5"


def test_blocked_decision_does_not_require_current_descriptor(legacy, monkeypatch):
    monkeypatch.delitem(graph_runner.NODES, "environment_proof")
    legacy.decision["next_action"] = "BLOCKED"
    result = _submit(legacy)
    assert result["handoff"]["state"] == "blocked"
    assert result["receipt"]["remaining_budget"] == 3
    assert len(legacy.calls) == 1


def test_new_task_snapshot_rejects_current_descriptor_drift(interactive, monkeypatch, tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("kind: Task\n")
    monkeypatch.setitem(graph_runner.NODES, "environment_proof", {
        **graph_runner.NODES["environment_proof"], "task_path": str(task),
        "task_sha256": graph_runner.file_digest(task),
    })
    assert invoke(monkeypatch, interactive.argv) == 4
    interactive.handoff = current_graph_handoff(interactive.scheduler, "r")
    interactive.decision = {
        "next_action": "RETRY", "diagnosis": "Observed readiness failure",
        "evidence_refs": [str(Path(interactive.handoff["source"]["artifacts"]) / "status.json")],
    }
    monkeypatch.setitem(graph_runner.NODES, "environment_proof", {
        **graph_runner.NODES["environment_proof"], "command": ["python3", "changed.py"],
    })
    before = list(interactive.scheduler.store.db.iterdump())
    with pytest.raises(ValueError, match="descriptor changed"):
        _submit(interactive)
    assert list(interactive.scheduler.store.db.iterdump()) == before
    assert len(interactive.calls) == 1
