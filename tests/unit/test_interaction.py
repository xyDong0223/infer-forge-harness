"""Graph decisions are durable, bounded, evidence-bound and accepted at most once."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from pathlib import Path
from threading import Barrier

import pytest

from engine.brain import DecisionRequest, FailureEvidence
from engine.interaction import (
    accept_graph_decision, canonical_digest, create_graph_handoff,
    current_graph_handoff, finish_graph_decision, validate_retry_parameters,
)
from engine.scheduler import TaskScheduler


@pytest.fixture
def scheduler(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(
        run_id="run", model_id="model", backend="xpu",
        environment={"hardware": "P800"},
        metadata={"evidence_mode": "simulation", "artifact_root": str(tmp_path / "run")},
    )
    scheduler.record_graph_transition("run", "graph_execution_context", {"workflow": "test"})
    yield scheduler
    scheduler.store.close()


def source(scheduler, tmp_path, attempt="attempt-1"):
    root = tmp_path / "run" / "tasks" / "service" / "attempts" / attempt
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "status.json"
    evidence.write_text('{"state":"FAILED"}\n', encoding="utf-8")
    run = scheduler.store.run("run")
    return {
        "node": "service", "task_type": "service_proof", "attempt_id": attempt,
        "artifacts": str(output),
        "source_files": {str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()},
        "execution_context_sha256": canonical_digest(run.metadata["graph_execution_context"]),
        "environment_sha256": canonical_digest(run.environment),
    }


def request(budget=3, history=None):
    return DecisionRequest(
        model="model", backend="xpu",
        failure=FailureEvidence(node="service", state="FAILED", reason="readiness timeout"),
        attempts_remaining=budget, history=history or [],
        available_actions=["RETRY", "RETRY_WITH_PARAMS", "BLOCKED"],
        context={"gpu_memory_utilization": "0.92", "max_model_len": "1024", "pod": "pod-1",
                 "model_path": "/weights", "target_file": "/target.yaml", "_private": "value",
                 "proof_health_interval": "10", "weights": "/checkpoint", "port": "8000",
                 "server_log": "/server.log", "served_model_name": "model"},
    )


def decision(handoff, **changes):
    return {
        "next_action": "RETRY", "diagnosis": "readiness has settled", "facts": ["timeout"],
        "hypotheses": [], "params": {}, "confidence": 0.7,
        "evidence_refs": [next(iter(handoff["source"]["source_files"]))], **changes,
    }


def handoff(scheduler, tmp_path, budget=3):
    return create_graph_handoff(scheduler, "run", source(scheduler, tmp_path), request(budget))


def accept(scheduler, item, payload=None, decision_id="decision-1"):
    return accept_graph_decision(
        scheduler, "run", item["handoff_id"], decision_id, item["source_version"],
        payload if payload is not None else decision(item),
    )


def snapshot(scheduler):
    return list(scheduler.store.db.iterdump())


def test_handoff_publication_and_query_are_idempotent(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    before = snapshot(scheduler)
    assert current_graph_handoff(scheduler, "run") == item
    assert create_graph_handoff(scheduler, "run", item["source"], request()) == item
    assert snapshot(scheduler) == before
    current_graph_handoff(scheduler, "run")["request"]["context"].clear()
    assert current_graph_handoff(scheduler, "run")["request"]["context"]


def test_a_new_handoff_cannot_replace_an_unresolved_one(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    with pytest.raises(ValueError, match="unresolved"):
        create_graph_handoff(scheduler, "run", source(scheduler, tmp_path, "attempt-2"), request())
    assert current_graph_handoff(scheduler, "run") == item


def test_acceptance_commits_before_execution_and_survives_new_process(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    receipt, execute = accept(scheduler, item)
    assert execute is True
    assert receipt["state"] == "executing"
    assert receipt["execution_status"] == "IN_PROGRESS_OR_UNKNOWN"
    assert receipt["remaining_budget"] == 2
    restarted = TaskScheduler(tmp_path / "state.db")
    try:
        before = snapshot(restarted)
        replay, execute = accept(restarted, item)
        assert execute is False and replay == receipt
        assert current_graph_handoff(restarted, "run")["state"] == "executing"
        assert snapshot(restarted) == before
    finally:
        restarted.store.close()


def test_completion_replay_is_readonly_even_after_source_changes(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    accept(scheduler, item)
    outcome = {"status": "RECOVERED", "state": "DEPLOYMENT_READY", "artifacts": "/external/new"}
    receipt = finish_graph_decision(scheduler, "run", item["handoff_id"], "decision-1", outcome)
    assert receipt["state"] == "completed" and receipt["execution_status"] == "FINISHED"
    assert current_graph_handoff(scheduler, "run") is None
    Path(next(iter(item["source"]["source_files"]))).write_text("changed by execution")
    before = snapshot(scheduler)
    assert accept(scheduler, item) == (receipt, False)
    assert finish_graph_decision(scheduler, "run", item["handoff_id"], "decision-1", outcome) == receipt
    assert snapshot(scheduler) == before
    with pytest.raises(ValueError, match="conflicts"):
        finish_graph_decision(scheduler, "run", item["handoff_id"], "decision-1", {"status": "FAILED"})


@pytest.mark.parametrize("change", ["payload", "handoff", "version", "new_id"])
def test_conflicting_or_duplicate_fresh_submissions_reject_without_mutation(scheduler, tmp_path, change):
    item = handoff(scheduler, tmp_path)
    accept(scheduler, item)
    altered = deepcopy(item)
    payload = decision(item)
    decision_id = "decision-1"
    if change == "payload":
        payload["diagnosis"] = "a different instruction"
    if change == "handoff":
        altered["handoff_id"] = "other"
    if change == "version":
        altered["source_version"] = "changed"
    if change == "new_id":
        decision_id = "decision-2"
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="conflicts|not pending"):
        accept(scheduler, altered, payload, decision_id)
    assert snapshot(scheduler) == before


@pytest.mark.parametrize("change", ["file", "context", "environment", "version", "missing_file", "symlink"])
def test_stale_sources_reject_before_acceptance(scheduler, tmp_path, change):
    item = handoff(scheduler, tmp_path)
    path = Path(next(iter(item["source"]["source_files"])))
    if change == "file":
        path.write_text("new failure")
    if change == "context":
        scheduler.record_graph_transition("run", "graph_execution_context", {"workflow": "changed"})
    if change == "environment":
        scheduler.record_environment_failure("run", {}, "environment no longer proven")
    if change == "version":
        item["source_version"] = "older-version"
    if change == "missing_file":
        path.unlink()
    if change == "symlink":
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(replacement)
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="stale|canonical"):
        accept(scheduler, item)
    assert snapshot(scheduler) == before


@pytest.mark.parametrize("changes", [
    {"next_action": "ROLLBACK"}, {"next_action": "RUN_TRIAGE"}, {"next_action": []},
    {"diagnosis": " "}, {"facts": [1]}, {"hypotheses": "guess"},
    {"evidence_refs": []}, {"evidence_refs": ["/unbound/file"]}, {"evidence_refs": [False]},
    {"confidence": True}, {"confidence": float("nan")}, {"confidence": float("inf")},
    {"confidence": 10 ** 400},
    {"params": {"max_model_len": 1}}, {"extra": "ignored would be unsafe"},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"max_model_len": []}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"max_model_len": None}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"max_model_len": float("inf")}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"new_parameter": 1}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"pod": "new-pod"}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"model_path": "/other-model"}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"target_file": "/other.yaml"}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"_private": "new"}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"weights": "/other-checkpoint"}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"server_log": "/other.log"}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"served_model_name": "other"}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"port": 8001}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"max_model_len": 512}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"gpu_memory_utilization": 0.8}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"proof_health_interval": -1}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"proof_health_interval": True}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"proof_health_interval": "1"}},
    {"next_action": "RETRY_WITH_PARAMS", "params": {"proof_health_interval": 10 ** 400}},
])
def test_invalid_decisions_never_consume_budget_or_change_state(scheduler, tmp_path, changes):
    item = handoff(scheduler, tmp_path)
    before = snapshot(scheduler)
    with pytest.raises(ValueError):
        accept(scheduler, item, decision(item, **changes))
    assert snapshot(scheduler) == before
    assert current_graph_handoff(scheduler, "run")["remaining_budget"] == 3


def test_params_are_accepted_only_from_declared_effective_retry_parameters(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    receipt, execute = accept(scheduler, item, decision(
        item, next_action="RETRY_WITH_PARAMS", params={"proof_health_interval": 0.5},
    ))
    assert execute
    assert receipt["decision"]["params"] == {"proof_health_interval": 0.5}


@pytest.mark.parametrize("task_type", ["environment_proof", "service_proof"])
def test_retry_parameter_whitelist_is_per_task_and_requires_declared_context(task_type):
    validate_retry_parameters(task_type, {"proof_health_interval": "10"}, {"proof_health_interval": 0})
    with pytest.raises(ValueError, match="not declared and allowed"):
        validate_retry_parameters(task_type, {}, {"proof_health_interval": 0})
    with pytest.raises(ValueError, match="not declared and allowed"):
        validate_retry_parameters("toy_bringup", {"proof_health_interval": "10"}, {"proof_health_interval": 0})


def test_declared_parameter_for_other_task_cannot_be_submitted(scheduler, tmp_path):
    origin = source(scheduler, tmp_path)
    origin["task_type"] = "toy_bringup"
    item = create_graph_handoff(scheduler, "run", origin, request())
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="not declared and allowed"):
        accept(scheduler, item, decision(item, next_action="RETRY_WITH_PARAMS", params={"proof_health_interval": 0}))
    assert snapshot(scheduler) == before


def test_only_available_actions_are_accepted(scheduler, tmp_path):
    limited = request()
    limited.available_actions = ["BLOCKED"]
    item = create_graph_handoff(scheduler, "run", source(scheduler, tmp_path), limited)
    with pytest.raises(ValueError, match="not available"):
        accept(scheduler, item)


def test_failed_retry_carries_remaining_budget_and_history_into_next_handoff(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path, budget=1)
    receipt, _ = accept(scheduler, item)
    assert receipt["remaining_budget"] == 0
    finish_graph_decision(scheduler, "run", item["handoff_id"], "decision-1", {"status": "REWORK"})
    previous = scheduler.store.run("run").metadata["graph_handoffs"][item["handoff_id"]]
    with pytest.raises(ValueError, match="preserve its remaining budget"):
        create_graph_handoff(scheduler, "run", source(scheduler, tmp_path, "attempt-2"), request(3))
    next_request = request(previous["remaining_budget"], previous["history"])
    next_item = create_graph_handoff(scheduler, "run", source(scheduler, tmp_path, "attempt-2"), next_request)
    assert next_item["state"] == "blocked" and len(next_item["history"]) == 1
    before = snapshot(scheduler)
    with pytest.raises(ValueError):
        accept(scheduler, next_item, decision_id="decision-2")
    assert snapshot(scheduler) == before
    blocked, execute = accept(scheduler, next_item, decision(next_item, next_action="BLOCKED"), "decision-2")
    assert execute and blocked["remaining_budget"] == 0
    finish_graph_decision(scheduler, "run", next_item["handoff_id"], "decision-2", {"status": "BLOCKED"})
    assert current_graph_handoff(scheduler, "run")["state"] == "blocked"


def test_blocked_decision_does_not_consume_budget(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    receipt, _ = accept(scheduler, item, decision(item, next_action="BLOCKED"))
    assert receipt["remaining_budget"] == 3
    finish_graph_decision(scheduler, "run", item["handoff_id"], "decision-1", {"status": "BLOCKED"})
    assert current_graph_handoff(scheduler, "run")["state"] == "blocked"


def test_concurrent_identical_submission_executes_only_once(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    barrier = Barrier(2)

    def submit():
        connection = TaskScheduler(tmp_path / "state.db")
        try:
            barrier.wait()
            return accept(connection, item)
        finally:
            connection.store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: submit(), range(2)))
    assert sorted(execute for _, execute in responses) == [False, True]
    assert responses[0][0] == responses[1][0]
    assert current_graph_handoff(scheduler, "run")["remaining_budget"] == 2


def test_cross_run_and_unknown_handoffs_do_not_mutate_state(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    scheduler.create_run(run_id="other", model_id="model", backend="xpu")
    before = snapshot(scheduler)
    with pytest.raises(KeyError, match="unknown handoff"):
        accept_graph_decision(scheduler, "other", item["handoff_id"], "decision-1",
                              item["source_version"], decision(item))
    with pytest.raises(KeyError, match="unknown run"):
        current_graph_handoff(scheduler, "missing")
    assert snapshot(scheduler) == before


def test_unaccepted_execution_result_is_rejected(scheduler, tmp_path):
    item = handoff(scheduler, tmp_path)
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="accepted decision"):
        finish_graph_decision(scheduler, "run", item["handoff_id"], "not-accepted", {"status": "RECOVERED"})
    assert snapshot(scheduler) == before
