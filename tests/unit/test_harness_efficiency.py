"""Accounting boundaries: overlap, retry, incomplete telemetry and tampering."""

from copy import deepcopy

import pytest

from operations.validation.harness_efficiency import evaluate
from validators.harness_efficiency_validator import validate_efficiency


def snapshot():
    events = []

    def event(kind, timestamp, task=None, **payload):
        events.append({"event_id": f"e{len(events)}", "event_type": kind,
                       "timestamp": timestamp, "task_id": task, "payload": payload})

    event("run_created", 100)
    event("task_created", 101, "a", stage="torch")
    event("task_created", 102, "b", stage="torch")
    event("task_claimed", 103, "a", stage="torch", attempt=1)
    event("task_claimed", 105, "b", stage="torch", attempt=1)
    event("task_failed", 108, "a", stage="torch")
    event("diagnosis_task_created", 108, "d", stage="diagnosis")
    event("task_claimed", 109, "d", stage="diagnosis", attempt=1)
    event("task_succeeded", 110, "b", stage="torch")
    event("task_succeeded", 111, "d", stage="diagnosis")
    event("diagnosis_applied", 112, "d", source_task_id="a", next_action="RETRY")
    event("task_claimed", 114, "a", stage="torch", attempt=2)
    event("task_succeeded", 118, "a", stage="torch")
    event("graph_progress", 120, status="WAITING")
    return {
        "schema_version": 1, "captured_at": 125,
        "run": {"run_id": "test", "evidence_mode": "simulation"},
        "window": {"started_at": 100, "ended_at": 120, "scope": "snapshot",
                   "endpoint": "last_persisted_event"},
        "outcome": {"graph_status": "WAITING"}, "events": events,
        "graph_blocks": [
            {"block_id": "g1", "sub_target": "service", "state": "READY",
             "started_at": 100, "finished_at": 106, "reused": False, "activity": "execution", "artifacts": []},
            {"block_id": "g2", "sub_target": "service", "state": "READY",
             "started_at": 116, "finished_at": 119, "reused": False, "activity": "execution", "artifacts": []},
            {"block_id": "g3", "sub_target": "intake", "state": "READY",
             "started_at": 119, "finished_at": 119.5, "reused": True, "activity": "reuse", "artifacts": []},
        ], "open_graph_block": None, "warnings": [],
    }


def test_overlapping_intervals_retry_and_required_regression_are_distinct():
    source = snapshot()
    report = evaluate(source)
    metrics = report["metrics"]
    assert metrics["elapsed_seconds"] == 20
    # [100,111] union [114,119.5], rather than summing nested/parallel work.
    assert metrics["recorded_wall_seconds"] == 16.5
    assert metrics["unobserved_wall_seconds"] == 3.5
    assert metrics["worker_attempts"] == 4
    assert metrics["worker_retries"] == 1
    assert metrics["failed_worker_attempts"] == 1
    assert metrics["diagnosis_attempts"] == 1
    assert metrics["repeated_graph_executions"] == 1
    assert metrics["graph_reuses"] == 1
    assert metrics["known_queue_wait_seconds"] == 8
    assert report["metrics"]["token_usage"] is None
    assert validate_efficiency(report, source) == []
    source["captured_at"] += 99999
    assert evaluate(source) == report


def test_reclaimed_worker_has_unknown_end_not_next_claim_time():
    source = snapshot()
    source["events"] = [event for event in source["events"] if event["event_type"] != "task_failed"]
    report = evaluate(source)
    first = report["worker_attempts"][0]
    assert first["duration_seconds"] is None
    assert first["finished_at"] is None
    assert report["metrics"]["unclosed_worker_attempts"] == 1
    assert report["warnings"]
    assert validate_efficiency(report, source) == []


def test_scheduler_handoffs_are_not_repeated_node_executions():
    source = snapshot()
    source["graph_blocks"][1]["activity"] = "bookkeeping"
    report = evaluate(source)
    assert report["metrics"]["repeated_graph_executions"] == 0
    assert report["metrics"]["graph_bookkeeping_blocks"] == 1
    assert validate_efficiency(report, source) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 99, True])
def test_invalid_timestamps_are_rejected(value):
    source = snapshot()
    source["graph_blocks"][0]["finished_at"] = value
    with pytest.raises(ValueError, match="timestamps"):
        evaluate(source)


def test_validator_rejects_invented_time_cost_and_references():
    source = snapshot()
    valid = evaluate(source)
    for key, value in (("recorded_wall_seconds", 40), ("worker_retries", 0), ("cost", 0)):
        changed = deepcopy(valid)
        changed["metrics"][key] = value
        assert validate_efficiency(changed, source)
    changed = deepcopy(valid)
    changed["findings"][0]["references"] = ["not-recorded"]
    assert validate_efficiency(changed, source)
    changed = deepcopy(valid)
    changed["stages"][0]["closed_interval_seconds"] = 9999
    assert validate_efficiency(changed, source)
    changed = deepcopy(valid)
    changed["stages"] = []
    assert validate_efficiency(changed, source)


def test_validator_rejects_omitted_completions_and_queue_evidence():
    source = snapshot()
    omitted = deepcopy(source)
    omitted["events"] = [e for e in source["events"] if not (
        e["event_type"] == "task_succeeded" and e["task_id"] == "b"
    )]
    assert validate_efficiency(evaluate(omitted), source)
    changed = evaluate(source)
    for attempt in changed["worker_attempts"]:
        attempt["queue_event_id"] = attempt["queue_wait_seconds"] = None
    changed["metrics"].update(known_queue_wait_seconds=0, unknown_queue_wait_count=4)
    assert validate_efficiency(changed, source)
