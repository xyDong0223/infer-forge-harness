"""Contract success states must be traversable, not accidental failure edges."""

import json

import pytest

from runners.graph_runner import NODES, node_passed, scheduled_spec


@pytest.mark.parametrize(("task_type", "state"), [
    ("runtime_drift_scan", "DRIFT_CLEAR"),
    ("runtime_drift_scan", "DRIFT_FOUND"),
    ("toy_bringup", "BRINGUP_PASS"),
    ("torch_shim_handoff", "HANDOFF_CLEAR"),
    ("support_matrix", "MATRIX_READY"),
    ("operator_candidate_integration", "OPERATORS_READY"),
])
def test_actual_contract_success_can_advance(tmp_path, task_type, state):
    spec = NODES[task_type]
    path = tmp_path / spec["state_file"]
    path.write_text(json.dumps({"state": state, "validator": {"passed": True, "errors": []}}))
    assert node_passed(tmp_path, spec, 0)
    assert not node_passed(tmp_path, spec, 1)
    path.write_text(json.dumps({"state": state, "validator": {"passed": False, "errors": ["bad"]}}))
    assert not node_passed(tmp_path, spec, 0)


@pytest.mark.parametrize("state", [
    "WAITING_FOR_OPERATORS", "OPERATORS_BLOCKED", "DISPATCH_BLOCKED", "HANDOFF_FOUND",
])
def test_unfinished_work_is_not_graph_success(tmp_path, state):
    spec = NODES["operator_candidate_integration"]
    (tmp_path / spec["state_file"]).write_text(json.dumps({"state": state}))
    assert not node_passed(tmp_path, spec, 0)


def test_scheduled_commands_preserve_the_legacy_node_definition():
    original = NODES["operator_task_dispatch"]
    command = list(original["command"])
    scheduled = scheduled_spec(original, {"scheduler_state": "/external/state.sqlite",
                                         "operator_report": "/external/report.json"})
    assert "--scheduler-state" in scheduled["command"]
    assert "--run-id" in scheduled["command"]
    assert "--operator-report" in scheduled["command"]
    assert original["command"] == command
    assert scheduled_spec(original, {}) is original
