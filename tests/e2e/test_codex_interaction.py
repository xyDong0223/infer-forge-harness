"""Codex-style turns use the production graph, scheduler and evidence gates."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from core.storage import ArtifactStore
from tests.e2e.test_model_adaptation import Scenario, scenario  # noqa: F401


pytestmark = pytest.mark.local_e2e


def toy_failure(case: Scenario, enabled: bool) -> None:
    """Change only the external runtime observation, never a validator result."""
    settings_path = Path(case.env["INFER_FORGE_E2E_SETTINGS"])
    settings = json.loads(settings_path.read_text())
    settings["toy_failure"] = enabled
    ArtifactStore(settings_path.parent).write_json(settings_path.name, settings, overwrite=True)


def advance(case: Scenario, expected=0):
    return json.loads(case.adaptation("advance", "--run-id", case.run_id, expected=expected).stdout)


def failed_toy(case: Scenario, budget=3) -> dict:
    toy_failure(case, True)
    graph = case.graph("--interaction-mode", "codex", "--recovery-budget", str(budget), expected=3)
    assert '"reason_code": "WAITING_FOR_OPERATORS"' in graph.stdout
    assert not any(fact["kind"] == "ToyBringupReport" for fact in case.facts())
    assert case.context()["handoff"] is None
    for stage in ("torch", "xpu", "integration"):
        case.worker(stage)
    started = time.monotonic()
    result = advance(case, expected=4)
    # The old AgentBrain waits 600 seconds by default. This path must return
    # promptly after the runtime failure, with no file-response poll loop.
    assert time.monotonic() - started < 30
    item = result["handoff"]
    assert item["state"] == "pending"
    assert item["source"]["task_type"] == "toy_bringup"
    assert item["remaining_budget"] == budget
    assert item["request"]["failure"]["state"] != "TOY_READY"
    assert Path(item["source"]["artifacts"]).is_dir()
    return item


def decision_file(case: Scenario, item: dict, *, filename="decision.json", action="RETRY") -> Path:
    source = item["source"]
    evidence = str(Path(source["artifacts"]) / source["state_file"])
    assert evidence in source["source_files"]
    return ArtifactStore(case.root / "controller").write_json(filename, {
        "next_action": action,
        "diagnosis": "The simulated runtime reports a decode failure; rerun the same prepared environment.",
        "facts": ["The preserved toy report records incomplete decode."],
        "hypotheses": ["The external simulated fault may be transient."],
        "params": {}, "confidence": 0.7, "evidence_refs": [evidence],
    })


def submit(case: Scenario, item: dict, path: Path, *, decision_id="decision-1", expected=0,
           version=None) -> dict:
    return json.loads(case.adaptation(
        "submit-decision", "--run-id", case.run_id,
        "--handoff-id", item["handoff_id"], "--decision-id", decision_id,
        "--expected-version", version or item["source_version"], "--decision", str(path),
        expected=expected,
    ).stdout)


def test_codex_nonblocking_decision_restart_and_delivery(scenario):
    item = failed_toy(scenario)
    before_query = scenario.persisted_snapshot()
    context = scenario.context()
    assert context["handoff"] == item
    assert context["progress"]["reason_code"] == "GRAPH_DECISION_REQUIRED"
    assert scenario.persisted_snapshot() == before_query
    # These commands are separate processes with no prior in-memory state.
    pending = advance(scenario, expected=4)
    assert pending["handoff"] == item
    assert scenario.persisted_snapshot() == before_query
    request = decision_file(scenario, item)
    toy_failure(scenario, False)
    recovered = submit(scenario, item, request)
    assert recovered["receipt"]["outcome"]["status"] == "RECOVERED"
    assert recovered["replayed"] is False
    assert recovered["receipt"]["remaining_budget"] == 2
    new_artifacts = Path(recovered["receipt"]["outcome"]["final_artifacts"])
    assert new_artifacts != Path(item["source"]["artifacts"])
    assert (new_artifacts / item["source"]["state_file"]).is_file()
    before_replay = scenario.persisted_snapshot()
    replay = submit(scenario, item, request)
    assert replay["replayed"] is True
    assert replay["receipt"] == recovered["receipt"]
    assert scenario.persisted_snapshot() == before_replay
    finished = advance(scenario)
    assert finished["progress"]["state"] == "COMPLETED"
    state = scenario.status()
    delivery = state["run"]["metadata"]["graph_delivery"]
    assert delivery["state"] == "SIMULATION_PASS"
    assert delivery["evidence_mode"] == "simulation"
    assert all(task["status"] == "succeeded" for task in state["tasks"])
    before_complete_replay = scenario.persisted_snapshot()
    assert advance(scenario)["progress"]["state"] == "COMPLETED"
    assert scenario.persisted_snapshot() == before_complete_replay


def test_codex_stale_and_conflicting_decisions_are_rejected(scenario):
    item = failed_toy(scenario)
    request = decision_file(scenario, item)
    before = scenario.persisted_snapshot()
    rejected = submit(scenario, item, request, version="obsolete-source-version", expected=2)
    assert "source_version" in rejected["error"]
    assert scenario.persisted_snapshot() == before
    # Explicit parameter/input resupply cannot route around a pending decision.
    rejected_settings = scenario.adaptation(
        "advance", "--run-id", scenario.run_id, "--set", "max_model_len=64", expected=2,
    )
    assert "unresolved handoff" in rejected_settings.stdout
    assert scenario.persisted_snapshot() == before
    toy_failure(scenario, False)
    recovered = submit(scenario, item, request)
    changed = json.loads(request.read_text())
    changed["diagnosis"] = "A different conclusion cannot reuse the accepted identity."
    conflict = ArtifactStore(scenario.root / "controller").write_json("conflicting.json", changed)
    before = scenario.persisted_snapshot()
    rejected = submit(scenario, item, conflict, expected=2)
    assert "conflicts" in rejected["error"]
    assert scenario.persisted_snapshot() == before
    rejected = submit(scenario, item, request, decision_id="new-decision-for-old-source", expected=2)
    assert "not pending" in rejected["error"]
    assert scenario.persisted_snapshot() == before
    assert submit(scenario, item, request)["receipt"] == recovered["receipt"]


def test_codex_source_evidence_change_rejects_decision(scenario):
    item = failed_toy(scenario)
    request = decision_file(scenario, item)
    # This is corruption of actual failing evidence, not a fabricated success.
    evidence = Path(item["source"]["artifacts"]) / item["source"]["state_file"]
    original = evidence.read_text()
    ArtifactStore(evidence.parent).write_text(evidence.name, original + "\n", overwrite=True)
    before = scenario.persisted_snapshot()
    rejected = submit(scenario, item, request, expected=2)
    assert "source evidence changed" in rejected["error"]
    assert scenario.persisted_snapshot() == before
    assert scenario.context()["handoff"]["remaining_budget"] == 3
    assert scenario.status()["run"]["metadata"].get("graph_decisions", {}) == {}


def test_codex_recovery_budget_survives_failure_and_restart(scenario):
    item = failed_toy(scenario, budget=2)
    first_request = decision_file(scenario, item)
    first = submit(scenario, item, first_request, expected=4)
    assert first["receipt"]["state"] == "completed"
    assert first["receipt"]["outcome"]["status"] == "REWORK"
    next_item = first["handoff"]
    assert next_item["handoff_id"] != item["handoff_id"]
    assert next_item["remaining_budget"] == 1
    assert len(next_item["history"]) == 1
    before = scenario.persisted_snapshot()
    assert advance(scenario, expected=4)["handoff"] == next_item
    assert scenario.persisted_snapshot() == before
    second_request = decision_file(scenario, next_item, filename="decision-2.json")
    second = submit(scenario, next_item, second_request, decision_id="decision-2", expected=2)
    exhausted = second["handoff"]
    assert exhausted["state"] == "blocked"
    assert exhausted["remaining_budget"] == 0
    assert len(exhausted["history"]) == 2
    before = scenario.persisted_snapshot()
    assert advance(scenario, expected=2)["handoff"] == exhausted
    assert scenario.persisted_snapshot() == before
    third_request = decision_file(scenario, exhausted, filename="decision-3.json")
    rejected = submit(scenario, exhausted, third_request, decision_id="decision-3", expected=2)
    assert "not pending" in rejected["error"] or "budget exhausted" in rejected["error"]
    assert scenario.persisted_snapshot() == before
    assert "graph_delivery" not in scenario.status()["run"]["metadata"]
