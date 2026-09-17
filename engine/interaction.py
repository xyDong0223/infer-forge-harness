"""Durable, nonblocking Graph decisions; execution remains the runner's job.

The acceptance transition commits before a caller starts any external action.
An accepted decision without a completion receipt is deliberately not retried:
its execution may still be running, or its response may have been lost.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4

from engine.brain import Decision, DecisionRequest


INTERACTIVE_ACTIONS = frozenset({"RETRY", "RETRY_WITH_PARAMS", "BLOCKED"})
PROTECTED_RETRY_KEYS = frozenset({
    "artifacts", "artifact_root", "attempt", "subject", "target", "target_file",
    "contract_instance", "user_id", "pod", "_environment_pod", "environment_text",
    "environment", "environment_fingerprint", "run_id", "journal", "loop_state",
    "scheduler_state", "operator_report", "shim_registry", "evidence_mode",
    "model", "model_id", "model_revision", "plugin_revision", "backend", "hardware",
    "skill", "method", "workflow", "workflow_sha256", "decision_id", "handoff_id",
    "weights", "server_log", "served_model_name", "port",
})
RETRY_PARAMETER_KEYS = {
    "environment_proof": frozenset({"proof_health_interval"}),
    "service_proof": frozenset({"proof_health_interval"}),
}


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def canonical_digest(value: Any) -> str:
    """Hash JSON content, rejecting nonfinite values instead of normalizing them."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_retry_parameters(task_type: str, context: dict, params: dict) -> None:
    """Allow only parameters with a reviewed, effective command mapping.

    Most Graph context fields identify inputs, methods or resources. Merely
    appearing in context does not make a field a safe retry-time tuning knob.
    """
    if not isinstance(params, dict) or any(not isinstance(key, str) for key in params):
        raise ValueError("params must be an object with string keys")
    allowed = RETRY_PARAMETER_KEYS.get(task_type, frozenset())
    for key, value in params.items():
        if key not in context or key not in allowed:
            raise ValueError(f"retry parameter is not declared and allowed for {task_type}: {key}")
        if not _finite_number(value) or value < 0:
            raise ValueError(f"retry parameter must be a finite nonnegative number: {key}")


def _run(scheduler, run_id: str):
    run = scheduler.store.run(run_id)
    if run is None:
        raise KeyError(f"unknown run: {run_id}")
    return run


def _identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")


def active_graph_handoff(run) -> dict | None:
    """Project an unresolved handoff from an already-loaded run, without I/O."""
    records = run.metadata.get("graph_handoffs", {}).values()
    active = [record for record in records if record["state"] in {
        "pending", "executing", "blocked",
    }]
    return deepcopy(max(active, key=lambda item: item["created_at"])) if active else None


def current_graph_handoff(scheduler, run_id: str) -> dict | None:
    """Read the current unresolved handoff without allocating or recovering work."""
    return active_graph_handoff(_run(scheduler, run_id))


def _validate_source(run, source: dict) -> None:
    if not isinstance(source, dict):
        raise ValueError("handoff source must be an object")
    for key in ("node", "task_type", "attempt_id"):
        _identifier(source.get(key), f"source.{key}")
    if not isinstance(source.get("artifacts"), str) or not Path(source["artifacts"]).is_absolute():
        raise ValueError("source.artifacts must be an absolute directory path")
    files = source.get("source_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source_files must bind at least one persisted evidence file")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str) or not re.fullmatch(
            r"[a-f0-9]{64}", expected,
        ):
            raise ValueError("source_files requires absolute paths and SHA-256 values")
        path = Path(name)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or str(path.resolve()) != name):
            raise ValueError(f"source evidence is not a regular canonical file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"stale handoff: source evidence changed: {name}")
    if source.get("execution_context_sha256") != canonical_digest(
        run.metadata.get("graph_execution_context"),
    ):
        raise ValueError("stale handoff: graph execution context changed")
    if source.get("environment_sha256") != canonical_digest(run.environment):
        raise ValueError("stale handoff: run environment changed")


def create_graph_handoff(scheduler, run_id: str, source: dict,
                         request: DecisionRequest) -> dict:
    """Freeze one failed Graph attempt; repeated publication is idempotent."""
    if (isinstance(request.attempts_remaining, bool)
            or not isinstance(request.attempts_remaining, int)
            or request.attempts_remaining < 0):
        raise ValueError("attempts_remaining must be a nonnegative integer")
    if not set(request.available_actions) <= INTERACTIVE_ACTIONS:
        raise ValueError("interactive request advertises an unsupported action")
    request_body = request.to_dict()
    source_version = canonical_digest({"source": source, "request": request_body})
    with scheduler.store.transaction():
        run = _run(scheduler, run_id)
        records = deepcopy(run.metadata.get("graph_handoffs", {}))
        for record in records.values():
            if record["source_version"] == source_version:
                return deepcopy(record)
        if active_graph_handoff(run) is not None:
            raise ValueError("an unresolved Graph handoff already exists for this run")
        prior = [record for record in records.values()
                 if record["source"]["node"] == source.get("node")]
        latest = max(prior, key=lambda item: item["created_at"]) if prior else None
        if (latest is not None and latest["state"] == "completed"
                and latest.get("outcome", {}).get("status") == "REWORK"
                and (request.attempts_remaining != latest["remaining_budget"]
                     or request.history != latest["history"])):
            raise ValueError("a failed recovery must preserve its remaining budget and history")
        _validate_source(run, source)
        handoff_id = uuid4().hex
        record = {
            "schema_version": 1, "handoff_id": handoff_id, "run_id": run_id,
            "source_kind": "graph", "source_version": source_version,
            "source": deepcopy(source), "request": request_body,
            "remaining_budget": request.attempts_remaining,
            "history": deepcopy(request.history),
            "state": "pending" if request.attempts_remaining > 0 else "blocked",
            "created_at": time.time(),
        }
        records[handoff_id] = record
        scheduler.record_graph_transition(run_id, "graph_handoffs", records)
        return deepcopy(record)


def _strict_decision(payload: dict, handoff: dict) -> Decision:
    if not isinstance(payload, dict):
        raise ValueError("decision must be an object")
    allowed_fields = {"next_action", "diagnosis", "facts", "hypotheses", "params",
                      "confidence", "evidence_refs"}
    if set(payload) - allowed_fields:
        raise ValueError("decision contains unknown fields")
    action = payload.get("next_action")
    if (not isinstance(action, str) or action not in INTERACTIVE_ACTIONS
            or action not in handoff["request"]["available_actions"]):
        raise ValueError("decision action is not available for this handoff")
    diagnosis = payload.get("diagnosis")
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        raise ValueError("diagnosis must be a nonempty string")
    for key in ("facts", "hypotheses", "evidence_refs"):
        values = payload.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{key} must be a list of nonempty strings")
    refs = payload.get("evidence_refs", [])
    if not refs or any(ref not in handoff["source"]["source_files"] for ref in refs):
        raise ValueError("evidence_refs must cite files bound in the handoff source_files")
    confidence = payload.get("confidence", 0.0)
    if not _finite_number(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a finite number in [0, 1]")
    params = payload.get("params", {})
    if not isinstance(params, dict) or any(not isinstance(key, str) for key in params):
        raise ValueError("params must be an object with string keys")
    if action != "RETRY_WITH_PARAMS" and params:
        raise ValueError("only RETRY_WITH_PARAMS accepts parameters")
    if action == "RETRY_WITH_PARAMS" and not params:
        raise ValueError("RETRY_WITH_PARAMS requires at least one allowed parameter")
    validate_retry_parameters(handoff["source"]["task_type"], handoff["request"]["context"], params)
    return Decision.from_dict(payload)


def accept_graph_decision(scheduler, run_id: str, handoff_id: str, decision_id: str,
                          expected_source_version: str, decision: dict) -> tuple[dict, bool]:
    """Atomically accept once, returning ``(receipt, execute_now)``.

    Replay is checked before source freshness: lost responses remain recoverable
    after execution has legitimately changed the source Journal or environment.
    """
    for name, value in (("handoff_id", handoff_id), ("decision_id", decision_id),
                        ("expected_source_version", expected_source_version)):
        _identifier(value, name)
    payload_sha256 = canonical_digest(decision)
    with scheduler.store.transaction():
        run = _run(scheduler, run_id)
        decisions = deepcopy(run.metadata.get("graph_decisions", {}))
        existing = decisions.get(decision_id)
        if existing is not None:
            if (existing["handoff_id"] != handoff_id
                    or existing["source_version"] != expected_source_version
                    or existing["payload_sha256"] != payload_sha256):
                raise ValueError("decision_id conflicts with an already accepted payload or source")
            return deepcopy(existing), False
        handoffs = deepcopy(run.metadata.get("graph_handoffs", {}))
        if handoff_id not in handoffs:
            raise KeyError(f"unknown handoff for run {run_id}: {handoff_id}")
        handoff = handoffs[handoff_id]
        if handoff["source_version"] != expected_source_version:
            raise ValueError("stale handoff: expected source_version does not match")
        candidate = _strict_decision(decision, handoff)
        exhausted = handoff["remaining_budget"] == 0
        if handoff["state"] != "pending" and not (
            exhausted and handoff["state"] == "blocked" and not handoff.get("decision_id")
            and candidate.next_action == "BLOCKED"
        ):
            raise ValueError("handoff is not pending; an accepted execution must not be repeated")
        _validate_source(run, handoff["source"])
        if exhausted and candidate.next_action != "BLOCKED":
            raise ValueError("recovery budget exhausted")
        if candidate.next_action != "BLOCKED":
            handoff["remaining_budget"] -= 1
        handoff.update(state="executing", decision_id=decision_id)
        receipt = {
            "schema_version": 1, "run_id": run_id, "handoff_id": handoff_id,
            "decision_id": decision_id, "source_version": expected_source_version,
            "payload_sha256": payload_sha256, "decision": candidate.to_dict(),
            "state": "executing", "execution_status": "IN_PROGRESS_OR_UNKNOWN",
            "accepted_at": time.time(), "remaining_budget": handoff["remaining_budget"],
        }
        decisions[decision_id] = receipt
        scheduler.record_graph_transition(run_id, "graph_handoffs", handoffs)
        scheduler.record_graph_transition(run_id, "graph_decisions", decisions)
        return deepcopy(receipt), True


def finish_graph_decision(scheduler, run_id: str, handoff_id: str, decision_id: str,
                          outcome: dict) -> dict:
    """Persist execution outcome, never manufacture a validator success."""
    if not isinstance(outcome, dict) or not isinstance(outcome.get("status"), str):
        raise ValueError("execution outcome requires a status string")
    canonical_digest(outcome)
    with scheduler.store.transaction():
        run = _run(scheduler, run_id)
        decisions = deepcopy(run.metadata.get("graph_decisions", {}))
        handoffs = deepcopy(run.metadata.get("graph_handoffs", {}))
        receipt = decisions.get(decision_id)
        handoff = handoffs.get(handoff_id)
        if (receipt is None or handoff is None or receipt["handoff_id"] != handoff_id
                or handoff.get("decision_id") != decision_id):
            raise ValueError("execution outcome does not match an accepted decision")
        if "outcome" in receipt:
            if canonical_digest(receipt["outcome"]) != canonical_digest(outcome):
                raise ValueError("execution outcome conflicts with its completed receipt")
            return deepcopy(receipt)
        if receipt["state"] != "executing" or handoff["state"] != "executing":
            raise ValueError("decision is not executing")
        state = "blocked" if (
            outcome["status"] == "BLOCKED" or receipt["decision"]["next_action"] == "BLOCKED"
        ) else "completed"
        receipt.update(state=state, execution_status="FINISHED", outcome=deepcopy(outcome),
                       finished_at=time.time())
        handoff.update(state=state, outcome=deepcopy(outcome))
        handoff["history"].append({"decision": deepcopy(receipt["decision"]),
                                   "outcome": deepcopy(outcome), "decision_id": decision_id})
        scheduler.record_graph_transition(run_id, "graph_handoffs", handoffs)
        scheduler.record_graph_transition(run_id, "graph_decisions", decisions)
        return deepcopy(receipt)
