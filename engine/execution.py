"""Durable process occupancy beneath tasks, without treating lease expiry as exit.

The managed runner owns process observation. This module records its conclusions
and serializes cooperating callers in one scheduler database; it is not an OS or
cross-database lock. STARTING and UNKNOWN deliberately continue to occupy their
task and Pod until a trusted runner observes termination.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import time
from typing import Any, Callable

from core.storage import ensure_external


TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED"})
ACTIVE_STATES = frozenset({"STARTING", "RUNNING", "UNKNOWN"})


class ExecutionConflict(ValueError):
    """Another unresolved execution owns this identity, task, or resource."""


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def canonical_resource(resource: dict | None) -> dict | None:
    if resource is None:
        return None
    fields = {"cluster", "namespace", "pod_uid"}
    if not isinstance(resource, dict) or set(resource) != fields:
        raise ValueError("resource must contain exactly cluster, namespace, and pod_uid")
    return {field: _nonempty(resource[field], f"resource.{field}") for field in sorted(fields)}


def resource_key(resource: dict | None) -> str | None:
    normalized = canonical_resource(resource)
    return "pod:" + _digest(normalized) if normalized is not None else None


def is_active(record: dict) -> bool:
    # Unknown/corrupt states fail closed rather than freeing an occupied Pod.
    return not (record.get("state") in TERMINAL_STATES
                and record.get("termination_confirmed") is True)


def _records(run) -> dict:
    records = run.metadata.get("execution_records", {})
    if not isinstance(records, dict) or any(not isinstance(item, dict) for item in records.values()):
        raise ValueError("run execution_records must be an object of records")
    return records


def get_execution(store, run_id: str, execution_id: str) -> dict | None:
    """Read a record without recovering leases or creating state."""
    run = store.run(run_id)
    if run is None:
        raise KeyError(f"unknown run: {run_id}")
    record = _records(run).get(execution_id)
    return _json_copy(record) if record is not None else None


def list_executions(store, run_id: str | None = None, *, active_only: bool = False) -> list[dict]:
    """Read occupancy across runs sharing this database."""
    run_ids = ([run_id] if run_id is not None else
               [row["run_id"] for row in store.db.execute("SELECT run_id FROM runs ORDER BY run_id")])
    result = []
    for identity in run_ids:
        run = store.run(identity)
        if run is None:
            raise KeyError(f"unknown run: {identity}")
        result.extend(_json_copy(record) for record in _records(run).values()
                      if not active_only or is_active(record))
    return result


def has_active_task_execution(store, task_id: str) -> bool:
    """Old unresolved attempts block a new attempt too: expiry proves no exit."""
    return any(item.get("task_id") == task_id
               for item in list_executions(store, active_only=True))


def _save(scheduler, record: dict, event_type: str) -> dict:
    run = scheduler.store.run(record["run_id"])
    if run is None:
        raise KeyError(f"unknown run: {record['run_id']}")
    records = dict(_records(run))
    records[record["execution_id"]] = _json_copy(record)
    run.metadata = {**run.metadata, "execution_records": records}
    scheduler.store.update_run(run)
    scheduler._emit(run.run_id, event_type, {
        "execution_id": record["execution_id"], "state": record["state"],
        "revision": record["revision"], "task_attempt": record.get("task_attempt"),
        "resource_key": record.get("resource_key"),
        "termination_confirmed": record["termination_confirmed"],
    }, record.get("task_id"))
    return _json_copy(record)


def _required(store, run_id: str, execution_id: str) -> dict:
    record = get_execution(store, run_id, execution_id)
    if record is None:
        raise KeyError(f"unknown execution: {execution_id}")
    return record


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _owner(record: dict, worker: str | None, lease_token: str | None) -> None:
    if record.get("task_id") is None:
        if worker is not None or lease_token is not None:
            raise ValueError("graph executions do not accept task lease credentials")
        return
    if (not isinstance(lease_token, str) or not lease_token
            or worker != record.get("worker")
            or not hmac.compare_digest(_token_hash(lease_token), record["lease_token_sha256"])):
        raise ValueError("execution requires its original worker and lease token")


def _live_owner(scheduler, record: dict, worker, lease_token) -> None:
    _owner(record, worker, lease_token)
    if record.get("task_id") is not None:
        row = scheduler._owned_lease(record["task_id"], worker, lease_token)
        if row["attempt"] != record["task_attempt"]:
            raise ValueError("execution belongs to a different task attempt")


def _no_token(value: Any, token: str | None) -> None:
    if token and token in json.dumps(value, sort_keys=True, allow_nan=False):
        raise ValueError("execution metadata must not contain the lease token")


def _payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("execution payload must be an object")
    result = _json_copy(payload)
    argv = result.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
        raise ValueError("payload.argv must be a nonempty string argv list")
    if not argv[0].strip():
        raise ValueError("payload.argv executable must be nonempty")
    result["cwd"] = str(Path(_nonempty(result.get("cwd"), "payload.cwd")).expanduser().resolve())
    result["output_dir"] = str(ensure_external(_nonempty(result.get("output_dir"), "payload.output_dir")))
    return result


def begin_execution(scheduler, run_id: str, execution_id: str, payload: dict, *,
                    task_id: str | None = None, worker: str | None = None,
                    lease_token: str | None = None, resource: dict | None = None) -> tuple[dict, bool]:
    """Atomically reserve a process execution before spawning anything.

    Repeating an identical request returns the original record, including after
    completion or owner-lease expiry. Callers must not spawn when is_new is false.
    """
    execution_id = _nonempty(execution_id, "execution_id")
    normalized_payload = _payload(payload)
    _no_token(normalized_payload, lease_token)
    normalized_resource = canonical_resource(resource)
    immutable = {"payload": normalized_payload, "task_id": task_id,
                 "resource": normalized_resource, "worker": worker}
    _no_token(immutable, lease_token)
    request_hash = _digest(immutable)
    with scheduler.store.transaction():
        run = scheduler.store.run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        existing = _records(run).get(execution_id)
        if existing is not None:
            _owner(existing, worker, lease_token)
            if existing["request_sha256"] != request_hash:
                raise ExecutionConflict("execution_id already belongs to a different request")
            return _json_copy(existing), False
        task_attempt = None
        if task_id is not None:
            if not worker or not lease_token:
                raise ValueError("task execution requires worker and lease_token")
            row = scheduler._owned_lease(task_id, worker, lease_token)
            if row["run_id"] != run_id:
                raise ValueError("execution task does not belong to run")
            task_attempt = row["attempt"]
        elif worker is not None or lease_token is not None:
            raise ValueError("worker and lease_token require task_id")
        key = resource_key(normalized_resource)
        for active in list_executions(scheduler.store, active_only=True):
            if ((key is not None and active.get("resource_key") == key)
                    or (task_id is not None and active.get("task_id") == task_id)):
                raise ExecutionConflict(
                    f"unresolved execution {active['run_id']}/{active['execution_id']} owns the task or resource"
                )
        now = time.time()
        record = {
            "schema_version": 1, "execution_id": execution_id, "run_id": run_id,
            **immutable, "request_sha256": request_hash,
            "task_attempt": task_attempt,
            "lease_token_sha256": _token_hash(lease_token) if lease_token else None,
            "resource_key": key, "state": "STARTING", "revision": 1,
            "created_at": now, "updated_at": now, "heartbeat_at": None,
            "pid": None, "process_identity": None, "host_id": None, "remote_handle": None,
            "returncode": None, "log_path": None, "error": None,
            "termination_confirmed": False,
        }
        return _save(scheduler, record, "execution_reserved"), True


def mark_started(scheduler, run_id: str, execution_id: str, *, pid: int | None,
                 process_identity: Any, host_id: str, remote_handle: Any = None,
                 worker: str | None = None, lease_token: str | None = None) -> dict:
    if pid is not None and (type(pid) is not int or pid <= 0):
        raise ValueError("pid must be a positive integer or None")
    host_id = _nonempty(host_id, "host_id")
    if not process_identity or (pid is None and remote_handle is None):
        raise ValueError("started execution requires process identity and a local pid or remote handle")
    details = _json_copy({"pid": pid, "process_identity": process_identity,
                          "host_id": host_id, "remote_handle": remote_handle})
    _no_token(details, lease_token)
    with scheduler.store.transaction():
        record = _required(scheduler.store, run_id, execution_id)
        _live_owner(scheduler, record, worker, lease_token)
        if record["state"] == "RUNNING" and all(record[key] == value for key, value in details.items()):
            return record
        if record["state"] != "STARTING":
            raise ExecutionConflict("only a STARTING execution may bind a process identity")
        record.update(details, state="RUNNING", updated_at=time.time(), revision=record["revision"] + 1)
        return _save(scheduler, record, "execution_started")


def heartbeat(scheduler, run_id: str, execution_id: str, *, worker: str | None = None,
              lease_token: str | None = None) -> dict:
    """Record liveness only while the original task lease is still valid."""
    with scheduler.store.transaction():
        record = _required(scheduler.store, run_id, execution_id)
        _live_owner(scheduler, record, worker, lease_token)
        if not is_active(record):
            raise ExecutionConflict("terminal execution cannot heartbeat")
        now = time.time()
        record.update(heartbeat_at=now, updated_at=now, revision=record["revision"] + 1)
        return _save(scheduler, record, "execution_heartbeat")


def _finish(scheduler, record: dict, *, state: str, returncode: int | None,
            log_path: str | None, error: Any, termination_confirmed: bool,
            event_type: str) -> dict:
    if state not in TERMINAL_STATES | {"UNKNOWN"}:
        raise ValueError("finish state must be SUCCEEDED, FAILED, or UNKNOWN")
    if type(termination_confirmed) is not bool:
        raise ValueError("termination_confirmed must be boolean")
    if state in TERMINAL_STATES and not termination_confirmed:
        raise ValueError("terminal state requires confirmed process/remote termination")
    if state == "UNKNOWN" and termination_confirmed:
        raise ValueError("UNKNOWN cannot assert confirmed termination")
    if returncode is not None and type(returncode) is not int:
        raise ValueError("returncode must be an integer or None")
    if state == "SUCCEEDED" and returncode != 0:
        raise ValueError("SUCCEEDED requires returncode 0")
    completion = {
        "state": state, "returncode": returncode,
        "log_path": str(ensure_external(log_path)) if log_path is not None else None,
        "error": _json_copy(error), "termination_confirmed": termination_confirmed,
    }
    if not is_active(record):
        if all(record.get(key) == value for key, value in completion.items()):
            return record
        raise ExecutionConflict("terminal execution result cannot be replaced")
    record.update(completion, updated_at=time.time(), revision=record["revision"] + 1)
    if termination_confirmed:
        record["finished_at"] = record["updated_at"]
    return _save(scheduler, record, event_type)


def finish_execution(scheduler, run_id: str, execution_id: str, *, state: str,
                     returncode: int | None = None, log_path: str | None = None,
                     error: Any = None, termination_confirmed: bool = False,
                     worker: str | None = None, lease_token: str | None = None) -> dict:
    """The original owner may report confirmed exit even after its lease expires."""
    _no_token({"log_path": log_path, "error": error}, lease_token)
    with scheduler.store.transaction():
        record = _required(scheduler.store, run_id, execution_id)
        _owner(record, worker, lease_token)
        return _finish(scheduler, record, state=state, returncode=returncode,
                       log_path=log_path, error=error, termination_confirmed=termination_confirmed,
                       event_type="execution_finished" if termination_confirmed else "execution_unknown")


def reconcile_execution(scheduler, run_id: str, execution_id: str, *,
                        observer: Callable[[dict], dict]) -> dict:
    """Accept a trusted runner's process observation, never an operator force flag.

    Observe outside the write transaction. Bind the result to the exact record
    revision so a concurrent process binding or finish cannot be overwritten.
    """
    if not callable(observer):
        raise ValueError("reconciliation requires a trusted process observer")
    snapshot = _required(scheduler.store, run_id, execution_id)
    if not is_active(snapshot):
        return snapshot
    observed = observer(_json_copy(snapshot))
    if not isinstance(observed, dict) or observed.get("execution_id") != execution_id:
        raise ValueError("observation must identify the observed execution")
    confirmed = (observed.get("terminal") is True
                 and observed.get("termination_confirmed") is True)
    with scheduler.store.transaction():
        record = _required(scheduler.store, run_id, execution_id)
        if record["revision"] != snapshot["revision"]:
            raise ExecutionConflict("execution changed during process observation; observe again")
        return _finish(scheduler, record,
                       state=observed.get("state") if confirmed else "UNKNOWN",
                       returncode=observed.get("returncode"), log_path=observed.get("log_path"),
                       error=observed.get("error"), termination_confirmed=confirmed,
                       event_type="execution_reconciled" if confirmed else "execution_unknown")
