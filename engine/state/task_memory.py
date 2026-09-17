"""Rebuildable Task Loop view over append-only Journal coordination events.

These observations summarize execution; they never certify evidence or replace
scheduler leases. The no-Journal API remains available for legacy callers.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from copy import deepcopy

from core.storage import ensure_external
from engine.state import journal as journal_module


PROJECTION_KIND = "TaskMemoryProjection"
_LIST_FIELDS = ("completed_loop_blocks", "execution_records", "claims")
_VALUE_FIELDS = ("status", "current_loop_block", "next_loop_block", "environment")


class _View(dict):
    """JSON-compatible view with an unpersisted optimistic-concurrency cursor."""

    revision: str | None = None
    source: tuple[str, str, str, str] | None = None
    legacy_sha256: str | None = None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _identity(journal: Path, run_id: str | None, task_id: str, subject: str) -> tuple:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Journal-backed Task Memory requires an explicit run_id")
    if any(not isinstance(value, str) or not value.strip() for value in (task_id, subject)):
        raise ValueError("Task Memory requires nonempty task_id and subject")
    return (str(Path(journal).resolve()), run_id, task_id, subject)


def _checked(memory: Any, task_id: str, subject: str) -> dict:
    if not isinstance(memory, dict):
        raise ValueError("task memory must be an object")
    if memory.get("task_id") != task_id or memory.get("subject") != subject:
        raise ValueError("task memory belongs to another task or subject")
    if memory.get("version") != 1:
        raise ValueError("unsupported task memory version")
    result = deepcopy(memory)
    for key, default in empty_memory(task_id, subject).items():
        result.setdefault(key, default)
    if any(not isinstance(result[key], list) for key in _LIST_FIELDS):
        raise ValueError("task memory history must be a list")
    if not isinstance(result["environment"], dict):
        raise ValueError("task memory environment must be an object")
    return result


def empty_memory(task_id: str, subject: str) -> dict[str, Any]:
    return {
        "version": 1,
        "task_id": task_id,
        "subject": subject,
        "status": "IN_PROGRESS",
        "current_loop_block": None,
        "completed_loop_blocks": [],
        "next_loop_block": None,
        "execution_records": [],
        "claims": [],
        "environment": {},
    }


def _legacy(path: Path, task_id: str, subject: str) -> tuple[dict, str | None]:
    if not path.exists():
        return empty_memory(task_id, subject), None
    # Decode bytes directly: universal-newline translation would lose the exact
    # legacy artifact (and its original hash) during a first-write migration.
    text = path.read_bytes().decode("utf-8")
    return _checked(json.loads(text), task_id, subject), text


def _events(journal: Path, identity: tuple, path: Path | None = None) -> list[dict]:
    """Select one run/workflow stream without treating environment changes as runs."""
    _, run_id, task_id, subject = identity
    selected = []
    for entry in journal_module.load(journal):
        if not isinstance(entry, dict):
            raise ValueError("Journal entries must be objects")
        if entry.get("kind") != PROJECTION_KIND:
            continue
        detail = entry.get("detail")
        if not isinstance(detail, dict):
            raise ValueError("malformed Task Memory projection event")
        if detail.get("run_id") != run_id:
            if path is not None and Path(entry.get("artifacts", "")).resolve() == path.resolve():
                raise ValueError("Task Memory view belongs to another run")
            continue
        if detail.get("task_id") != task_id or entry.get("subject") != subject:
            raise ValueError("Task Memory Journal run belongs to another task or subject")
        selected.append(detail)
    return selected


def _project(events: list[dict], task_id: str, subject: str) -> tuple[dict, str | None]:
    memory = empty_memory(task_id, subject)
    previous = None
    seen = {}
    for event in events:
        event_id = event.get("event_id")
        body = {key: value for key, value in event.items() if key != "event_id"}
        if not isinstance(event_id, str) or _digest(body) != event_id:
            raise ValueError("Task Memory projection event digest mismatch")
        if event_id in seen:
            continue  # Byte-identical retries cannot duplicate observations.
        if event.get("schema_version") != 1 or event.get("previous_event_id") != previous:
            raise ValueError("Task Memory projection event chain is invalid")
        operations = event.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError("Task Memory projection event requires operations")
        initialized = previous is not None
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("malformed Task Memory projection operation")
            kind = operation.get("type")
            if kind in {"initialized", "legacy_import"}:
                if initialized:
                    raise ValueError("Task Memory initialization must be first")
                initialized = True
                if kind == "legacy_import":
                    source = operation["source"]
                    if not isinstance(source, dict) or not isinstance(source.get("raw_text"), str):
                        raise ValueError("malformed Task Memory legacy seed")
                    raw = source["raw_text"]
                    if (source.get("revalidated") is not False or
                            hashlib.sha256(raw.encode()).hexdigest() != source["sha256"]):
                        raise ValueError("Task Memory legacy seed provenance mismatch")
                    memory = _checked(json.loads(raw), task_id, subject)
            elif not initialized:
                raise ValueError("Task Memory stream must begin with initialization")
            elif kind == "block_completed":
                record = deepcopy(operation["record"])
                memory["completed_loop_blocks"].append(record)
                memory["execution_records"].append({
                    "block_id": record["block_id"], "state": record["state"],
                    "artifacts": record["artifacts"],
                })
            elif kind == "claim_recorded":
                memory["claims"].append(deepcopy(operation["claim"]))
            elif kind == "fields_updated":
                fields = operation["fields"]
                if not isinstance(fields, dict) or set(fields) - set(_VALUE_FIELDS):
                    raise ValueError("invalid Task Memory projection fields")
                memory.update(deepcopy(fields))
            else:
                raise ValueError(f"unknown Task Memory projection operation: {kind}")
        previous = event_id
        seen[event_id] = True
    return memory, previous


def load(path: Path, task_id: str, subject: str, *, journal: Path | None = None,
         run_id: str | None = None) -> dict[str, Any]:
    """Read the authoritative projection or the unmigrated legacy file; never write."""
    path = Path(path)
    if journal is None:
        return _legacy(path, task_id, subject)[0]
    identity = _identity(journal, run_id, task_id, subject)
    events = _events(Path(journal), identity, path)
    if events:
        memory, revision = _project(events, task_id, subject)
        raw = None
    else:
        memory, raw = _legacy(path, task_id, subject)
        revision = None
    view = _View(memory)
    view.source, view.revision = identity, revision
    view.legacy_sha256 = hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None
    return view


def _write_view(path: Path, memory: dict[str, Any]) -> None:
    """Atomically replace the expendable JSON view after its events are durable."""
    path = ensure_external(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(memory, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _delta(before: dict, after: dict) -> list[dict]:
    operations = []
    for key in ("completed_loop_blocks", "claims"):
        if after[key][:len(before[key])] != before[key]:
            raise ValueError("Task Memory history is append-only")
    records = after["completed_loop_blocks"][len(before["completed_loop_blocks"]):]
    expected = before["execution_records"] + [{
        "block_id": record["block_id"], "state": record["state"], "artifacts": record["artifacts"],
    } for record in records]
    if after["execution_records"] != expected:
        raise ValueError("Task Memory execution_records must derive from completed blocks")
    operations.extend({"type": "block_completed", "record": record} for record in records)
    operations.extend({"type": "claim_recorded", "claim": claim}
                      for claim in after["claims"][len(before["claims"]):])
    fields = {key: after[key] for key in _VALUE_FIELDS if after[key] != before[key]}
    if fields:
        operations.append({"type": "fields_updated", "fields": fields})
    other = set(before) | set(after)
    other -= set(_LIST_FIELDS) | set(_VALUE_FIELDS)
    if any(before.get(key) != after.get(key) for key in other):
        raise ValueError("Task Memory identity and legacy extension fields cannot be changed")
    return operations


def _append_event(journal: Path, path: Path, identity: tuple, previous: str | None,
                  operations: list[dict], environment: dict) -> str:
    _, run_id, task_id, subject = identity
    detail = {"schema_version": 1, "run_id": run_id, "task_id": task_id,
              "previous_event_id": previous, "operations": operations}
    detail["event_id"] = _digest(detail)
    journal_module.record(journal, PROJECTION_KIND, subject, "RECORDED", path,
                          environment, extra=detail, _locked=True)
    return detail["event_id"]


def save(path: Path, memory: dict[str, Any], *, journal: Path | None = None,
         run_id: str | None = None) -> None:
    """Commit a typed delta to the Journal, then atomically refresh its JSON view."""
    if journal is None:
        _write_view(path, memory)
        return
    path, journal = ensure_external(path), ensure_external(journal)
    if path == journal:
        raise ValueError("Task Memory view and Journal must have distinct paths")
    task_id, subject = memory.get("task_id"), memory.get("subject")
    desired = _checked(memory, task_id, subject)
    identity = _identity(journal, run_id, task_id, subject)
    if isinstance(memory, _View) and memory.source != identity:
        raise ValueError("Task Memory view belongs to another Journal or run")
    with journal_module.locked(journal):
        events = _events(journal, identity, path)
        before, revision = _project(events, task_id, subject)
        raw = None
        if not events:
            before, raw = _legacy(path, task_id, subject)
        if isinstance(memory, _View):
            legacy_hash = hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None
            if ((memory.revision != revision or not events and memory.legacy_sha256 != legacy_hash)
                    and desired != before):
                raise ValueError("Task Memory projection changed; reload before writing")
        operations = _delta(before, desired)
        if not events:
            seed = {"type": "initialized"} if raw is None else {
                "type": "legacy_import", "source": {
                    "path": str(path), "raw_text": raw,
                    "sha256": hashlib.sha256(raw.encode()).hexdigest(), "revalidated": False,
                },
            }
            operations = [seed, *operations]
        if operations:
            revision = _append_event(journal, path, identity, revision, operations,
                                     desired["environment"])
        _write_view(path, desired)
        if isinstance(memory, _View):
            memory.revision, memory.legacy_sha256 = revision, None


def rebuild(path: Path, task_id: str, subject: str, *, journal: Path,
            run_id: str) -> dict[str, Any]:
    """Explicitly materialize existing authority; never import or certify legacy data."""
    path, journal = ensure_external(path), ensure_external(journal)
    if path == journal:
        raise ValueError("Task Memory view and Journal must have distinct paths")
    identity = _identity(journal, run_id, task_id, subject)
    # Reject a missing/wrong source before creating even a lock sidecar.
    if not _events(journal, identity, path):
        raise ValueError("no authoritative Task Memory events for this run")
    with journal_module.locked(journal):
        memory, _ = _project(_events(journal, identity, path), task_id, subject)
        _write_view(path, memory)
    return memory


def start_block(
    memory: dict[str, Any],
    block_id: str,
    sub_target: str,
    exit_condition: dict[str, Any],
    routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block = {
        "block_id": block_id,
        "sub_target": sub_target,
        "exit_condition": exit_condition,
        "routing": routing or {},
        "started_at": time.time(),
    }
    memory["current_loop_block"] = block
    memory["next_loop_block"] = None
    return block


def finish_block(
    memory: dict[str, Any],
    state: str,
    artifacts: list[str] | None = None,
    next_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = memory.get("current_loop_block")
    if not current:
        raise ValueError("cannot finish a Task Loop without a current block")
    record = {
        **current,
        "state": state,
        "artifacts": artifacts or [],
        "finished_at": time.time(),
    }
    memory["completed_loop_blocks"].append(record)
    memory["execution_records"].append(
        {
            "block_id": record["block_id"],
            "state": state,
            "artifacts": record["artifacts"],
        }
    )
    memory["current_loop_block"] = None
    memory["next_loop_block"] = next_block
    if next_block is None and state in {"DELIVERED", "TASK_COMPLETE"}:
        memory["status"] = "COMPLETED"
    return record


def set_next_block(memory: dict[str, Any], next_block: dict[str, Any]) -> None:
    memory["next_loop_block"] = next_block


def record_claim(
    memory: dict[str, Any],
    claim: str,
    status: str,
    evidence: list[str],
    environment: dict[str, str],
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Record measured claims and rejected hypotheses without erasing history."""
    entry = {
        "claim": claim,
        "status": status,
        "evidence": evidence,
        "environment": dict(environment),
    }
    if supersedes:
        entry["supersedes"] = supersedes
    memory["claims"].append(entry)
    memory["environment"] = dict(environment)
    return entry


def record_observed_issue(
    memory: dict[str, Any],
    issue: str,
    evidence: list[str],
    environment: dict[str, str],
    source: str = "runner",
) -> dict[str, Any]:
    """Persist a machine-observed issue so routing need not depend on free text."""
    entry = record_claim(memory, issue, "OBSERVED", evidence, environment)
    entry["observed_issue"] = issue
    entry["source"] = source
    return entry


def execute(args) -> int:
    journal, run_id = getattr(args, "journal", None), getattr(args, "run_id", None)
    if getattr(args, "rebuild", False):
        memory = rebuild(args.path, args.task_id, args.subject, journal=journal, run_id=run_id)
        print(json.dumps(memory, indent=2, ensure_ascii=False))
        return 0
    memory = load(args.path, args.task_id, args.subject, journal=journal, run_id=run_id)
    if args.show:
        print(json.dumps(memory, indent=2, ensure_ascii=False))
        return 0
    save(args.path, memory, journal=journal, run_id=run_id)
    print(json.dumps(memory, indent=2, ensure_ascii=False))
    return 0
