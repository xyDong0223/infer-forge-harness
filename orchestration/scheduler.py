"""Persistent event-driven scheduler for operator adaptation tasks.

The scheduler is intentionally small: operator discovery creates a torch task;
completion of each stage creates the next stage without blocking the caller.
SQLite provides durable state and idempotency across process restarts.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .contracts import AdaptationRun, BugReport, DiagnosticTask, OperatorSpec, OperatorTask, TaskEvent


_STAGES = ("torch", "xpu", "integration")
_DIAGNOSIS_STAGE = "diagnosis"


class EventStore:
    """SQLite-backed store for runs, operators, tasks and append-only events."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
              created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operators (
              run_id TEXT NOT NULL, operator_key TEXT NOT NULL,
              payload TEXT NOT NULL, PRIMARY KEY(run_id, operator_key)
            );
            CREATE TABLE IF NOT EXISTS tasks (
              task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
              operator_key TEXT NOT NULL, stage TEXT NOT NULL,
              status TEXT NOT NULL, attempt INTEGER NOT NULL,
              input_json TEXT NOT NULL, output_json TEXT NOT NULL,
              lease_token TEXT, lease_worker TEXT, lease_expires REAL,
              created_at REAL NOT NULL, updated_at REAL NOT NULL,
              UNIQUE(run_id, operator_key, stage)
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_ready ON tasks(status, stage, lease_expires);
            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
              event_type TEXT NOT NULL, task_id TEXT, payload TEXT NOT NULL,
              timestamp REAL NOT NULL, schema_version INTEGER NOT NULL
            );
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def append_event(self, event: TaskEvent) -> None:
        self.db.execute(
            "INSERT INTO events(event_id,run_id,event_type,task_id,payload,timestamp,schema_version) VALUES(?,?,?,?,?,?,?)",
            (event.event_id, event.run_id, event.event_type, event.task_id,
             json.dumps(event.payload, sort_keys=True), event.timestamp, event.schema_version),
        )
        self.db.commit()

    def events(self, run_id: str | None = None) -> list[TaskEvent]:
        q, args = "SELECT * FROM events ORDER BY timestamp, rowid", ()
        if run_id is not None:
            q, args = "SELECT * FROM events WHERE run_id=? ORDER BY timestamp, rowid", (run_id,)
        rows = self.db.execute(q, args).fetchall()
        return [TaskEvent(event_id=r["event_id"], run_id=r["run_id"], event_type=r["event_type"],
                          task_id=r["task_id"], payload=json.loads(r["payload"]),
                          timestamp=r["timestamp"], schema_version=r["schema_version"]) for r in rows]

    def save_run(self, run: AdaptationRun) -> AdaptationRun:
        payload = json.dumps(run.to_dict(), sort_keys=True)
        now = time.time()
        self.db.execute(
            "INSERT OR IGNORE INTO runs(run_id,payload,created_at,updated_at) VALUES(?,?,?,?)",
            (run.run_id, payload, run.created_at or now, run.updated_at or now),
        )
        self.db.commit()
        row = self.db.execute("SELECT payload FROM runs WHERE run_id=?", (run.run_id,)).fetchone()
        return AdaptationRun.from_dict(json.loads(row["payload"]))

    def run(self, run_id: str) -> AdaptationRun | None:
        row = self.db.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return AdaptationRun.from_dict(json.loads(row["payload"])) if row else None

    def put_operator(self, run_id: str, spec: OperatorSpec) -> bool:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO operators(run_id,operator_key,payload) VALUES(?,?,?)",
            (run_id, spec.operator_key, json.dumps(spec.to_dict(), sort_keys=True)),
        )
        self.db.commit()
        return cur.rowcount > 0

    def operator(self, run_id: str, key: str) -> OperatorSpec | None:
        row = self.db.execute("SELECT payload FROM operators WHERE run_id=? AND operator_key=?", (run_id, key)).fetchone()
        return OperatorSpec.from_dict(json.loads(row["payload"])) if row else None

    def insert_task(self, task: OperatorTask) -> bool:
        now = time.time()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO tasks(task_id,run_id,operator_key,stage,status,attempt,input_json,output_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (task.task_id, task.run_id, task.operator_key, task.stage, task.status, task.attempt,
             json.dumps(task.input, sort_keys=True), json.dumps(task.output, sort_keys=True), now, now),
        )
        self.db.commit()
        return cur.rowcount > 0

    def _task(self, row: sqlite3.Row) -> OperatorTask | DiagnosticTask:
        if row["stage"] == _DIAGNOSIS_STAGE:
            payload = json.loads(row["input_json"])
            # Diagnosis rows use a source-task-specific storage key because
            # one operator can fail at multiple stages.  Expose the original
            # operator key to workers from the input contract.
            operator_key = payload.get("operator_key", row["operator_key"])
            return DiagnosticTask(task_id=row["task_id"], run_id=row["run_id"], operator_key=operator_key,
                                  source_task_id=payload.get("source_task_id", ""), status=row["status"],
                                  attempt=row["attempt"], input=payload, output=json.loads(row["output_json"]),
                                  lease_token=row["lease_token"])
        return OperatorTask(task_id=row["task_id"], run_id=row["run_id"], operator_key=row["operator_key"],
                            stage=row["stage"], status=row["status"], attempt=row["attempt"],
                            input=json.loads(row["input_json"]), output=json.loads(row["output_json"]),
                            lease_token=row["lease_token"])

    def get_task(self, task_id: str) -> OperatorTask | None:
        row = self.db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def tasks(self, run_id: str | None = None) -> list[OperatorTask]:
        q, args = "SELECT * FROM tasks ORDER BY created_at", ()
        if run_id is not None:
            q, args = "SELECT * FROM tasks WHERE run_id=? ORDER BY created_at", (run_id,)
        return [self._task(r) for r in self.db.execute(q, args).fetchall()]

    def pending_tasks(self, run_id: str | None = None, stage: str | None = None,
                      limit: int | None = None) -> list[OperatorTask | DiagnosticTask]:
        """Return durable pending work without claiming it."""
        clauses = ["status='pending'"]
        args: list[Any] = []
        if run_id is not None:
            clauses.append("run_id=?")
            args.append(run_id)
        if stage is not None:
            clauses.append("stage=?")
            args.append(stage)
        sql = "SELECT * FROM tasks WHERE " + " AND ".join(clauses) + " ORDER BY created_at"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(max(0, int(limit)))
        return [self._task(r) for r in self.db.execute(sql, tuple(args)).fetchall()]


class TaskScheduler:
    """Coordinates asynchronous operator stages with durable, idempotent state."""

    def __init__(self, store: EventStore | str | Path):
        self.store = store if isinstance(store, EventStore) else EventStore(store)

    def create_run(self, run: AdaptationRun | None = None, **kwargs: Any) -> AdaptationRun:
        if run is None:
            run = AdaptationRun(run_id=kwargs.pop("run_id", str(uuid.uuid4())), **kwargs)
        existing = self.store.run(run.run_id)
        if existing:
            return existing
        saved = self.store.save_run(run)
        self._emit(saved.run_id, "run_created", {"model_id": saved.model_id})
        return saved

    def discover_operator(self, run_id: str, spec: OperatorSpec) -> OperatorTask:
        if self.store.run(run_id) is None:
            raise KeyError(f"unknown run: {run_id}")
        key = spec.operator_key
        is_new = self.store.put_operator(run_id, spec)
        existing = next((t for t in self.store.tasks(run_id) if t.operator_key == key and t.stage == "torch"), None)
        if existing:
            return existing
        task = OperatorTask(task_id=f"{run_id}:{key}:torch", run_id=run_id, operator_key=key,
                            stage="torch", status="pending", input={"operator_spec": spec.to_dict()})
        self.store.insert_task(task)
        self._emit(run_id, "operator_discovered", {"operator_key": key, "new": is_new})
        self._emit(run_id, "task_created", {"stage": "torch", "operator_key": key}, task.task_id)
        return task

    def claim_ready(self, worker_id: str, stage: str | None = None, lease_seconds: float = 300,
                    limit: int = 1) -> list[OperatorTask]:
        if not worker_id or limit <= 0:
            return []
        now = time.time()
        # Claiming is a single IMMEDIATE transaction.  The conditional update
        # then remains safe even when several workers race on the same queue.
        claimed: list[OperatorTask] = []
        claim_events: list[tuple[str, str, dict[str, Any], str]] = []
        db = self.store.db
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE tasks SET status='pending', lease_token=NULL, lease_worker=NULL, lease_expires=NULL, updated_at=? WHERE status='running' AND lease_expires <= ?", (now, now))
            sql = "SELECT * FROM tasks WHERE status='pending'"
            args: tuple[Any, ...] = ()
            if stage:
                sql += " AND stage=?"
                args = (stage,)
            candidates = db.execute(sql + " ORDER BY created_at", args).fetchall()
            for row in candidates:
                if len(claimed) >= limit or not self._deps_succeeded(row["run_id"], row["operator_key"], row["stage"]):
                    continue
                token = secrets.token_urlsafe(18)
                cur = db.execute("UPDATE tasks SET status='running',attempt=attempt+1,lease_token=?,lease_worker=?,lease_expires=?,updated_at=? WHERE task_id=? AND status='pending'", (token, worker_id, now + lease_seconds, now, row["task_id"]))
                if cur.rowcount:
                    claimed.append(self.store._task(db.execute("SELECT * FROM tasks WHERE task_id=?", (row["task_id"],)).fetchone()))
                    claim_events.append((row["run_id"], "task_claimed", {"worker_id": worker_id, "stage": row["stage"]}, row["task_id"]))
            db.commit()
        except sqlite3.OperationalError:
            db.rollback()
            return []
        for event in claim_events:
            self._emit(*event)
        return [t for t in claimed if t is not None]

    def _deps_succeeded(self, run_id: str, key: str, stage: str) -> bool:
        # Diagnostics are deliberately independent of the operator pipeline:
        # a failed task must be explainable even when an upstream stage failed.
        if stage == _DIAGNOSIS_STAGE:
            return True
        if stage == "torch":
            return True
        prev = _STAGES[_STAGES.index(stage) - 1]
        row = self.store.db.execute("SELECT status FROM tasks WHERE run_id=? AND operator_key=? AND stage=?", (run_id, key, prev)).fetchone()
        return bool(row and row["status"] == "succeeded")

    def complete(self, task_id: str, worker_id: str | None = None, result: dict[str, Any] | None = None,
                 lease_token: str | None = None) -> OperatorTask | DiagnosticTask:
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        row = self.store.db.execute("SELECT lease_worker,lease_token,status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        # Accept the concise positional form complete(task_id, lease_token, result).
        if worker_id and row["lease_token"] == worker_id and not lease_token:
            lease_token, worker_id = worker_id, None
        if row["status"] != "running" or (worker_id and row["lease_worker"] != worker_id) or (lease_token and row["lease_token"] != lease_token):
            raise ValueError("task is not owned by worker or is not running")
        output = result or {}
        if not isinstance(output, dict):
            raise ValueError("result must be a mapping")
        if not self._result_passes_gate(task.stage, output):
            # A worker may report a rejected verdict through ``complete``.  It
            # is terminal failure for this stage and must never create the
            # downstream task; persist it through the same ownership checks as
            # an explicit ``fail`` call.
            return self.fail(
                task_id,
                worker_id=worker_id,
                lease_token=lease_token,
                error={"reason": "result_validation_failed", "result": output},
            )
        now = time.time()
        self.store.db.execute("UPDATE tasks SET status='succeeded',output_json=?,lease_token=NULL,lease_worker=NULL,lease_expires=NULL,updated_at=? WHERE task_id=?", (json.dumps(output, sort_keys=True), now, task_id))
        self.store.db.commit()
        self._emit(task.run_id, "task_succeeded", {"stage": task.stage, "output": output}, task_id)
        if task.stage == _DIAGNOSIS_STAGE:
            # Keep a purpose-specific event so the main agent can consume the
            # repair conclusion without understanding queue internals.
            conclusion = output.get("repair_conclusion", output.get("conclusion", output.get("recommendation")))
            self._emit(task.run_id, "diagnostic_completed",
                       {"source_task_id": getattr(task, "source_task_id", task.input.get("source_task_id")),
                        "bug_report": task.input.get("bug_report", {}),
                        "diagnosis": output.get("diagnosis", output.get("analysis")),
                        "repair_conclusion": conclusion,
                        "status": output.get("status", "succeeded")}, task_id)
            return self.store.get_task(task_id)
        if task.stage not in {"integration", _DIAGNOSIS_STAGE}:
            next_stage = _STAGES[_STAGES.index(task.stage) + 1]
            self._ensure_next(task, next_stage)
        return self.store.get_task(task_id)

    @staticmethod
    def _result_passes_gate(stage: str, result: dict[str, Any]) -> bool:
        """Reject explicit failure reports while keeping stage-specific payloads open.

        Agents may attach arbitrary evidence.  The scheduler only interprets
        conventional verdict fields, and therefore cannot accidentally promote
        a task whose report explicitly says it failed.
        """
        for key in ("ok", "success", "passed"):
            if key in result and result[key] is False:
                return False
        verdict = result.get("status", result.get("verdict"))
        if isinstance(verdict, str) and verdict.strip().lower() in {
            "fail", "failed", "failure", "error", "rejected", "blocked", "false",
        }:
            return False
        return True

    def _ensure_next(self, task: OperatorTask, stage: str) -> None:
        next_task = OperatorTask(task_id=f"{task.run_id}:{task.operator_key}:{stage}", run_id=task.run_id,
                                 operator_key=task.operator_key, stage=stage, status="pending",
                                 input={"upstream_task_id": task.task_id})
        if self.store.insert_task(next_task):
            self._emit(task.run_id, "task_created", {"stage": stage, "operator_key": task.operator_key}, next_task.task_id)

    def fail(self, task_id: str, worker_id: str | None = None, error: Any = None,
             lease_token: str | None = None) -> OperatorTask | DiagnosticTask:
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        row = self.store.db.execute("SELECT lease_worker,lease_token,status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if worker_id and row["lease_token"] == worker_id and not lease_token:
            lease_token, worker_id = worker_id, None
        if row["status"] != "running" or (worker_id and row["lease_worker"] != worker_id) or (lease_token and row["lease_token"] != lease_token):
            raise ValueError("task is not owned by worker or is not running")
        report = BugReport.from_error(task_id, error)
        # Attach the failed task's execution contract when the runner did not
        # provide context itself.  This gives diagnosis workers enough detail
        # to reproduce the issue without querying volatile worker state.
        report.context.setdefault("failed_stage", task.stage)
        report.context.setdefault("task_input", task.input)
        if task.output:
            report.context.setdefault("task_output", task.output)
        failure_output = {"error": report.to_dict()}
        self.store.db.execute("UPDATE tasks SET status='failed',output_json=?,lease_token=NULL,lease_worker=NULL,lease_expires=NULL,updated_at=? WHERE task_id=?", (json.dumps(failure_output, sort_keys=True), time.time(), task_id))
        self.store.db.commit()
        if task.stage != _DIAGNOSIS_STAGE:
            self._ensure_diagnosis(task, report)
        # Keep task_failed as the terminal event for the failed task; consumers
        # can observe diagnosis_task_created immediately before it.
        self._emit(task.run_id, "task_failed", {"stage": task.stage, "error": report.to_dict()}, task_id)
        return self.store.get_task(task_id)

    def _ensure_diagnosis(self, task: OperatorTask, report: BugReport) -> DiagnosticTask:
        """Create one diagnosis task per failed task, safely under retries."""
        task_id = f"{task.task_id}:diagnosis"
        diagnostic = DiagnosticTask(
            task_id=task_id,
            run_id=task.run_id,
            operator_key=task.operator_key,
            source_task_id=task.task_id,
            input={"source_task_id": task.task_id, "operator_key": task.operator_key,
                   "failed_stage": task.stage, "bug_report": report.to_dict()},
        )
        # The task table has one row per (run, operator, stage).  Scope the
        # persisted key by source task so torch and xpu failures each receive
        # their own diagnosis queue item.
        storage_key = f"diagnosis:{task.task_id}"
        if self.store.insert_task(OperatorTask(task_id=diagnostic.task_id, run_id=diagnostic.run_id,
                                               operator_key=storage_key, stage=_DIAGNOSIS_STAGE,
                                               input=diagnostic.input)):
            self._emit(task.run_id, "diagnosis_task_created",
                       {"source_task_id": task.task_id, "operator_key": task.operator_key,
                        "failed_stage": task.stage}, diagnostic.task_id)
        row = self.store.db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return self.store._task(row)

    def recover(self, force: bool = False) -> int:
        clause = "1=1" if force else "lease_expires <= ?"
        args: tuple[Any, ...] = () if force else (time.time(),)
        cur = self.store.db.execute(f"UPDATE tasks SET status='pending',lease_token=NULL,lease_worker=NULL,lease_expires=NULL,updated_at=? WHERE status='running' AND {clause}", (time.time(), *args))
        self.store.db.commit()
        return cur.rowcount

    def task_status(self, task_id: str) -> OperatorTask | DiagnosticTask | None:
        """Read one task's durable status for polling or recovery."""
        return self.store.get_task(task_id)

    def diagnosis_for(self, source_task_id: str) -> DiagnosticTask | None:
        """Return the durable diagnosis task associated with a failed task."""
        row = self.store.db.execute(
            "SELECT * FROM tasks WHERE task_id=? AND stage=?", (f"{source_task_id}:diagnosis", _DIAGNOSIS_STAGE)
        ).fetchone()
        task = self.store._task(row) if row else None
        return task if isinstance(task, DiagnosticTask) else None

    def pending_tasks(self, run_id: str | None = None, stage: str | None = None,
                      limit: int | None = None) -> list[OperatorTask | DiagnosticTask]:
        return self.store.pending_tasks(run_id=run_id, stage=stage, limit=limit)

    def _emit(self, run_id: str, event_type: str, payload: dict[str, Any], task_id: str | None = None) -> None:
        self.store.append_event(TaskEvent(event_id=str(uuid.uuid4()), run_id=run_id, event_type=event_type, task_id=task_id, payload=payload))


__all__ = ["EventStore", "TaskScheduler"]

# Backwards-compatible concise aliases used by callers integrating the scheduler.
TaskScheduler.discover = TaskScheduler.discover_operator
TaskScheduler.poll_ready = TaskScheduler.claim_ready
TaskScheduler.complete_task = TaskScheduler.complete
TaskScheduler.fail_task = TaskScheduler.fail
