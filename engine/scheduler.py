"""Persistent event-driven scheduler for operator adaptation tasks.

The scheduler is intentionally small: operator discovery creates a torch task;
completion of each stage creates the next stage without blocking the caller.
SQLite provides durable state and idempotency across process restarts.
"""
from __future__ import annotations

import json
import hashlib
import math
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

from .contracts import AdaptationRun, BugReport, DiagnosticTask, OperatorSpec, OperatorTask, TaskEvent
from .result_validation import validate_result
from core.storage import ArtifactStore, RunPaths, WritePolicyError, default_state_root, ensure_external, safe_component


_STAGES = ("torch", "xpu", "integration")
_DIAGNOSIS_STAGE = "diagnosis"


def atomic_transition(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.store.transaction():
            return method(self, *args, **kwargs)
    return wrapped


class EventStore:
    """SQLite-backed store for runs, operators, tasks and append-only events."""

    def __init__(self, path: str | Path, *, readonly: bool = False):
        self.path = ":memory:" if str(path) == ":memory:" else str(ensure_external(path))
        if readonly and (self.path == ":memory:" or not Path(self.path).is_file()):
            raise ValueError("read-only scheduler requires an existing state database")
        if self.path != ":memory:" and not readonly:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            Path(self.path).as_uri() + "?mode=ro" if readonly else self.path,
            uri=readonly, check_same_thread=False,
        )
        self.db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._transaction_depth = 0
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self._init_schema()

    @contextmanager
    def transaction(self):
        """Serialize shared connections and commit a complete transition once."""
        with self._lock:
            outermost = self._transaction_depth == 0
            if outermost:
                self.db.execute("BEGIN IMMEDIATE")
            self._transaction_depth += 1
            try:
                yield
                if outermost:
                    self.db.commit()
            except BaseException:
                if outermost:
                    self.db.rollback()
                raise
            finally:
                self._transaction_depth -= 1

    def commit(self) -> None:
        if not self._transaction_depth:
            self.db.commit()

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
        self.commit()

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
        self.commit()
        row = self.db.execute("SELECT payload FROM runs WHERE run_id=?", (run.run_id,)).fetchone()
        return AdaptationRun.from_dict(json.loads(row["payload"]))

    def update_run(self, run: AdaptationRun) -> AdaptationRun:
        """Persist a changed run context without replacing its identity."""
        now = time.time()
        run.updated_at = now
        self.db.execute(
            "UPDATE runs SET payload=?, updated_at=? WHERE run_id=?",
            (json.dumps(run.to_dict(), sort_keys=True), now, run.run_id),
        )
        self.commit()
        return run

    def run(self, run_id: str) -> AdaptationRun | None:
        row = self.db.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return AdaptationRun.from_dict(json.loads(row["payload"])) if row else None

    def put_operator(self, run_id: str, spec: OperatorSpec) -> bool:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO operators(run_id,operator_key,payload) VALUES(?,?,?)",
            (run_id, spec.operator_key, json.dumps(spec.to_dict(), sort_keys=True)),
        )
        self.commit()
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
        self.commit()
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
                                  lease_token=row["lease_token"], lease_expires=row["lease_expires"])
        return OperatorTask(task_id=row["task_id"], run_id=row["run_id"], operator_key=row["operator_key"],
                            stage=row["stage"], status=row["status"], attempt=row["attempt"],
                            input=json.loads(row["input_json"]), output=json.loads(row["output_json"]),
                            lease_token=row["lease_token"], lease_expires=row["lease_expires"])

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

    @atomic_transition
    def create_run(self, run: AdaptationRun | None = None, **kwargs: Any) -> AdaptationRun:
        if run is None:
            run = AdaptationRun(run_id=kwargs.pop("run_id", str(uuid.uuid4())), **kwargs)
        existing = self.store.run(run.run_id)
        if existing:
            return existing
        mode = run.metadata.get("evidence_mode", "real")
        if mode not in ("real", "simulation"):
            raise ValueError("evidence_mode must be real or simulation")
        if mode == "real":
            run.metadata["environment_required"] = True
            run.status = "WAITING_FOR_ENVIRONMENT"
            root = run.metadata.get("artifact_root") or (
                (Path(self.store.path).parent if self.store.path != ":memory:" else default_state_root())
                / "runs" / safe_component(run.run_id)
            )
            run.metadata["artifact_root"] = str(ensure_external(root))
        elif run.metadata.get("artifact_root"):
            run.metadata["artifact_root"] = str(ensure_external(run.metadata["artifact_root"]))
        saved = self.store.save_run(run)
        self._emit(saved.run_id, "run_created", {"model_id": saved.model_id})
        return saved

    @atomic_transition
    def bind_environment(
        self, run_id: str, proof: dict[str, Any], artifact_root: str | Path | None = None,
    ) -> AdaptationRun:
        """Bind a validated deployment environment to a run.

        The caller must supply the output of the environment-proof task.  This
        method deliberately checks the minimum handoff fields again so a stale
        or hand-written status cannot unlock discovery.
        """
        run = self.store.run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        mode = proof.get("evidence_mode", "real")
        if mode not in {"real", "simulation"}:
            raise ValueError("environment proof evidence_mode must be real or simulation")
        if run.metadata.get("evidence_mode", "real") == "real" and mode != "real":
            raise ValueError("simulation environment proof cannot satisfy a real run")
        checks = proof.get("checks") or {}
        if not isinstance(checks, dict):
            raise ValueError("environment proof checks must be an object")
        required = {
            "state": proof.get("state"),
            "pod": proof.get("pod"),
            "pod_ready": checks.get("pod_ready"),
            "runtime_importable": checks.get("runtime_importable"),
            "code_ready": checks.get("code_ready"),
            "device_ready": checks.get("device_ready"),
            "base_model_loaded": checks.get("base_model_loaded"),
            "base_prefill": checks.get("base_prefill"),
            "base_decode": checks.get("base_decode"),
        }
        missing = [name for name, value in required.items()
                   if (name == "state" and value != "ENVIRONMENT_READY")
                   or (name != "state" and value is not True and name != "pod")
                   or (name == "pod" and not value)]
        if missing:
            raise ValueError("environment proof is not ready: " + ", ".join(missing))
        for key, expected in (
            ("base_health_check", 200), ("base_chat_completion", "non_empty"),
            ("unexpected_fallback", False),
        ):
            if key not in checks or checks[key] != expected or (
                isinstance(expected, bool) and checks[key] is not expected
            ):
                raise ValueError(f"environment proof checks.{key} must equal {expected!r}")
        validator = proof.get("validator")
        if validator is not None and (
            not isinstance(validator, dict) or validator.get("passed") is not True
            or validator.get("errors")
        ):
            raise ValueError("environment proof validator rejected the evidence")
        artifacts = proof.get("artifacts") or []
        if not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
            raise ValueError("environment proof artifacts must be a list of file names")
        required_artifacts = {"environment_fingerprint.txt", "runtime_import.txt",
                              "code_readiness.json", "device_readiness.json",
                              "base_model_identity.json", "base_health_result.txt",
                              "base_chat_result.json", "base_server_log.txt"}
        absent = sorted(required_artifacts - set(artifacts))
        if absent:
            raise ValueError(f"environment proof is missing evidence: {', '.join(absent)}")
        root_value = artifact_root or proof.get("artifact_root")
        if not root_value:
            raise ValueError("environment proof requires its artifact_root")
        root = Path(root_value).resolve()
        evidence_hashes = {}
        for name in sorted(required_artifacts):
            path = root / name
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"cannot read environment evidence {path}: {exc}") from exc
            if not content:
                raise ValueError(f"environment evidence is empty: {path}")
            evidence_hashes[name] = hashlib.sha256(content).hexdigest()
        status_path = root / "status.json"
        if status_path.is_file():
            evidence_hashes["status.json"] = hashlib.sha256(status_path.read_bytes()).hexdigest()
        fingerprint = evidence_hashes["environment_fingerprint.txt"]
        previous = run.environment.get("environment_proof", {}).get("fingerprint")
        if previous and previous != fingerprint and self.store.tasks(run_id):
            raise ValueError("environment changed after discovery; existing tasks cannot inherit a new proof")
        run.environment = {
            **run.environment,
            "environment_proof": {
                "pod": proof["pod"],
                "checks": dict(checks),
                "artifacts": list(artifacts),
                "fingerprint": fingerprint,
                "artifact_root": str(root),
                "evidence_sha256": evidence_hashes,
                "evidence_mode": mode,
            },
        }
        run.status = "ENVIRONMENT_READY"
        self.store.update_run(run)
        self._emit(run_id, "environment_bound", {"pod": proof["pod"], "artifacts": list(artifacts)})
        return run

    @atomic_transition
    def record_environment_failure(self, run_id: str, proof: dict[str, Any], error: str) -> AdaptationRun:
        run = self.store.run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        run.status = "ENVIRONMENT_FAILED"
        run.environment = {**run.environment, "failed_environment_proof": {
            "state": proof.get("state", "FAILED"), "pod": proof.get("pod"),
            "checks": proof.get("checks", {}), "artifacts": proof.get("artifacts", []),
            "diagnosis": proof.get("reason", error),
        }}
        self.store.update_run(run)
        self._emit(run_id, "environment_failed", {"error": error, "artifacts": proof.get("artifacts", [])})
        return run

    @atomic_transition
    def record_graph_transition(
        self, run_id: str, key: str, payload: dict[str, Any],
    ) -> AdaptationRun:
        """Persist graph handoffs without changing the environment status meaning."""
        if key not in {"graph_environment_required", "graph_environment",
                       "graph_discovery", "graph_shim_discovery", "graph_delivery"}:
            raise ValueError(f"unsupported graph transition: {key}")
        run = self.store.run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        value = True if key == "graph_environment_required" else dict(payload)
        if run.metadata.get(key) != value:
            run.metadata[key] = value
            self.store.update_run(run)
            self._emit(run_id, key, dict(payload))
        return run

    @atomic_transition
    def discover_operator(self, run_id: str, spec: OperatorSpec) -> OperatorTask:
        run = self.store.run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        environment_required = (
            run.metadata.get("evidence_mode", "real") != "simulation"
            or run.metadata.get("environment_required")
            or run.metadata.get("graph_environment_required")
        )
        if environment_required and run.status != "ENVIRONMENT_READY":
            raise ValueError(
                "environment proof is required before operator discovery; bind a validated "
                "deployment environment first"
            )
        if environment_required:
            fingerprint = run.environment.get("environment_proof", {}).get("fingerprint")
            if not fingerprint or spec.environment.get("fingerprint") != fingerprint:
                raise ValueError("operator specification must reference the bound environment fingerprint")
            if spec.model_id != run.model_id or spec.backend != run.backend:
                raise ValueError("operator specification model/backend does not match the run")
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

    @atomic_transition
    def claim_ready(self, worker_id: str, stage: str | None = None, lease_seconds: float = 300,
                    limit: int = 1) -> list[OperatorTask]:
        if not worker_id or limit <= 0 or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("worker, positive limit and positive finite lease_seconds are required")
        if stage is not None and stage not in {*_STAGES, _DIAGNOSIS_STAGE}:
            raise ValueError(f"invalid stage: {stage}")
        now = time.time()
        # Claiming is a single IMMEDIATE transaction.  The conditional update
        # then remains safe even when several workers race on the same queue.
        claimed: list[OperatorTask] = []
        claim_events: list[tuple[str, str, dict[str, Any], str]] = []
        db = self.store.db
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
                run = self.store.run(row["run_id"])
                if run.metadata.get("evidence_mode", "real") == "real" and not run.metadata.get("artifact_root"):
                    base = Path(self.store.path).parent if self.store.path != ":memory:" else default_state_root()
                    run.metadata["artifact_root"] = str(ensure_external(base / "runs" / safe_component(run.run_id)))
                    self.store.update_run(run)
                    self._emit(run.run_id, "run_workspace_bound", {
                        "artifact_root": run.metadata["artifact_root"],
                    })
                if run.metadata.get("artifact_root"):
                    paths = RunPaths(run.metadata["artifact_root"], run.run_id).allocate_attempt(row["task_id"])
                    payload = json.loads(row["input_json"])
                    payload["workspace"] = {
                        **paths.identity, "root": str(paths.root),
                        "input": str(paths.input), "scratch": str(paths.scratch),
                        "output": str(paths.output), "logs": str(paths.logs),
                    }
                    ArtifactStore(paths.input).write_json("task.json", {
                        "task_id": row["task_id"], "run_id": row["run_id"],
                        "stage": row["stage"], "attempt": row["attempt"] + 1,
                        "input": payload,
                    })
                    db.execute("UPDATE tasks SET input_json=? WHERE task_id=?",
                               (json.dumps(payload, sort_keys=True), row["task_id"]))
                claimed.append(self.store._task(db.execute("SELECT * FROM tasks WHERE task_id=?", (row["task_id"],)).fetchone()))
                claim_events.append((row["run_id"], "task_claimed", {
                    "worker_id": worker_id, "stage": row["stage"], "attempt": row["attempt"] + 1,
                }, row["task_id"]))
        for event in claim_events:
            self._emit(*event)
        return [t for t in claimed if t is not None]

    def _deps_succeeded(self, run_id: str, key: str, stage: str) -> bool:
        # Diagnostics are deliberately independent of the operator pipeline:
        # a failed task must be explainable even when an upstream stage failed.
        if stage == _DIAGNOSIS_STAGE:
            return True
        run = self.store.run(run_id)
        if (run.metadata.get("evidence_mode", "real") != "simulation"
                or run.metadata.get("environment_required")
                or run.metadata.get("graph_environment_required")):
            if run.status != "ENVIRONMENT_READY":
                return False
            spec = self.store.operator(run_id, key)
            fingerprint = run.environment.get("environment_proof", {}).get("fingerprint")
            if not fingerprint or spec.environment.get("fingerprint") != fingerprint:
                return False
        if stage == "torch":
            return True
        prev = _STAGES[_STAGES.index(stage) - 1]
        row = self.store.db.execute("SELECT status FROM tasks WHERE run_id=? AND operator_key=? AND stage=?", (run_id, key, prev)).fetchone()
        return bool(row and row["status"] == "succeeded")

    def _owned_lease(self, task_id: str, worker_id: str | None, lease_token: str | None):
        row = self.store.db.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,),
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        # Retain the concise positional API, but never accept a worker name alone.
        if not lease_token and worker_id == row["lease_token"]:
            lease_token, worker_id = worker_id, None
        if (not lease_token or row["lease_token"] != lease_token
                or (worker_id is not None and row["lease_worker"] != worker_id)
                or row["status"] != "running"
                or row["lease_expires"] is None or row["lease_expires"] <= time.time()):
            raise ValueError("task requires its current unexpired lease token and matching worker")
        return row

    @atomic_transition
    def renew_lease(self, task_id: str, worker_id: str, lease_token: str,
                    lease_seconds: float = 300) -> OperatorTask | DiagnosticTask:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive and finite")
        row = self._owned_lease(task_id, worker_id, lease_token)
        expires = time.time() + lease_seconds
        self.store.db.execute(
            "UPDATE tasks SET lease_expires=?,updated_at=? WHERE task_id=? AND lease_token=?",
            (expires, time.time(), task_id, lease_token),
        )
        self._emit(row["run_id"], "lease_renewed", {"lease_expires": expires}, task_id)
        return self.store.get_task(task_id)

    @atomic_transition
    def complete(self, task_id: str, worker_id: str | None = None, result: dict[str, Any] | None = None,
                 lease_token: str | None = None) -> OperatorTask | DiagnosticTask:
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        row = self._owned_lease(task_id, worker_id, lease_token)
        worker_id, lease_token = row["lease_worker"], row["lease_token"]
        output = {} if result is None else result
        if not isinstance(output, dict):
            raise ValueError("result must be a mapping")
        errors = validate_result(task, self.store.run(task.run_id), output, worker_id)
        if errors:
            # A worker may report a rejected verdict through ``complete``.  It
            # is terminal failure for this stage and must never create the
            # downstream task; persist it through the same ownership checks as
            # an explicit ``fail`` call.
            return self.fail(
                task_id,
                worker_id=worker_id,
                lease_token=lease_token,
                error={"reason": "result_validation_failed", "result": output,
                       "validation_errors": errors},
            )
        self._owned_lease(task_id, worker_id, lease_token)
        output = {**output, "_submission": {"worker": worker_id, "attempt": task.attempt}}
        try:
            self._record_attempt(task, output, "PASS")
        except (OSError, WritePolicyError) as exc:
            return self.fail(
                task_id, worker_id=worker_id, lease_token=lease_token,
                error={"reason": "artifact_registration_failed", "message": str(exc)},
            )
        self._owned_lease(task_id, worker_id, lease_token)
        now = time.time()
        self.store.db.execute("UPDATE tasks SET status='succeeded',output_json=?,lease_token=NULL,lease_worker=NULL,lease_expires=NULL,updated_at=? WHERE task_id=?", (json.dumps(output, sort_keys=True), now, task_id))
        task.output = output
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

    def _record_attempt(self, task: OperatorTask, output: dict[str, Any], outcome: str) -> None:
        workspace = task.input.get("workspace")
        if workspace is None:
            return
        store = ArtifactStore(workspace["root"])
        store.write_json("result.json", output, overwrite=True)
        manifest = store.register(identity={
            "run_id": task.run_id, "task_id": task.task_id, "stage": task.stage,
            "attempt": task.attempt, "attempt_id": workspace["attempt_id"],
        }, outcome=outcome, required=["input/task.json", "result.json"])
        output["artifact_manifest"] = str(manifest)

    def _ensure_next(self, task: OperatorTask, stage: str) -> None:
        next_task = OperatorTask(task_id=f"{task.run_id}:{task.operator_key}:{stage}", run_id=task.run_id,
                                 operator_key=task.operator_key, stage=stage, status="pending",
                                 input={"upstream_task_id": task.task_id,
                                        "operator_spec": self.store.operator(task.run_id, task.operator_key).to_dict(),
                                        "upstream_result": task.output})
        if self.store.insert_task(next_task):
            self._emit(task.run_id, "task_created", {"stage": stage, "operator_key": task.operator_key}, next_task.task_id)

    @atomic_transition
    def fail(self, task_id: str, worker_id: str | None = None, error: Any = None,
             lease_token: str | None = None) -> OperatorTask | DiagnosticTask:
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        self._owned_lease(task_id, worker_id, lease_token)
        report = BugReport.from_error(task_id, error)
        # Attach the failed task's execution contract when the runner did not
        # provide context itself.  This gives diagnosis workers enough detail
        # to reproduce the issue without querying volatile worker state.
        report.context.setdefault("failed_stage", task.stage)
        report.context.setdefault("task_input", task.input)
        if task.output:
            report.context.setdefault("task_output", task.output)
        failure_output = {"error": report.to_dict()}
        try:
            self._record_attempt(task, failure_output, "FAILED")
        except (OSError, WritePolicyError) as exc:
            report.metadata["artifact_registration_error"] = str(exc)
            failure_output["error"] = report.to_dict()
        self._owned_lease(task_id, worker_id, lease_token)
        self.store.db.execute("UPDATE tasks SET status='failed',output_json=?,lease_token=NULL,lease_worker=NULL,lease_expires=NULL,updated_at=? WHERE task_id=?", (json.dumps(failure_output, sort_keys=True), time.time(), task_id))
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

    @atomic_transition
    def recover(self, force: bool = False) -> int:
        clause = "1=1" if force else "lease_expires <= ?"
        args: tuple[Any, ...] = () if force else (time.time(),)
        cur = self.store.db.execute(f"UPDATE tasks SET status='pending',lease_token=NULL,lease_worker=NULL,lease_expires=NULL,updated_at=? WHERE status='running' AND {clause}", (time.time(), *args))
        return cur.rowcount

    @atomic_transition
    def reconcile(self, run_id: str) -> dict[str, Any]:
        """Repair historical missing edges without trusting legacy bare verdicts."""
        run = self.store.run(run_id)
        if run is None:
            raise KeyError(run_id)
        created: list[str] = []
        blocked: list[dict[str, Any]] = []
        for task in self.store.tasks(run_id):
            if task.stage == _DIAGNOSIS_STAGE:
                continue
            if task.status == "failed" and self.diagnosis_for(task.task_id) is None:
                error = task.output.get("error", {"message": "historical task failure"})
                diagnosis = self._ensure_diagnosis(task, BugReport.from_error(task.task_id, error))
                created.append(diagnosis.task_id)
            elif task.status == "succeeded" and task.stage != "integration":
                next_stage = _STAGES[_STAGES.index(task.stage) + 1]
                next_id = f"{run_id}:{task.operator_key}:{next_stage}"
                if self.store.get_task(next_id) is not None:
                    continue
                submission = task.output.get("_submission", {})
                worker = submission.get("worker") if submission.get("attempt") == task.attempt else None
                if not worker:
                    claims = [
                        event for event in self.store.events(run_id)
                        if event.task_id == task.task_id and event.event_type == "task_claimed"
                    ]
                    matching = [event for event in claims if event.payload.get("attempt") == task.attempt]
                    if matching:
                        worker = matching[-1].payload.get("worker_id")
                    elif len(claims) == task.attempt:
                        worker = claims[-1].payload.get("worker_id")
                if not worker:
                    errors = ["cannot establish producer identity for independent validation"]
                else:
                    errors = validate_result(task, run, task.output, worker)
                if errors:
                    blocked.append({"task_id": task.task_id, "validation_errors": errors})
                    continue
                self._ensure_next(task, next_stage)
                created.append(next_id)
        result = {"created": created, "blocked": blocked}
        self._emit(run_id, "run_reconciled", result)
        return result

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
