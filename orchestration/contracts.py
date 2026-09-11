"""Stable contracts for event-driven operator adaptation orchestration."""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


def _canonical(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class BugReport:
    """Structured evidence passed to a diagnostic sub-agent.

    ``error`` values from runners are often strings or arbitrary JSON.  The
    scheduler normalizes them into this contract so a diagnostic worker always
    receives the original message plus any available execution context.
    """

    source_task_id: str
    error_type: str = "unknown"
    message: str = ""
    traceback: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    occurred_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_error(cls, source_task_id: str, error: Any) -> "BugReport":
        if isinstance(error, cls):
            return error
        if isinstance(error, dict):
            # Preserve arbitrary fields under metadata while accepting the
            # conventional names emitted by subprocess and validator runners.
            payload = dict(error)
            source = str(payload.pop("source_task_id", source_task_id))
            error_type = str(payload.pop("error_type", payload.pop("type", "unknown")))
            message_value = payload.pop("message", None)
            if message_value is None:
                message_value = payload.pop("error", "")
            traceback = payload.pop("traceback", None)
            context = dict(payload.pop("context", {}) or {})
            occurred_at = float(payload.pop("occurred_at", time.time()))
            schema_version = int(payload.pop("schema_version", 1))
            metadata = {**dict(payload.pop("metadata", {}) or {}), **payload}
            return cls(source_task_id=source, error_type=error_type,
                       message=str(message_value), traceback=traceback,
                       context=context, occurred_at=occurred_at,
                       metadata=metadata, schema_version=schema_version)
        return cls(source_task_id=source_task_id, message=str(error or ""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BugReport":
        return cls(**d)

@dataclass
class IOSpec:
    name: str
    dtype: str
    shape: Any
    layout: str
    def __post_init__(self):
        for k in ("name", "dtype", "layout"):
            if not str(getattr(self, k)).strip():
                raise ValueError(f"{k} is required")
        if self.shape is None:
            raise ValueError("shape is required")
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d): return cls(**d)

@dataclass
class OperatorSpec:
    operator_id: str
    model_id: str
    model_revision: str
    plugin_revision: str
    backend: str
    inputs: list[IOSpec]
    outputs: list[IOSpec]
    semantics: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    def __post_init__(self):
        if not self.operator_id or not self.model_id or not self.backend:
            raise ValueError("operator_id, model_id and backend are required")
        if not self.inputs or not self.outputs:
            raise ValueError("inputs and outputs are required")
        if not self.semantics:
            raise ValueError("semantics are required")
        self.inputs = [x if isinstance(x, IOSpec) else IOSpec.from_dict(x) for x in self.inputs]
        self.outputs = [x if isinstance(x, IOSpec) else IOSpec.from_dict(x) for x in self.outputs]
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")
    @property
    def operator_key(self) -> str:
        body = {"model_id":self.model_id,"model_revision":self.model_revision,"plugin_revision":self.plugin_revision,"backend":self.backend,"environment":self.environment,"semantics":self.semantics,"inputs":[x.to_dict() for x in self.inputs],"outputs":[x.to_dict() for x in self.outputs]}
        return hashlib.sha256(_canonical(body).encode()).hexdigest()
    def to_dict(self):
        d = asdict(self)
        d["inputs"] = [x.to_dict() for x in self.inputs]
        d["outputs"] = [x.to_dict() for x in self.outputs]
        return d
    @classmethod
    def from_dict(cls, d):
        values = dict(d)
        # operator_key is derived from the canonical specification and is
        # intentionally ignored on load so persisted records remain portable.
        values.pop("operator_key", None)
        return cls(**values)
    def to_json(self): return _canonical(self.to_dict())
    @classmethod
    def from_json(cls,s): return cls.from_dict(json.loads(s))

@dataclass
class OperatorTask:
    task_id: str
    run_id: str
    operator_key: str
    stage: str
    status: str = "pending"
    attempt: int = 0
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    lease_token: str | None = None
    schema_version: int = 1
    def __post_init__(self):
        if self.stage not in {"torch", "xpu", "integration", "diagnosis"}:
            raise ValueError("invalid stage")
        if self.status not in {"pending", "running", "succeeded", "failed"}:
            raise ValueError("invalid status")
    @property
    def idempotency_key(self): return f"{self.run_id}:{self.operator_key}:{self.stage}"
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls,d): return cls(**d)


@dataclass
class DiagnosticTask:
    """Typed view of a diagnosis task persisted in the scheduler queue."""

    task_id: str
    run_id: str
    operator_key: str
    source_task_id: str
    status: str = "pending"
    attempt: int = 0
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    lease_token: str | None = None
    stage: str = "diagnosis"
    schema_version: int = 1

    def __post_init__(self):
        if self.status not in {"pending", "running", "succeeded", "failed"}:
            raise ValueError("invalid status")

    @property
    def idempotency_key(self) -> str:
        return f"{self.run_id}:{self.operator_key}:diagnosis:{self.source_task_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiagnosticTask":
        return cls(**d)

@dataclass
class TaskEvent:
    event_id: str
    run_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    schema_version: int = 1
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls,d): return cls(**d)

@dataclass
class AdaptationRun:
    run_id: str
    model_id: str
    model_revision: str = "unknown"
    plugin_revision: str = "unknown"
    backend: str = "unknown"
    status: str = "running"
    environment: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = 1
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls,d): return cls(**d)
