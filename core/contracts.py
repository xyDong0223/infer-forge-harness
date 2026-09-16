"""Stable data contracts shared by adaptation and performance workflows."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class TargetContext:
    model: str
    hardware: str
    engine: str
    backend: str
    plugin: str | None = None
    revisions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Capability:
    name: str
    status: str = "UNKNOWN"
    constraints: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceSnapshot:
    devices: int
    memory_total_bytes: int | None = None
    memory_free_bytes: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Workload:
    name: str
    input_lengths: tuple[int, ...] = ()
    output_lengths: tuple[int, ...] = ()
    concurrency: tuple[int, ...] = ()
    request_rate: tuple[float, ...] = ()


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: str
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: str
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    name: str
    verdict: str
    reason: str = ""
    evidence: tuple[str, ...] = ()


class RuntimeAdapter(Protocol):
    """Runtime seam consumed by platform-neutral deployment tasks."""

    def env_prefix(self) -> str: ...
    def installer_path(self, repo_root: Path) -> Path: ...
    def import_check_command(self) -> str: ...
    def build_serve_command(self, server: dict[str, Any]) -> str: ...
    def fallback_markers(self) -> tuple[str, ...]: ...
    def environment_fingerprint_command(self, workdir: str) -> str: ...


class HardwareAdapter(Protocol):
    """Hardware/cluster seam consumed by deployment and probes."""

    def exec(self, pod: str, command: str, **kwargs: Any) -> Any: ...
    def copy_into(self, pod: str, source: Path, destination: str) -> Any: ...
    def xpu_smi(self, pod: str) -> Any: ...


class PerformanceAdapter(Protocol):
    """Profiler and benchmark seam shared by performance workflows."""

    def prepare_workload(self, workload: Workload) -> Any: ...
    def run_benchmark(self, workload: Workload) -> Sequence[Metric]: ...
    def collect_trace(self, workload: Workload) -> Sequence[Artifact]: ...
    def extract_metrics(self, artifacts: Sequence[Artifact]) -> Sequence[Metric]: ...


@dataclass(frozen=True)
class RunContext:
    """Shared context for adaptation and performance workflows."""

    target: TargetContext
    artifact_root: str
    workload: Workload | None = None
    environment_fingerprint: str | None = None


class EngineAdapter(Protocol):
    def build_serve_command(self, server: dict[str, Any]) -> str: ...
    def readiness_probe(self, endpoint: str) -> Any: ...
