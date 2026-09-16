"""Deterministic performance workflow primitives.

The runner delegates device-specific work to a PerformanceAdapter and never
assumes a profiler, device, or serving engine.
"""

from pathlib import Path
from typing import Any

from core.contracts import Artifact, Metric, PerformanceAdapter, Workload
from core.performance import compare_metrics


class PerformanceRunner:
    def __init__(self, adapter: PerformanceAdapter, artifact_root: str | Path):
        self.adapter = adapter
        self.artifact_root = Path(artifact_root)

    def run(self, workload: Workload, baseline: list[Metric] | None = None) -> dict[str, Any]:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.adapter.prepare_workload(workload)
        metrics = list(self.adapter.run_benchmark(workload))
        artifacts = list(self.adapter.collect_trace(workload))
        extracted = list(self.adapter.extract_metrics(artifacts))
        gates = compare_metrics(baseline or [], extracted or metrics)
        return {
            "status": "PASS" if all(g.verdict in ("PASS", "UNKNOWN") for g in gates) else "FAIL",
            "metrics": [m.__dict__ for m in (extracted or metrics)],
            "artifacts": [a.__dict__ for a in artifacts],
            "gates": [g.__dict__ for g in gates],
        }
