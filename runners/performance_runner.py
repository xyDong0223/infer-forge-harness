"""Deterministic performance workflow primitives.

The runner delegates device-specific work to a PerformanceAdapter and never
assumes a profiler, device, or serving engine.
"""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.contracts import Metric, PerformanceAdapter, Workload
from core.performance import compare_metrics


def _metric_record(metric: Metric) -> dict[str, Any]:
    record = asdict(metric)
    if not math.isfinite(metric.value):
        # Preserve invalid measurements without emitting nonstandard JSON numbers.
        record["value"] = str(metric.value)
    return record


class PerformanceRunner:
    def __init__(self, adapter: PerformanceAdapter, artifact_root: str | Path):
        self.adapter = adapter
        self.artifact_root = Path(artifact_root)

    def run(self, workload: Workload, baseline: list[Metric] | None = None) -> dict[str, Any]:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        baseline = list(baseline) if baseline is not None else []
        self.adapter.prepare_workload(workload)
        metrics = list(self.adapter.run_benchmark(workload))
        artifacts = list(self.adapter.collect_trace(workload))
        extracted = list(self.adapter.extract_metrics(artifacts))
        gates = compare_metrics(baseline, metrics)
        verdicts = {gate.verdict for gate in gates}
        if "FAIL" in verdicts:
            status = "FAIL"
        elif "INCOMPARABLE" in verdicts:
            status = "INCOMPARABLE"
        elif gates and verdicts == {"PASS"}:
            status = "PASS"
        else:
            status = "UNKNOWN"
        report_path = (self.artifact_root / "performance_report.json").resolve()
        benchmark_metrics = [_metric_record(metric) for metric in metrics]
        report = {
            "status": status,
            "workload": asdict(workload),
            "baseline": [_metric_record(metric) for metric in baseline],
            "metrics": benchmark_metrics,
            "benchmark_metrics": benchmark_metrics,
            "trace_metrics": [_metric_record(metric) for metric in extracted],
            "artifacts": [asdict(artifact) for artifact in artifacts],
            "gates": [{**asdict(gate), "evidence": [str(report_path)]} for gate in gates],
            "report_path": str(report_path),
        }
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        pending = report_path.with_name(f".{report_path.name}.{uuid4().hex}.pending")
        try:
            with pending.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pending, report_path)
        finally:
            pending.unlink(missing_ok=True)
        return json.loads(serialized)
