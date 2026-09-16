"""Deterministic performance workflow primitives.

The runner delegates device-specific work to a PerformanceAdapter and never
assumes a profiler, device, or serving engine.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
from typing import Any

from core.contracts import Metric, PerformanceAdapter, Workload
from core.performance import compare_metrics
from core.storage import ArtifactStore, RunPaths, WritePolicyError, ensure_external, locate_attempt


def _metric_record(metric: Metric) -> dict[str, Any]:
    record = asdict(metric)
    if not math.isfinite(metric.value):
        # Preserve invalid measurements without emitting nonstandard JSON numbers.
        record["value"] = str(metric.value)
    return record


class PerformanceRunner:
    def __init__(self, adapter: PerformanceAdapter, artifact_root: str | Path, run_id: str | None = None):
        self.adapter = adapter
        self.artifact_root = ensure_external(artifact_root)
        if locate_attempt(self.artifact_root) is not None:
            raise WritePolicyError("performance artifact_root must be a run root, not an existing attempt")
        self.run_id = run_id

    def run(self, workload: Workload, baseline: list[Metric] | None = None) -> dict[str, Any]:
        attempt = RunPaths(self.artifact_root, self.run_id).initialize().allocate_attempt(
            f"performance-{workload.name}"
        )
        store = ArtifactStore(attempt.output)
        inventory = ArtifactStore(attempt.root)
        ArtifactStore(attempt.input).write_json("workload.json", asdict(workload))
        try:
            report = self._run(workload, baseline, store)
            report["artifact_root"] = str(attempt.output)
            report["manifest_path"] = str(attempt.root / "manifest.json")
            report["workspace_identity"] = attempt.identity
            store.write_json("performance_report.json", report)
            inventory.register(
                identity=attempt.identity, outcome=report["status"],
                required=["output/performance_report.json"],
            )
        except (OSError, ValueError, RuntimeError) as error:
            failure = {
                "status": "BLOCKED" if isinstance(error, WritePolicyError) else "ERROR",
                "message": str(error), "workload": asdict(workload),
                "artifact_root": str(attempt.output),
                "report_path": str(attempt.output / "performance_report.json"),
                "manifest_path": str(attempt.root / "manifest.json"),
                "workspace_identity": attempt.identity,
            }
            try:
                store.write_json("performance_report.json", failure, overwrite=True)
                inventory.register(
                    identity=attempt.identity, outcome=failure["status"],
                    required=["output/performance_report.json"],
                )
            except (OSError, ValueError, RuntimeError) as publication_error:
                failure.update(manifest_path=None, publication_error=str(publication_error))
                try:
                    store.write_json("performance_report.json", failure, overwrite=True)
                except (OSError, ValueError, RuntimeError) as status_error:
                    raise error from status_error
                raise error from publication_error
            raise
        return json.loads(json.dumps(report, allow_nan=False))

    def _run(self, workload, baseline, store: ArtifactStore) -> dict[str, Any]:
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
        report_path = store.path("performance_report.json")
        artifact_records = []
        for index, artifact in enumerate(artifacts):
            record = asdict(artifact)
            source = Path(artifact.path)
            if source.is_file():
                destination = store.path(Path("artifacts") / f"{index:04d}" / source.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                record["path"] = str(destination)
                record["metadata"] = {**record["metadata"], "original_path": artifact.path}
            elif source.is_absolute() and (
                source == store.root or store.root in source.parents
            ):
                raise WritePolicyError(f"missing local performance artifact: {source}")
            artifact_records.append(record)
        benchmark_metrics = [_metric_record(metric) for metric in metrics]
        report = {
            "status": status,
            "workload": asdict(workload),
            "baseline": [_metric_record(metric) for metric in baseline],
            "metrics": benchmark_metrics,
            "benchmark_metrics": benchmark_metrics,
            "trace_metrics": [_metric_record(metric) for metric in extracted],
            "artifacts": artifact_records,
            "gates": [{**asdict(gate), "evidence": [str(report_path)]} for gate in gates],
            "report_path": str(report_path),
        }
        return report
