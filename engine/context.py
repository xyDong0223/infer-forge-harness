"""Read-only Agent views over existing run/task contracts, not a second state model."""

from __future__ import annotations

import hashlib
from pathlib import Path

from core.paths import REPO_ROOT
from .progress import run_progress
from .result_validation import STAGE_EVIDENCE
from .interaction import active_graph_handoff


def resource_reference(path: Path) -> dict:
    """Identify guidance/evidence without treating a hash as validation."""
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _task_view(task) -> dict:
    # Observation and persisted packets are not lease credentials. Only the
    # claim/renew response is used to authorize a worker's later submission.
    return {key: value for key, value in task.to_dict().items() if key != "lease_token"}


def task_packet(store, run, task) -> dict:
    """Project an actionable task packet from the authoritative scheduler data.

    The scheduler persists this same projection in input/task.json on claim.
    Querying a task never allocates an attempt or revalidates accepted evidence.
    """
    if task.run_id != run.run_id:
        raise ValueError("task does not belong to the requested run")
    spec = store.operator(run.run_id, task.operator_key)
    previous = {"torch": (), "xpu": ("torch",), "integration": ("torch", "xpu"),
                "diagnosis": ()}[task.stage]
    source_task_id = task.input.get("source_task_id")
    upstream = [
        _task_view(item) for item in store.tasks(run.run_id)
        if item.task_id == source_task_id or (
            item.operator_key == task.operator_key and item.stage in previous
            and item.status == "succeeded"
        )
    ]
    return {
        "schema_version": 1,
        **_task_view(task),
        "run_identity": {
            "run_id": run.run_id, "model_id": run.model_id,
            "model_revision": run.model_revision, "plugin_revision": run.plugin_revision,
            "backend": run.backend, "evidence_mode": run.metadata.get("evidence_mode", "real"),
            "artifact_root": run.metadata.get("artifact_root"),
        },
        "environment": run.environment,
        "operator_spec": spec.to_dict() if spec is not None else None,
        "upstream_tasks": upstream,
        "acceptance": {
            "required_evidence": list(STAGE_EVIDENCE[task.stage]),
            "result_identity": {
                "task_id": task.task_id, "operator_key": task.operator_key,
                "stage": task.stage, "attempt": task.attempt,
                "environment_fingerprint": run.environment.get("environment_proof", {}).get("fingerprint"),
                "evidence_mode": run.metadata.get("evidence_mode", "real"),
            },
            "result_schema": resource_reference(REPO_ROOT / "contracts/worker_result.schema.yaml"),
            "validator": resource_reference(REPO_ROOT / "engine/result_validation.py"),
            "output_directory": task.input.get("workspace", {}).get("output"),
        },
        "guidance": {
            "agent_protocol": resource_reference(REPO_ROOT / "AGENTS.md"),
            "worker_results": resource_reference(REPO_ROOT / "docs/migration/worker-results.md"),
        },
        "evidence_revalidated": False,
    }


def run_context(store, run_id: str, task_id: str | None = None) -> dict:
    """Return current context under the caller's read transaction, without writes."""
    run = store.run(run_id)
    if run is None:
        raise ValueError(f"unknown run: {run_id}")
    selected = None
    if task_id is not None:
        selected = store.get_task(task_id)
        if selected is None:
            raise ValueError(f"unknown task: {task_id}")
        if selected.run_id != run_id:
            raise ValueError("task does not belong to the requested run")
    packet_reference = None
    if selected is not None:
        input_dir = selected.input.get("workspace", {}).get("input")
        if input_dir:
            path = Path(input_dir) / "task.json"
            if path.is_file():
                packet_reference = resource_reference(path)
    return {
        "schema_version": 1,
        "run": run.to_dict(),
        "tasks": [
            {key: value for key, value in _task_view(task).items()
             if key not in {"input", "output"}}
            for task in store.tasks(run_id)
        ],
        "task_context": task_packet(store, run, selected) if selected is not None else None,
        # A live projection can differ from the immutable claim-time packet.
        # The reference makes that distinction explicit for resumed workers.
        "claimed_packet": packet_reference,
        "restart": {
            "execution_context": run.metadata.get("graph_execution_context"),
            "resume_command": run.metadata.get("graph_progress", {}).get("resume_command"),
        },
        "handoff": active_graph_handoff(run),
        "evidence_revalidated": False,
        **run_progress(store, run_id),
    }
