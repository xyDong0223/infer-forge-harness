"""One foreground controller turn over the existing Graph and decision contracts."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

from core.storage import ArtifactStore, RunPaths, ensure_external
from engine.brain import Decision
from engine.interaction import (
    accept_graph_decision, current_graph_handoff, finish_graph_decision,
)
from engine.progress import run_progress
from engine.run_control import control_run


def graph_arguments(scheduler, run_id: str, settings=()) -> SimpleNamespace:
    """Reconstruct typed inputs, never execute a recorded shell command."""
    run = scheduler.store.run(run_id)
    if run is None:
        raise ValueError(f"unknown run: {run_id}")
    saved = run.metadata.get("graph_execution_context")
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        raise ValueError("no supported graph execution context; configure the first connected Graph invocation")
    if saved.get("run_id") != run_id or saved.get("subject") != run.model_id:
        raise ValueError("saved Graph identity does not match the run")
    if (ensure_external(saved["scheduler_state"]) != ensure_external(scheduler.store.path)
            or ensure_external(saved["artifact_root"]) != ensure_external(run.metadata["artifact_root"])):
        raise ValueError("saved Graph state/artifact paths do not match the run")
    workflow = Path(saved["workflow"])
    if (not workflow.is_absolute() or not workflow.is_file()
            or hashlib.sha256(workflow.read_bytes()).hexdigest() != saved.get("workflow_sha256")):
        raise ValueError("saved workflow changed or is missing; inspect it and configure a new Graph invocation explicitly")
    if saved.get("auto_recover") or saved.get("decide_command"):
        raise ValueError("Codex interaction cannot share control with automatic recovery or an external decider")
    for key in ("env", "set"):
        if not isinstance(saved.get(key), list) or any(
            not isinstance(value, str) or "=" not in value for value in saved[key]
        ):
            raise ValueError(f"saved {key} must contain ordered KEY=VALUE strings")
    if any(not isinstance(value, str) or "=" not in value for value in settings):
        raise ValueError("settings must contain KEY=VALUE strings")
    budget = saved.get("recovery_budget", 3)
    if type(budget) is not int or budget < 1:
        raise ValueError("saved recovery_budget must be a positive integer")
    watch = saved.get("watch_interval", 30.0)
    if isinstance(watch, bool) or not isinstance(watch, (int, float)) or not math.isfinite(watch) or watch < 0:
        raise ValueError("saved watch_interval must be finite and nonnegative")
    args = SimpleNamespace(
        workflow=workflow, subject=run.model_id, run_id=run_id,
        scheduler_state=ensure_external(saved["scheduler_state"]),
        artifact_root=ensure_external(saved["artifact_root"]),
        journal=ensure_external(saved["journal"]), loop_state=ensure_external(saved["loop_state"]),
        env=list(saved["env"]), set=[*saved["set"], *settings],
        target=Path(saved["target"]) if saved.get("target") else None,
        operator_report=Path(saved["operator_report"]) if saved.get("operator_report") else None,
        shim_registry=Path(saved["shim_registry"]) if saved.get("shim_registry") else None,
        execute=True, resume=True, from_node=None, until_node=None, json=True,
        interaction_mode="codex", auto_recover=False, brain="agent", recovery_budget=budget,
        decide_command=None, decide_timeout=600.0, watch_interval=watch,
    )
    return args


def _view(scheduler, run_id: str, *, exit_code=0, **extra) -> dict:
    handoff = current_graph_handoff(scheduler, run_id)
    if handoff and exit_code == 0:
        exit_code = 4 if handoff["state"] == "pending" else 2
    return {"run_id": run_id, "exit_code": exit_code, "handoff": handoff,
            **run_progress(scheduler.store, run_id), **extra}


def _capture(args, operation: str, invocation: dict, execute):
    """Keep the CLI's stdout one JSON document; full Graph logs stay durable."""
    attempt = RunPaths(args.artifact_root, args.run_id).allocate_attempt(operation)
    ArtifactStore(attempt.input).write_json("invocation.json", invocation)
    try:
        with (attempt.logs / "console.log").open("w", encoding="utf-8") as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                result = execute()
        ArtifactStore(attempt.output).write_json("result.json", result)
    except BaseException as error:
        ArtifactStore(attempt.output).write_json("execution_unknown.json", {
            "status": "UNKNOWN", "error_type": type(error).__name__, "error": str(error),
        })
        ArtifactStore(attempt.root).register(identity=attempt.identity, outcome="UNKNOWN")
        raise
    ArtifactStore(attempt.root).register(identity=attempt.identity, outcome="RECORDED")
    return result, str(attempt.root)


def advance_run(scheduler, run_id: str, *, settings=()) -> dict:
    """Advance deterministic work, or return the current decision/worker boundary."""
    if scheduler.store.run(run_id) is None:
        raise ValueError(f"unknown run: {run_id}")
    with control_run(scheduler.store.path, run_id):
        if current_graph_handoff(scheduler, run_id):
            if settings:
                raise ValueError("an unresolved handoff cannot be bypassed with new settings")
            return _view(scheduler, run_id)
        tasks = scheduler.store.tasks(run_id)
        if any(task.status != "succeeded" for task in tasks):
            if settings:
                raise ValueError("finish the pending worker/diagnosis work before changing Graph settings")
            result = _view(scheduler, run_id, exit_code=3)
            if result["progress"]["state"] == "BLOCKED":
                result["exit_code"] = 2
            return result
        progress = run_progress(scheduler.store, run_id)["progress"]
        if progress["state"] == "COMPLETED" and not settings:
            return _view(scheduler, run_id)  # Readback, not a new delivery claim.
        args = graph_arguments(scheduler, run_id, settings)
        from runners.graph_runner import run
        result, artifact_root = _capture(args, "codex-advance", {
            "command": "advance", "run_id": run_id, "settings": list(settings),
        }, lambda: {"exit_code": run(args)})
        return _view(scheduler, run_id, exit_code=result["exit_code"], artifact_root=artifact_root)


def submit_graph_decision(scheduler, run_id: str, handoff_id: str, decision_id: str,
                          expected_version: str, decision: dict) -> dict:
    """Accept once, execute once, then persist the execution/next handoff atomically."""
    if scheduler.store.run(run_id) is None:
        raise ValueError(f"unknown run: {run_id}")
    with control_run(scheduler.store.path, run_id):
        # Replays must work even after execution legitimately changed the
        # workflow/context. Acceptance validates an existing ID first.
        prior = scheduler.store.run(run_id).metadata.get("graph_decisions", {}).get(decision_id)
        args = None if prior else graph_arguments(scheduler, run_id)
        receipt, execute_now = accept_graph_decision(
            scheduler, run_id, handoff_id, decision_id, expected_version, decision,
        )
        if not execute_now:
            return _view(scheduler, run_id, receipt=receipt, replayed=True)
        handoff = current_graph_handoff(scheduler, run_id)
        candidate = Decision.from_dict(receipt["decision"])
        if candidate.next_action == "BLOCKED":
            receipt = finish_graph_decision(scheduler, run_id, handoff_id, decision_id, {
                "status": "BLOCKED", "reason": candidate.diagnosis,
            })
            return _view(scheduler, run_id, receipt=receipt, replayed=False)
        from runners.graph_runner import (
            NODES, create_interactive_handoff, execute_graph_decision, scheduled_spec,
        )
        try:
            outcome, artifact_root = _capture(args, "codex-decision", {
                "receipt": receipt,
            }, lambda: execute_graph_decision(args, handoff, candidate, scheduler))
        except Exception as error:
            # Do not mark an uncertain external execution complete or replay it.
            return _view(scheduler, run_id, exit_code=2, receipt=receipt, replayed=False,
                         execution_error=str(error))
        internal = {key: outcome[key] for key in ("retry_context", "environment", "skill", "task_type")
                    if key in outcome}
        persisted = {key: value for key, value in outcome.items() if key not in internal}
        persisted["controller_artifacts"] = artifact_root
        # No crash window may complete an old handoff without preserving the
        # remaining budget on the new one. If finalization fails, acceptance
        # stays executing/unknown and cannot be retried automatically.
        try:
            with scheduler.store.transaction():
                receipt = finish_graph_decision(scheduler, run_id, handoff_id, decision_id, persisted)
                if outcome["status"] == "REWORK":
                    finished = scheduler.store.run(run_id).metadata["graph_handoffs"][handoff_id]
                    create_interactive_handoff(
                        args, node=outcome["node"],
                        spec=scheduled_spec(NODES[internal["task_type"]], internal["retry_context"]),
                        context=internal["retry_context"], artifacts=Path(outcome["final_artifacts"]),
                        environment=internal["environment"], state=outcome["state"],
                        task_type=internal["task_type"], skill=internal["skill"],
                        bridge=SimpleNamespace(scheduler=scheduler),
                        remaining_budget=receipt["remaining_budget"], history=finished["history"],
                    )
                elif outcome["status"] == "RECOVERED":
                    scheduler.record_graph_transition(run_id, "graph_progress", {
                        "status": "READY", "reason_code": "GRAPH_RECOVERY_SUCCEEDED",
                        "node": outcome["node"], "artifacts": [outcome["final_artifacts"]],
                        "message": "Submitted recovery passed the node validator; advance to continue the graph.",
                    })
        except Exception as error:
            receipt = scheduler.store.run(run_id).metadata["graph_decisions"][decision_id]
            return _view(scheduler, run_id, exit_code=2, receipt=receipt, replayed=False,
                         execution_error=f"execution finished but receipt finalization failed: {error}")
        return _view(scheduler, run_id, receipt=receipt, replayed=False,
                     artifact_root=artifact_root)
