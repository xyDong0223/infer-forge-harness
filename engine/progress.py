"""Read-only explanations of graph and scheduler state, never acceptance verdicts.

The original states remain authoritative. This projection gives people and
controllers the same vocabulary for location, reason, owner and next action.
Commands are argv lists for review, never executed by this module.
"""
from __future__ import annotations

import shlex
import sys
import time
from datetime import datetime, timezone

from core.paths import REPO_ROOT
from .interaction import active_graph_handoff


def explanation(state, location, reason_code, summary, owner, action, instruction,
                *, command=None, evidence=(), observed_at=None, **details):
    return {
        "schema_version": 1, "state": state, "location": location,
        "reason_code": reason_code, "summary": summary,
        "next_action": {"owner": owner, "action": action,
                        "instruction": instruction, "command": command},
        "evidence": list(dict.fromkeys(str(item) for item in evidence if item)),
        "observed_at": time.time() if observed_at is None else observed_at,
        "details": details,
    }


# These are explanations, not routing rules. The workflow/validators still
# decide whether execution can proceed.
GRAPH_REASONS = {
    "WRITE_POLICY": ("BLOCKED", "main_agent", "FIX_PATHS",
                     "Use fresh external runtime paths; preserve existing attempts."),
    "INVALID_INPUT": ("BLOCKED", "main_agent", "FIX_INPUT",
                      "Correct the reported input and rerun the same run."),
    "INPUT_UNRESOLVED": ("BLOCKED", "main_agent", "SUPPLY_INPUT",
                         "Supply the missing input or execute its producer, then resume."),
    "TARGET_MISMATCH": ("BLOCKED", "main_agent", "FIX_TARGET",
                        "Check the supported target and pinned run identity before resuming."),
    "ENVIRONMENT_MISMATCH": ("BLOCKED", "main_agent", "PROVE_ENVIRONMENT",
                             "Inspect the bound Pod and environment evidence; re-prove the environment."),
    "ENVIRONMENT_FAILED": ("BLOCKED", "main_agent", "PROVE_ENVIRONMENT",
                           "Inspect the failed proof and repair/re-prove the same prepared Pod."),
    "ENVIRONMENT_COMMAND_FAILED": ("BLOCKED", "main_agent", "INSPECT_ENVIRONMENT_COMMAND",
                                   "Inspect the command exit code and logs, repair the failure, then rerun "
                                   "the environment node without --resume so it executes a fresh attempt."),
    "DISPATCH_BLOCKED": ("BLOCKED", "main_agent", "COMPLETE_OPERATOR_CONTRACT",
                         "Collect the missing measured operator fields; update the report and resume."),
    "SHIM_DISPATCH_BLOCKED": ("BLOCKED", "main_agent", "COMPLETE_SHIM_CONTRACT",
                              "Resolve the listed shim contracts or documented waivers, then resume."),
    "DELIVERY_GATE": ("BLOCKED", "main_agent", "REVALIDATE_DELIVERY",
                      "Resolve the evidence rejection and resume the full graph, including final regressions."),
    "FAILURE_EDGE_CYCLE": ("ACTION_REQUIRED", "main_agent", "BREAK_FAILURE_CYCLE",
                           "Inspect repeated failures and repair their cause before resuming; do not blindly retry."),
    "MANUAL_STEP": ("ACTION_REQUIRED", "main_agent", "EXECUTE_MANUAL_STEP",
                    "Execute the node's task contract and persist its evidence, then resume."),
    "NO_EXECUTOR": ("BLOCKED", "maintainer", "REGISTER_EXECUTOR",
                    "Implement/register the task executor before resuming this node."),
    "NO_CONTRACT": ("BLOCKED", "maintainer", "DEFINE_CONTRACT",
                    "Provide an executable task contract before resuming this node."),
    "SKILL_UNRESOLVED": ("BLOCKED", "maintainer", "FIX_SKILL",
                         "Correct the task's skill binding and method references, then resume."),
    "METHOD_DOC_INVALID": ("BLOCKED", "maintainer", "FIX_SKILL",
                           "Repair the reported method documentation references before execution."),
    "WAITING_FOR_OPERATORS": ("WAITING", "main_agent", "INSPECT_OPERATOR_QUEUE",
                              "Inspect scheduler status, finish pending worker stages, then resume the graph."),
    "OPERATORS_BLOCKED": ("BLOCKED", "main_agent", "INSPECT_OPERATOR_QUEUE",
                          "Inspect scheduler tasks and diagnosis conclusions before resuming."),
    "WAITING_FOR_DECISION": ("WAITING", "decision_agent", "PROVIDE_DECISION",
                             "Read the decision request and write a validated decision to the response path."),
    "RECOVERY_BLOCKED": ("BLOCKED", "main_agent", "REVIEW_RECOVERY",
                         "Inspect the recovery outcome and address its unmet requirement before resuming."),
    "UNTIL_NODE_REACHED": ("READY", "main_agent", "RESUME_GRAPH",
                           "The requested partial walk completed. Resume without --until-node for delivery."),
    "PLAN_COMPLETE": ("READY", "main_agent", "EXECUTE_GRAPH",
                      "Review the plan, then execute with the same inputs and --execute."),
    "GRAPH_ONLY_COMPLETE": ("ACTION_REQUIRED", "main_agent", "VERIFY_DELIVERY",
                            "This graph-only result is not functional readiness; use the scheduler-backed delivery gate."),
    "WORKFLOW_STOPPED": ("ACTION_REQUIRED", "main_agent", "REVIEW_TERMINAL",
                         "Review the workflow's terminal outcome before choosing the next action."),
    "COMMAND_REJECTED": ("BLOCKED", "main_agent", "FIX_COMMAND",
                         "Inspect the original error and current task/attempt before retrying the command."),
}


def graph_progress(summary: dict, resume_command=None) -> dict:
    code = summary.get("reason_code", "UNKNOWN")
    status = summary.get("status", "UNKNOWN")
    location = summary.get("node") or summary.get("command") or "graph"
    message = summary.get("message") or code.replace("_", " ").lower()
    command = None
    if code == "DELIVERY_RECORDED" and status in {"FUNCTIONAL_READY", "SIMULATION_PASS"}:
        values = ("COMPLETED", "none", "NONE",
                  "Delivery recorded. Simulation does not certify real hardware."
                  if status == "SIMULATION_PASS" else "Functional delivery recorded.")
    elif code in GRAPH_REASONS:
        values = GRAPH_REASONS[code]
    elif status in {"CONTINUE", "REUSED", "RECOVERED", "RUNNING"}:
        values = ("RUNNING", "graph_runner", "MONITOR_GRAPH",
                  "Inspect node logs/watch records for liveness; this is the last observed transition.")
    elif status == "REWORK":
        values = ("ACTION_REQUIRED", "main_agent", "INSPECT_FAILURE",
                  "Inspect the original failure and follow the reported failure edge.")
    else:
        values = ("BLOCKED", "main_agent", "INSPECT_EVIDENCE",
                  "Inspect the original status and evidence before deciding how to resume.")
    if values[2] in {"RESUME_GRAPH", "REVALIDATE_DELIVERY", "EXECUTE_GRAPH"}:
        command = resume_command
    elif code == "ENVIRONMENT_COMMAND_FAILED" and resume_command:
        command = [part for part in resume_command if part != "--resume"]
        command.extend(["--from-node", location, "--until-node", location])
    state, owner, action, instruction = values
    return explanation(state, location, code, message, owner, action, instruction,
                       command=command, evidence=summary.get("artifacts", []),
                       observed_at=summary.get("observed_at"),
                       raw_status=status, raw_state=summary.get("state"),
                       next_node=summary.get("next_task"),
                       resume_command=resume_command,
                       **summary.get("progress_details", {}))


def adaptation_command(state, *args):
    return [sys.executable, str(REPO_ROOT / "cli/adaptation.py"),
            "--state", str(state), *args]


def _task_progress(task, tasks, events, state, now):
    location = task.task_id
    owner = f"{task.stage}_worker"
    claims = [event for event in events if event.event_type == "task_claimed"
              and event.task_id == task.task_id and event.payload.get("attempt") == task.attempt]
    worker = claims[-1].payload.get("worker_id") if claims else None
    evidence = [task.input.get("workspace", {}).get(key) for key in ("output", "logs")]
    details = {"stage": task.stage, "attempt": task.attempt, "worker": worker,
               "lease_expires": task.lease_expires, "raw_status": task.status}
    if task.stage == "diagnosis":
        bug = task.input.get("bug_report", {})
        original = bug.get("message") or str(bug.get("metadata", {}).get("validation_errors") or "")
        details["source_error"] = original

    def result(status, code, message, who, action, instruction, command=None):
        return explanation(status, location, code, message, who, action, instruction,
                           command=command, evidence=evidence, observed_at=now, **details)

    claim = adaptation_command(state, "claim", "--worker", f"{task.stage}-worker",
                               "--run-id", task.run_id, "--task-id", task.task_id,
                               "--stage", task.stage, "--limit", "1")
    claim_note = ("Start a worker for this task. Claim and expired-lease recovery are scoped to this run/task. "
                  "No worker availability registry exists; unclaimed does not prove no worker is configured.")
    if task.status == "running":
        if task.lease_expires is not None and task.lease_expires <= now:
            return result("ACTION_REQUIRED", "LEASE_EXPIRED", "The worker lease has expired.",
                          "main_agent", "RECLAIM_TASK",
                          "Claim again to recover expired work with a new attempt/token. " + claim_note, claim)
        return result("RUNNING", "WORKER_RUNNING", "A worker holds an unexpired task lease.",
                      worker or owner, "MONITOR_WORKER",
                      "Inspect logs; renew the lease during long work, then submit complete or fail. "
                      "A valid lease does not establish process liveness.")
    if task.status == "pending":
        previous = {"xpu": "torch", "integration": "xpu"}.get(task.stage)
        if previous and not any(t.operator_key == task.operator_key and t.stage == previous
                                and t.status == "succeeded" for t in tasks):
            return result("WAITING", "DEPENDENCY_PENDING", f"Waiting for {previous} evidence.",
                          "main_agent", "COMPLETE_DEPENDENCY", "Finish the previous operator stage first.")
        return result("WAITING", "DIAGNOSIS_PENDING" if task.stage == "diagnosis" else "WORKER_UNCLAIMED",
                      f"Diagnosis is queued. {details['source_error']}" if task.stage == "diagnosis"
                      else "Task is queued and unclaimed.",
                      owner, "CLAIM_TASK", claim_note, claim)
    if task.stage == "diagnosis" and task.status == "succeeded":
        source = next((t for t in tasks if t.task_id == task.input.get("source_task_id")), None)
        applied = any(event.event_type == "diagnosis_applied" and event.task_id == task.task_id
                      and event.payload.get("source_attempt") == task.input.get("source_attempt")
                      and event.payload.get("diagnosis_attempt") == task.attempt for event in events)
        action = task.output.get("next_action")
        current = source and source.status == "failed" and source.attempt == task.input.get("source_attempt")
        if not current or (applied and action == "RETRY"):
            return result("COMPLETED", "DIAGNOSIS_CONSUMED", "This diagnosis no longer needs action.",
                          "none", "NONE", "Follow the current source task attempt.")
        if action in {"RETRY", "BLOCKED"} and not applied:
            return result("ACTION_REQUIRED", "DIAGNOSIS_NOT_APPLIED",
                          f"Diagnosis completed with {action}; its conclusion has not been applied.",
                          "main_agent", "APPLY_DIAGNOSIS", "Review and apply the saved conclusion.",
                          adaptation_command(state, "apply-diagnosis", "--task-id", task.task_id))
        return result("BLOCKED", "DIAGNOSIS_BLOCKED" if action == "BLOCKED" else "EXTERNAL_REPAIR_REQUIRED",
                      str(task.output.get("repair_conclusion") or action or "Diagnosis requires review."),
                      "main_agent", "RESOLVE_BLOCKER" if action == "BLOCKED" else action or "REVIEW_DIAGNOSIS",
                      "Resolve the documented blocking condition before retrying." if action == "BLOCKED"
                      else "Perform the diagnosed external repair/rediscovery; do not relabel it RETRY to bypass work.")
    if task.status == "failed":
        error = task.output.get("error", task.output.get("bug_report", {}))
        if isinstance(error, dict):
            metadata = error.get("metadata", {})
            details["validation_errors"] = metadata.get("validation_errors", [])
            message = error.get("message") or str(metadata.get("validation_errors") or metadata.get("reason") or error)
        else:
            message = str(error)
        diagnosis = next((t for t in tasks if t.stage == "diagnosis"
                          and t.input.get("source_task_id") == task.task_id
                          and t.input.get("source_attempt") == task.attempt), None)
        details["diagnosis_task_id"] = diagnosis.task_id if diagnosis else None
        if diagnosis:
            return result("WAITING", "DIAGNOSIS_AVAILABLE" if diagnosis.status == "succeeded"
                          else "WAITING_FOR_DIAGNOSIS", message or "Task failed; diagnosis exists.",
                          "main_agent" if diagnosis.status == "succeeded" else "diagnosis_worker",
                          "FOLLOW_DIAGNOSIS", f"Follow diagnosis {diagnosis.task_id}.")
        return result("BLOCKED", "TASK_FAILED", message or "Task failed.", "main_agent",
                      "RECONCILE" if task.stage != "diagnosis" else "REVIEW_DIAGNOSIS_FAILURE",
                      "Restore the missing diagnosis edge." if task.stage != "diagnosis"
                      else "Inspect diagnosis validation errors; reconcile cannot retry a failed diagnosis.",
                      adaptation_command(state, "reconcile", "--run-id", task.run_id)
                      if task.stage != "diagnosis" else None)
    return result("COMPLETED", "TASK_COMPLETE", "Task completed; this alone is not model readiness.",
                  "none", "NONE", "Continue the run's remaining stages and final validation.")


def run_progress(store, run_id, *, graph=None, now=None):
    """Project persisted state without claiming, reconciling or validating it.

    Callers needing an atomic multi-query view should hold a read transaction.
    Graph transitions are last observations, not a worker/process heartbeat.
    """
    now = time.time() if now is None else now
    run = store.run(run_id)
    if run is None:
        raise ValueError(f"unknown run: {run_id}")
    tasks, events = store.tasks(run_id), store.events(run_id)
    task_views = [_task_progress(task, tasks, events, store.path, now) for task in tasks]
    graph = graph if graph is not None else run.metadata.get("graph_progress", {})
    resume = graph.get("resume_command")
    graph_view = graph_progress(graph, resume) if graph else None

    def result(view):
        return {"progress": view, "task_progress": task_views}

    from .execution import list_executions
    active_executions = list_executions(store, run_id, active_only=True)
    if active_executions:
        unknown = any(item.get("state") in {"STARTING", "UNKNOWN"} for item in active_executions)
        execution_view = explanation(
            "BLOCKED" if unknown else "RUNNING", "executions",
            "EXECUTION_UNCERTAIN" if unknown else "EXECUTION_ACTIVE",
            "An execution still owns its task/resource; lease expiry does not prove termination.",
            "main_agent", "INSPECT_EXECUTION",
            "Inspect execution records and logs. Reconcile known local process identity before reclaiming; "
            "never force unlock a Pod or rerun an unknown execution.",
            command=adaptation_command(store.path, "execution-status", "--run-id", run_id, "--active-only"),
            evidence=[item.get("log_path") for item in active_executions], observed_at=now,
            execution_ids=[item["execution_id"] for item in active_executions])
        busy = {item.get("task_id") for item in active_executions}
        task_views[:] = [({**execution_view, "location": view["location"]}
                         if view["location"] in busy else view) for view in task_views]
        return result(execution_view)

    validations = [item for item in run.metadata.get("managed_validations", {}).values()
                   if item.get("state") == "executing"]
    if validations:
        view = explanation(
            "BLOCKED", "validation", "VALIDATION_COORDINATOR_UNCERTAIN",
            "A validation coordinator has no final receipt; do not repeat its probes.",
            "main_agent", "RECONCILE_VALIDATION",
            "Inspect the recorded controller and child executions. Reconcile confirmed dead local "
            "controllers only after all child executions are known to have terminated.",
            command=adaptation_command(store.path, "reconcile-validation", "--run-id", run_id,
                                       "--validation-id", validations[0]["validation_id"]),
            observed_at=now, validation_ids=[item["validation_id"] for item in validations])
        busy = {item.get("identity", {}).get("task_id") for item in validations}
        task_views[:] = [({**view, "location": item["location"]}
                         if item["location"] in busy else item) for item in task_views]
        return result(view)

    handoff = active_graph_handoff(run)
    if handoff:
        pending = handoff["state"] == "pending"
        uncertain = handoff["state"] == "executing"
        return result(explanation(
            "ACTION_REQUIRED" if pending else "BLOCKED", handoff["source"]["node"],
            "GRAPH_DECISION_REQUIRED" if pending else
            "GRAPH_EXECUTION_UNCERTAIN" if uncertain else "GRAPH_RECOVERY_BLOCKED",
            "A persisted Graph failure needs a decision; the runner is not waiting in a shell."
            if pending else "An accepted execution has no completion receipt; do not execute it again."
            if uncertain else "Graph recovery is blocked or its retry budget is exhausted.",
            "main_agent", "SUBMIT_DECISION" if pending else "INSPECT_EXECUTION" if uncertain else "REVIEW_BLOCKER",
            "Read context, diagnose the bound evidence, then submit one decision with its source_version."
            if pending else "Inspect the recorded execution and receipt before requesting any further change.",
            command=adaptation_command(store.path, "context", "--run-id", run_id),
            evidence=list(handoff["source"]["source_files"]), observed_at=now,
            handoff_id=handoff["handoff_id"], source_version=handoff["source_version"],
            remaining_budget=handoff["remaining_budget"], evidence_revalidated=False,
        ))
    if (graph.get("reason_code") == "GRAPH_RECOVERY_SUCCEEDED"
            and run.status not in {"WAITING_FOR_ENVIRONMENT", "ENVIRONMENT_FAILED"}
            and all(task.status == "succeeded" for task in tasks)):
        return result(explanation(
            "READY", graph.get("node", "graph"), "GRAPH_RECOVERY_SUCCEEDED",
            graph.get("message", "Recovery passed; continue the remaining Graph gates."),
            "main_agent", "ADVANCE_GRAPH", "Advance the same run; recovery is not model delivery.",
            command=adaptation_command(store.path, "advance", "--run-id", run_id),
            evidence=graph.get("artifacts", []), observed_at=now, evidence_revalidated=False,
        ))
    if graph.get("reason_code") == "GRAPH_RECOVERY_SUCCEEDED":
        graph_view = None  # Current environment/worker/diagnosis state takes precedence.

    if run.status in {"WAITING_FOR_ENVIRONMENT", "ENVIRONMENT_FAILED"}:
        failures = [event.timestamp for event in events if event.event_type == "environment_failed"]
        if (graph_view and graph.get("observed_at", 0) > max(failures, default=0)
                and (run.status == "WAITING_FOR_ENVIRONMENT" or graph_view["state"] == "RUNNING"
                     or graph.get("reason_code") == "WAITING_FOR_DECISION")
                and graph.get("reason_code") not in {"WAITING_FOR_OPERATORS", "OPERATORS_BLOCKED"}):
            # Intake/proof may be running, or have a more specific failure.
            # Do not label active environment preparation as a new blockage.
            return result(graph_view)
        failed = run.environment.get("failed_environment_proof", {})
        proof = run.environment.get("environment_proof", {})
        return result(explanation(
            "BLOCKED", "environment", run.status,
            str(failed.get("diagnosis") or "The run has no accepted environment proof."),
            "main_agent", "PROVE_ENVIRONMENT",
            "Run/import environment proof for this run; repair the recorded Pod if one already exists.",
            evidence=[failed.get("artifact_root") or proof.get("artifact_root")],
            observed_at=now, pod=failed.get("pod") or proof.get("pod"), resume_command=resume,
            checks=failed.get("checks", {})))

    # A graph stop (e.g. missing shapes) must not be hidden by an empty queue.
    # Operator waits are recomputed from live rows so completed workers unblock
    # the explanation even before the graph is restarted.
    queue_codes = {"WAITING_FOR_OPERATORS", "OPERATORS_BLOCKED"}
    if graph_view and graph.get("reason_code") not in queue_codes:
        if graph_view["state"] != "COMPLETED":
            return result(graph_view)
        changes = [event for event in events if not event.event_type.startswith("graph_")]
        if not any(event.timestamp > graph_view["observed_at"] for event in changes):
            graph_view["details"]["evidence_revalidated"] = False
            return result(graph_view)

    active = [view for view in task_views if view["state"] != "COMPLETED"
              and view["reason_code"] not in {"WAITING_FOR_DIAGNOSIS", "DIAGNOSIS_AVAILABLE"}]
    priority = {"BLOCKED": 0, "ACTION_REQUIRED": 1, "RUNNING": 2, "WAITING": 3}
    if active:
        selected = min(active, key=lambda view: priority[view["state"]])
        return result(selected)
    if graph_view and graph.get("reason_code") == "OPERATORS_BLOCKED" and not any(
        event.task_id and event.timestamp > graph_view["observed_at"] for event in events
    ):
        # Succeeded rows can still fail evidence revalidation. Preserve that
        # rejection unless later task activity gives a reason to try again.
        return result(graph_view)
    # Missing historical stage edges are not readiness, even if every extant
    # row succeeded. Reconcile remains an explicit user/controller action.
    operators = [task for task in tasks if task.stage != "diagnosis"]
    keys = {task.operator_key for task in operators}
    missing = [f"{key}:{stage}" for key in keys for stage in ("torch", "xpu", "integration")
               if not any(t.operator_key == key and t.stage == stage and t.status == "succeeded"
                          for t in operators)]
    if missing:
        return result(explanation("BLOCKED", "operators", "MISSING_STAGE_EDGE",
                                  "Operator lifecycle stages are missing or incomplete.",
                                  "main_agent", "RECONCILE", "Reconcile historical task edges using evidence.",
                                  command=adaptation_command(store.path, "reconcile", "--run-id", run_id),
                                  observed_at=now, missing=sorted(missing)))
    continuing = bool(operators or graph)
    action = "RESUME_GRAPH" if continuing else "START_GRAPH"
    instruction = ("Resume the full graph with the same run and inputs." if continuing else
                   "Supply model_path and user_id inputs; the harness generates deployment contracts. Execute the first graph "
                   "with --execute for this run.")
    return result(explanation("READY", "graph", action,
                              "No active operator work remains; graph validation/delivery is still required."
                              if continuing else "The run is ready for graph execution/discovery.",
                              "main_agent", action, instruction,
                              command=(adaptation_command(store.path, "advance", "--run-id", run_id)
                                       if continuing and run.metadata.get("graph_execution_context", {}).get("interaction_mode") == "codex"
                                       else resume if continuing else None),
                              observed_at=now, evidence_revalidated=False))


def render_progress(progress: dict) -> str:
    action = progress["next_action"]
    lines = [f"[{progress['state']}] {progress['location']}",
             "Observed: " + datetime.fromtimestamp(progress["observed_at"], timezone.utc).isoformat(),
             f"Reason: {progress['reason_code']} — {progress['summary']}",
             f"Next ({action['owner']} / {action['action']}): {action['instruction']}"]
    if action.get("command"):
        lines.append("Command: " + shlex.join(action["command"]))
    details = progress.get("details", {})
    if details.get("response_path"):
        lines.append("Decision response: " + details["response_path"])
    if details.get("resume_command") and progress["state"] in {"BLOCKED", "ACTION_REQUIRED"}:
        lines.append("After resolving: " + shlex.join(details["resume_command"]))
    if progress["evidence"]:
        lines.append("Evidence: " + ", ".join(progress["evidence"]))
    return "\n".join(lines)
