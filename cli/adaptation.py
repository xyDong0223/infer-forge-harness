"""Unified command-line entry point for an adaptation run.

The command is deliberately thin: durable state transitions remain owned by
``engine.TaskScheduler`` while this module provides one stable entry
point that a main Agent can invoke.  Every successful command writes exactly
one JSON document to stdout so callers can safely pipe the result to another
Agent or a workflow step.

Examples::

    python cli/adaptation.py --state state.db create \
        --run-id deepseek-v4 --model DeepSeek-V4.1 --backend kunlun-p800
    python cli/adaptation.py --state state.db discover \
        --run-id deepseek-v4 --report gaps.json
    python cli/adaptation.py --state state.db claim --worker torch-agent \
        --run-id deepseek-v4 --stage torch
    python cli/adaptation.py --state state.db context --run-id deepseek-v4
    python cli/adaptation.py --state state.db status --run-id deepseek-v4
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

# Support direct execution from a checkout (``python cli/adaptation.py``)
# as well as ``python -m cli.adaptation``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import AdaptationRun, TaskScheduler, load_report, operator_specs_from_report  # noqa: E402
from core.storage import RunPaths, default_state_root, ensure_external, safe_component  # noqa: E402
from engine.scheduler import EventStore  # noqa: E402
from engine.progress import graph_progress, render_progress, run_progress  # noqa: E402
from engine.context import run_context  # noqa: E402


def _json_file(path: Path | None, *, field: str) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {field} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} JSON must contain an object")
    return value


def _environment_command(
    contract: Path | None, artifact_dir: Path | None, attach_pod: str | None,
    run_id: str | None = None, user_id: str | None = None,
    evidence_mode: str | None = None, health_interval_seconds: float | None = None,
) -> list[str]:
    """The task_runner invocation behind `environment --contract`.

    The caller restores the Pod from durable run state on a reproof.
    """
    command = [
        sys.executable,
        str(REPO_ROOT / "cli" / "deployment" / "proof.py"),
        *([str(contract)] if contract is not None else []),
        "--execute",
        "--phase",
        "environment",
    ]
    if artifact_dir:
        command += ["--artifact-dir", str(artifact_dir)]
    if attach_pod:
        command += ["--attach-pod", attach_pod]
    if run_id:
        command += ["--run-id", run_id]
    if user_id is not None:
        command += ["--user-id", user_id]
    if evidence_mode:
        command += ["--evidence-mode", evidence_mode]
    if health_interval_seconds is not None:
        command += ["--health-interval-seconds", str(health_interval_seconds)]
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path, default=default_state_root() / "state.sqlite",
        help="External SQLite scheduler database (defaults under the state root)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser(
        "create", aliases=["create-run"], help="create or reopen an adaptation run"
    )
    create.add_argument("--run-id", required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--backend", required=True)
    create.add_argument("--model-revision", default="unknown")
    create.add_argument("--plugin-revision", default="unknown")
    create.add_argument("--environment", type=Path)
    create.add_argument("--metadata", type=Path)
    create.add_argument("--artifact-root", type=Path, help="External run directory")
    create.add_argument("--worker-protocol", choices=("managed-v2",),
                        help="Require frozen candidates and runner-owned validation receipts")

    environment = sub.add_parser(
        "environment",
        aliases=["prove-environment", "bind-environment"],
        help="run or import the deployment environment proof before discovery",
    )
    environment.add_argument("--run-id", required=True)
    source = environment.add_mutually_exclusive_group()
    source.add_argument("--status", type=Path, help="status.json from an environment-proof task")
    source.add_argument("--contract", type=Path, help="external generated proof to replay; omit to generate from the harness profile")
    environment.add_argument("--health-interval-seconds", type=float)
    environment.add_argument(
        "--attach-pod",
        help="prove against an already-prepared Pod (Imported Context) instead of "
             "creating a new FedDeployment; required for re-proofs of an existing run",
    )
    environment.add_argument("--artifact-dir", type=Path)
    environment.add_argument("--user-id", help="Resource owner's user ID, supplied by the user; reused on retry")

    discover = sub.add_parser("discover", help="convert a gap report into torch tasks")
    discover.add_argument("--run-id", required=True)
    discover.add_argument("--report", type=Path, required=True)
    discover.add_argument("--model")
    discover.add_argument("--backend")
    discover.add_argument("--model-revision")
    discover.add_argument("--plugin-revision")
    discover.add_argument("--include-waived", action="store_true")

    status = sub.add_parser("status", help="show durable run and task status")
    status.add_argument("--run-id", required=True)
    status.add_argument("--events", action="store_true", help="include append-only events")
    status.add_argument("--format", choices=("json", "text"), default="json",
                        help="JSON (default) or a readable location/reason/next-action summary")

    context = sub.add_parser("context", help="read Agent context without claiming, recovering, or validating")
    context.add_argument("--run-id", required=True)
    context.add_argument("--task-id", help="include one task's input, upstream results, and acceptance requirements")

    advance = sub.add_parser("advance", help="resume recorded Graph inputs until a worker, decision, input or delivery boundary")
    advance.add_argument("--run-id", required=True)
    advance.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="explicit context additions/corrections; cannot bypass a pending decision")

    submit = sub.add_parser("submit-decision", help="accept and execute one evidence-backed Graph recovery decision")
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--handoff-id", required=True)
    submit.add_argument("--decision-id", required=True, help="stable client idempotency key for this logical decision")
    submit.add_argument("--expected-version", required=True, help="source_version returned by context")
    submit.add_argument("--decision", type=Path, required=True, help="external JSON with existing Decision fields")

    reconcile = sub.add_parser("reconcile", help="repair historical missing task edges using evidence")
    reconcile.add_argument("--run-id", required=True)

    claim = sub.add_parser("claim", help="claim ready work for a child Agent")
    claim.add_argument("--worker", required=True)
    claim.add_argument("--run-id", help="limit selection AND expired-lease recovery to this run")
    claim.add_argument("--task-id", help="claim exactly this task; requires --run-id")
    claim.add_argument("--stage", choices=["torch", "xpu", "integration", "diagnosis"])
    claim.add_argument("--limit", type=int, default=1)
    claim.add_argument("--lease-seconds", type=float, default=300.0)

    complete = sub.add_parser("complete", help="submit a successful child Agent result")
    complete.add_argument("--task-id", required=True)
    complete.add_argument("--worker", required=True)
    complete.add_argument("--lease-token", required=True)
    complete.add_argument("--result", type=Path, required=True)

    fail = sub.add_parser("fail", help="report a task failure and create diagnosis work")
    fail.add_argument("--task-id", required=True)
    fail.add_argument("--worker", required=True)
    fail.add_argument("--lease-token", required=True)
    fail.add_argument("--error", required=True)

    diagnosis = sub.add_parser("resolve-diagnosis", help="submit a diagnosis Agent result")
    diagnosis.add_argument("--task-id", required=True)
    diagnosis.add_argument("--worker", required=True)
    diagnosis.add_argument("--lease-token", required=True)
    diagnosis.add_argument("--result", type=Path, required=True)

    apply_diagnosis = sub.add_parser(
        "apply-diagnosis", help="apply a completed RETRY or BLOCKED diagnosis conclusion"
    )
    apply_diagnosis.add_argument("--task-id", required=True)

    renew = sub.add_parser("renew-lease", help="extend a live worker lease")
    renew.add_argument("--task-id", required=True)
    renew.add_argument("--worker", required=True)
    renew.add_argument("--lease-token", required=True)
    renew.add_argument("--lease-seconds", type=float, default=300.0)

    freeze = sub.add_parser("freeze-candidate", help="bind a producer's immutable attempt-local candidate")
    validate = sub.add_parser("validate-worker", help="execute independent measurements for a frozen candidate")
    execute = sub.add_parser("execute-worker", help="supervise one local task process with durable identity")
    for command in (freeze, validate, execute):
        command.add_argument("--task-id", required=True)
        command.add_argument("--worker", required=True, help="current lease-owning controller")
        command.add_argument("--lease-token", required=True)
    freeze.add_argument("--producer", required=True)
    freeze.add_argument("--candidate-root", type=Path, required=True)
    freeze.add_argument("--base-revision", required=True)
    validate.add_argument("--validator", required=True)
    validate.add_argument("--candidate-id", required=True)
    validate.add_argument("--validation-id", required=True)
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--timeout", type=float, default=300)
    execute.add_argument("--execution-id", required=True)
    execute.add_argument("--cwd", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--resource", type=Path, help="JSON stable cluster/namespace/pod_uid identity")
    execute.add_argument("--timeout", type=float, default=300)
    execute.add_argument("argv", nargs=argparse.REMAINDER)
    executions = sub.add_parser("execution-status", help="read durable execution records without recovery")
    executions.add_argument("--run-id", required=True)
    executions.add_argument("--active-only", action="store_true")
    observe = sub.add_parser("reconcile-execution", help="observe a local process group; never force unlock")
    observe.add_argument("--run-id", required=True)
    observe.add_argument("--execution-id", required=True)
    validation_recovery = sub.add_parser("reconcile-validation", help="close a dead local validation coordinator without replay")
    validation_recovery.add_argument("--run-id", required=True)
    validation_recovery.add_argument("--validation-id", required=True)
    return parser


def _run(args: argparse.Namespace, scheduler: TaskScheduler) -> dict[str, Any]:
    if args.command == "advance":
        from runners.codex_interaction import advance_run
        return {"command": "advance", **advance_run(scheduler, args.run_id, settings=args.set)}

    if args.command == "submit-decision":
        from runners.codex_interaction import submit_graph_decision
        decision = _json_file(ensure_external(args.decision), field="decision")
        return {"command": "submit-decision", **submit_graph_decision(
            scheduler, args.run_id, args.handoff_id, args.decision_id,
            args.expected_version, decision,
        )}

    if args.command in {"create", "create-run"}:
        metadata = _json_file(args.metadata, field="metadata")
        if args.worker_protocol:
            metadata["worker_protocol"] = args.worker_protocol
        metadata.setdefault("environment_required", True)
        artifact_root = args.artifact_root or metadata.get("artifact_root")
        explicit_root = artifact_root is not None
        if artifact_root is None:
            artifact_root = args.state.resolve().parent / "runs" / safe_component(args.run_id)
        metadata["artifact_root"] = str(ensure_external(artifact_root))
        existing = scheduler.store.run(args.run_id)
        if existing is not None:
            if args.worker_protocol and existing.metadata.get("worker_protocol") != args.worker_protocol:
                raise ValueError("cannot retroactively change an existing run's worker protocol")
            saved_root = existing.metadata.get("artifact_root")
            if explicit_root and saved_root and (
                ensure_external(saved_root) != ensure_external(artifact_root)
            ):
                raise ValueError("existing run has a different artifact_root; resume its recorded directory")
            return {"command": "create", "run": existing.to_dict()}
        RunPaths(metadata["artifact_root"], args.run_id).initialize()
        run = scheduler.create_run(
            AdaptationRun(
                run_id=args.run_id,
                model_id=args.model,
                backend=args.backend,
                model_revision=args.model_revision,
                plugin_revision=args.plugin_revision,
                environment=_json_file(args.environment, field="environment"),
                metadata=metadata,
                status="WAITING_FOR_ENVIRONMENT",
            )
        )
        return {"command": "create", "run": run.to_dict()}

    if args.command in {"environment", "prove-environment", "bind-environment"}:
        if args.status:
            proof = _json_file(args.status, field="environment status")
        else:
            existing = scheduler.store.run(args.run_id)
            if existing is None:
                raise ValueError(f"unknown run: {args.run_id}")
            from runners.managed_boundary import require_supported_runtime
            require_supported_runtime(existing)
            artifact_root = args.artifact_dir or existing.metadata.get("artifact_root")
            if artifact_root is not None:
                artifact_root = ensure_external(artifact_root)
            # A later successful bind supersedes an older failed Pod handoff.
            handoff = (
                existing.environment.get("failed_environment_proof", {})
                if getattr(existing, "status", None) == "ENVIRONMENT_FAILED"
                else existing.environment.get("environment_proof", {})
            )
            command = _environment_command(
                args.contract, artifact_root,
                args.attach_pod or handoff.get("pod")
                or existing.environment.get("environment_proof", {}).get("pod"),
                run_id=args.run_id,
                user_id=(args.user_id if args.user_id is not None else
                         handoff.get("user_id") or
                         existing.environment.get("environment_proof", {}).get("user_id")),
                evidence_mode=existing.metadata.get("evidence_mode"),
                health_interval_seconds=args.health_interval_seconds,
            )
            completed = subprocess.run(
                command, cwd=REPO_ROOT, text=True, capture_output=True, check=False,
            )
            try:
                proof = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "environment task did not return JSON status: "
                    + (completed.stderr.strip() or completed.stdout[-500:])
                ) from error
            if completed.returncode != 0:
                outcome = proof.get("state") or proof.get("status", "UNKNOWN")
                error = f"environment task failed with exit code {completed.returncode}: {outcome}"
                if proof.get("message"):
                    error += ": " + proof["message"]
                run = scheduler.record_environment_failure(args.run_id, proof, error)
                return {"command": "environment", "run": run.to_dict(), "proof": proof, "error": error}
        root = args.status.parent if args.status else proof.get("artifact_root")
        if root is None and args.artifact_dir is not None:
            root = REPO_ROOT / args.artifact_dir
        try:
            run = scheduler.bind_environment(args.run_id, proof, artifact_root=root)
        except ValueError as error:
            run = scheduler.record_environment_failure(args.run_id, proof, str(error))
            return {"command": "environment", "run": run.to_dict(),
                    "proof": proof, "error": str(error)}
        return {"command": "environment", "run": run.to_dict(), "proof": proof}

    if args.command == "discover":
        run = scheduler.store.run(args.run_id)
        if run is None:
            raise ValueError(f"unknown run: {args.run_id}")
        if run.metadata.get("environment_required") and run.status != "ENVIRONMENT_READY":
            raise ValueError(
                "environment proof is required before discovery; run the environment command "
                "with the deployment task status first"
            )
        report = load_report(args.report)
        proof_context = run.environment.get("environment_proof", {})
        specs = operator_specs_from_report(
            report,
            model_id=args.model or run.model_id,
            backend=args.backend or run.backend,
            model_revision=args.model_revision or run.model_revision,
            plugin_revision=args.plugin_revision or run.plugin_revision,
            environment=proof_context,
            include_waived=args.include_waived,
        )
        tasks = [scheduler.discover_operator(args.run_id, spec) for spec in specs]
        return {
            "command": "discover",
            "run_id": args.run_id,
            "specs": [spec.to_dict() | {"operator_key": spec.operator_key} for spec in specs],
            "tasks": [task.to_dict() for task in tasks],
        }

    if args.command == "claim":
        tasks = scheduler.claim_ready(
            args.worker,
            stage=args.stage,
            lease_seconds=args.lease_seconds,
            limit=args.limit,
            run_id=args.run_id,
            task_id=args.task_id,
        )
        return {"command": "claim", "worker": args.worker, "tasks": [task.to_dict() for task in tasks]}

    if args.command == "complete":
        result = _json_file(args.result, field="result")
        task = scheduler.complete(args.task_id, worker_id=args.worker, result=result,
                                  lease_token=args.lease_token)
        return {"command": "complete", "task": task.to_dict()}

    if args.command == "fail":
        task = scheduler.fail(args.task_id, worker_id=args.worker, error=args.error,
                              lease_token=args.lease_token)
        return {"command": "fail", "task": task.to_dict()}

    if args.command == "resolve-diagnosis":
        result = _json_file(args.result, field="result")
        task = scheduler.complete(args.task_id, worker_id=args.worker, result=result,
                                  lease_token=args.lease_token)
        return {"command": "resolve-diagnosis", "task": task.to_dict()}

    if args.command == "apply-diagnosis":
        task = scheduler.apply_diagnosis(args.task_id)
        return {"command": "apply-diagnosis", "task": task.to_dict()}

    if args.command == "renew-lease":
        task = scheduler.renew_lease(
            args.task_id, args.worker, args.lease_token, args.lease_seconds,
        )
        return {"command": "renew-lease", "task": task.to_dict()}

    if args.command == "freeze-candidate":
        from engine.managed_validation import freeze_candidate
        return {"command": args.command, "candidate": freeze_candidate(
            scheduler, args.task_id, args.worker, args.lease_token, args.producer,
            args.candidate_root, args.base_revision)}

    if args.command == "validate-worker":
        from runners.worker_validation import validate_worker
        return {"command": args.command, **validate_worker(
            scheduler, args.task_id, args.worker, args.lease_token, args.validator,
            args.candidate_id, args.validation_id,
            _json_file(ensure_external(args.evidence), field="evidence"), timeout=args.timeout)}

    if args.command == "execute-worker":
        from runners.managed_execution import execute_managed
        task = scheduler.store.get_task(args.task_id)
        if task is None:
            raise ValueError(f"unknown task: {args.task_id}")
        command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        record = execute_managed(
            scheduler, task.run_id, args.execution_id, command, cwd=args.cwd,
            output_dir=args.output_dir, task_id=args.task_id, worker=args.worker,
            lease_token=args.lease_token,
            resource=None if args.resource is None else _json_file(args.resource, field="resource"),
            timeout=args.timeout)
        return {"command": args.command, "execution": record,
                "blocked": record["state"] != "SUCCEEDED"}

    if args.command == "execution-status":
        from engine.execution import list_executions
        if scheduler.store.run(args.run_id) is None:
            raise ValueError(f"unknown run: {args.run_id}")
        return {"command": args.command, "run_id": args.run_id,
                "executions": list_executions(scheduler.store, args.run_id, active_only=args.active_only)}

    if args.command == "reconcile-execution":
        from runners.managed_execution import reconcile_local_execution
        record = reconcile_local_execution(scheduler, args.run_id, args.execution_id)
        return {"command": args.command, "execution": record,
                "blocked": record["state"] not in {"SUCCEEDED", "FAILED"}}

    if args.command == "reconcile-validation":
        from runners.validation_recovery import reconcile_local_validation
        receipt = reconcile_local_validation(scheduler, args.run_id, args.validation_id)
        return {"command": args.command, "receipt": receipt,
                "blocked": receipt["state"] == "executing"}

    if args.command == "context":
        return {"command": "context", **run_context(scheduler.store, args.run_id, args.task_id)}

    if args.command == "status":
        run = scheduler.store.run(args.run_id)
        if run is None:
            raise ValueError(f"unknown run: {args.run_id}")
        result: dict[str, Any] = {
            "command": "status",
            "run": run.to_dict(),
            "tasks": [task.to_dict() for task in scheduler.store.tasks(args.run_id)],
            **run_progress(scheduler.store, args.run_id),
        }
        if args.events:
            result["events"] = [event.to_dict() for event in scheduler.store.events(args.run_id)]
        return result

    if args.command == "reconcile":
        return {"command": "reconcile", "run_id": args.run_id, **scheduler.reconcile(args.run_id)}

    raise ValueError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    store = None
    try:
        # Observations never create/migrate a DB, allocate attempts, or recover leases.
        readonly = args.command in {"status", "context", "execution-status"}
        if args.command in {"advance", "submit-decision", "reconcile-execution", "reconcile-validation",
                            "freeze-candidate", "validate-worker", "execute-worker"}:
            probe = EventStore(args.state, readonly=True)
            try:
                if getattr(args, "task_id", None):
                    if probe.get_task(args.task_id) is None:
                        raise ValueError(f"unknown task: {args.task_id}")
                elif probe.run(args.run_id) is None:
                    raise ValueError(f"unknown run: {args.run_id}")
            finally:
                probe.close()
        store = EventStore(args.state, readonly=readonly)
        if readonly:
            store.db.execute("BEGIN")
        result = _run(args, TaskScheduler(store))
        run_id = result.get("run", {}).get("run_id") or result.get("task", {}).get("run_id")
        if run_id and "progress" not in result:
            result.update(run_progress(store, run_id))
    except (KeyError, ValueError, OSError, sqlite3.Error) as exc:
        # Keep failures machine-readable as well.  The non-zero exit code
        # lets a shell/Agent distinguish a rejected transition from success.
        failure = {"error": str(exc), "command": args.command}
        failure["progress"] = graph_progress({
            "status": "BLOCKED", "command": args.command,
            "reason_code": "COMMAND_REJECTED", "message": str(exc),
        })
        print(render_progress(failure["progress"]) if getattr(args, "format", None) == "text"
              else json.dumps(failure, ensure_ascii=False))
        return 2
    finally:
        if store is not None:
            store.close()
    if getattr(args, "format", None) == "text":
        print(render_progress(result["progress"]))
        for progress in result.get("task_progress", []):
            if progress["state"] != "COMPLETED" and progress != result["progress"]:
                print("\n" + render_progress(progress))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.command in {"advance", "submit-decision"}:
        return result["exit_code"]
    return 6 if result.get("error") or result.get("blocked") or result.get("task", {}).get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
