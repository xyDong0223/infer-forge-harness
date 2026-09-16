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
        --stage torch
    python cli/adaptation.py --state state.db status --run-id deepseek-v4
"""

from __future__ import annotations

import argparse
import json
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
    contract: Path, artifact_dir: Path | None, attach_pod: str | None,
    run_id: str | None = None,
) -> list[str]:
    """The task_runner invocation behind `environment --contract`.

    Without --attach-pod this creates a new FedDeployment; a reproof must
    pass the prepared pod instead of minting a second one.
    """
    command = [
        sys.executable,
        str(REPO_ROOT / "cli" / "deployment" / "proof.py"),
        str(contract),
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

    environment = sub.add_parser(
        "environment",
        aliases=["prove-environment", "bind-environment"],
        help="run or import the deployment environment proof before discovery",
    )
    environment.add_argument("--run-id", required=True)
    source = environment.add_mutually_exclusive_group(required=True)
    source.add_argument("--status", type=Path, help="status.json from an environment-proof task")
    source.add_argument("--contract", type=Path, help="deployment task contract to execute")
    environment.add_argument(
        "--attach-pod",
        help="prove against an already-prepared Pod (Imported Context) instead of "
             "creating a new FedDeployment; required for re-proofs of an existing run",
    )
    environment.add_argument("--artifact-dir", type=Path)

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

    reconcile = sub.add_parser("reconcile", help="repair historical missing task edges using evidence")
    reconcile.add_argument("--run-id", required=True)

    claim = sub.add_parser("claim", help="claim ready work for a child Agent")
    claim.add_argument("--worker", required=True)
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

    renew = sub.add_parser("renew-lease", help="extend a live worker lease")
    renew.add_argument("--task-id", required=True)
    renew.add_argument("--worker", required=True)
    renew.add_argument("--lease-token", required=True)
    renew.add_argument("--lease-seconds", type=float, default=300.0)
    return parser


def _run(args: argparse.Namespace, scheduler: TaskScheduler) -> dict[str, Any]:
    if args.command in {"create", "create-run"}:
        metadata = _json_file(args.metadata, field="metadata")
        metadata.setdefault("environment_required", True)
        artifact_root = args.artifact_root or metadata.get("artifact_root")
        explicit_root = artifact_root is not None
        if artifact_root is None:
            artifact_root = args.state.resolve().parent / "runs" / safe_component(args.run_id)
        metadata["artifact_root"] = str(ensure_external(artifact_root))
        existing = scheduler.store.run(args.run_id)
        if existing is not None:
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
            artifact_root = args.artifact_dir or existing.metadata.get("artifact_root")
            if artifact_root is not None:
                artifact_root = ensure_external(artifact_root)
            command = _environment_command(
                args.contract, artifact_root, args.attach_pod, run_id=args.run_id,
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
                error = f"environment task failed with exit code {completed.returncode}: {proof.get('state', 'UNKNOWN')}"
                run = scheduler.record_environment_failure(args.run_id, proof, error)
                return {"command": "environment", "run": run.to_dict(), "proof": proof, "error": error}
        root = args.status.parent if args.status else proof.get("artifact_root")
        if root is None and args.artifact_dir is not None:
            root = REPO_ROOT / args.artifact_dir
        run = scheduler.bind_environment(args.run_id, proof, artifact_root=root)
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

    if args.command == "renew-lease":
        task = scheduler.renew_lease(
            args.task_id, args.worker, args.lease_token, args.lease_seconds,
        )
        return {"command": "renew-lease", "task": task.to_dict()}

    if args.command == "status":
        run = scheduler.store.run(args.run_id)
        if run is None:
            raise ValueError(f"unknown run: {args.run_id}")
        result: dict[str, Any] = {
            "command": "status",
            "run": run.to_dict(),
            "tasks": [task.to_dict() for task in scheduler.store.tasks(args.run_id)],
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
    try:
        result = _run(args, TaskScheduler(args.state))
    except (KeyError, ValueError, OSError) as exc:
        # Keep failures machine-readable as well.  The non-zero exit code
        # lets a shell/Agent distinguish a rejected transition from success.
        print(json.dumps({"error": str(exc), "command": args.command}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 6 if result.get("error") or result.get("blocked") or result.get("task", {}).get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
