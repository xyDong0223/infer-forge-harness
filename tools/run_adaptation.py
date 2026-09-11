"""Unified command-line entry point for an adaptation run.

The command is deliberately thin: durable state transitions remain owned by
``orchestration.TaskScheduler`` while this module provides one stable entry
point that a main Agent can invoke.  Every successful command writes exactly
one JSON document to stdout so callers can safely pipe the result to another
Agent or a workflow step.

Examples::

    python tools/run_adaptation.py --state state.db create \
        --run-id deepseek-v4 --model DeepSeek-V4.1 --backend kunlun-p800
    python tools/run_adaptation.py --state state.db discover \
        --run-id deepseek-v4 --report gaps.json
    python tools/run_adaptation.py --state state.db claim --worker torch-agent \
        --stage torch
    python tools/run_adaptation.py --state state.db status --run-id deepseek-v4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Support direct execution from a checkout (``python tools/run_adaptation.py``)
# as well as ``python -m tools.run_adaptation``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestration import AdaptationRun, TaskScheduler, load_report, operator_specs_from_report  # noqa: E402


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path, required=True, help="SQLite scheduler database path"
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

    environment = sub.add_parser(
        "environment",
        aliases=["prove-environment", "bind-environment"],
        help="run or import the deployment environment proof before discovery",
    )
    environment.add_argument("--run-id", required=True)
    source = environment.add_mutually_exclusive_group(required=True)
    source.add_argument("--status", type=Path, help="status.json from an environment-proof task")
    source.add_argument("--contract", type=Path, help="deployment task contract to execute")
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

    claim = sub.add_parser("claim", help="claim ready work for a child Agent")
    claim.add_argument("--worker", required=True)
    claim.add_argument("--stage", choices=["torch", "xpu", "integration", "diagnosis"])
    claim.add_argument("--limit", type=int, default=1)
    claim.add_argument("--lease-seconds", type=float, default=300.0)

    complete = sub.add_parser("complete", help="submit a successful child Agent result")
    complete.add_argument("--task-id", required=True)
    complete.add_argument("--worker", required=True)
    complete.add_argument("--result", type=Path, required=True)

    fail = sub.add_parser("fail", help="report a task failure and create diagnosis work")
    fail.add_argument("--task-id", required=True)
    fail.add_argument("--worker", required=True)
    fail.add_argument("--error", required=True)

    diagnosis = sub.add_parser("resolve-diagnosis", help="submit a diagnosis Agent result")
    diagnosis.add_argument("--task-id", required=True)
    diagnosis.add_argument("--worker", required=True)
    diagnosis.add_argument("--result", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace, scheduler: TaskScheduler) -> dict[str, Any]:
    if args.command in {"create", "create-run"}:
        metadata = _json_file(args.metadata, field="metadata")
        metadata.setdefault("environment_required", True)
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
            command = [
                sys.executable,
                str(REPO_ROOT / "runners" / "task_runner.py"),
                str(args.contract),
                "--execute",
                "--phase",
                "environment",
            ]
            if args.artifact_dir:
                command += ["--artifact-dir", str(args.artifact_dir)]
            completed = subprocess.run(
                command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
            )
            try:
                proof = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "environment task did not return JSON status: "
                    + (completed.stderr.strip() or completed.stdout[-500:])
                ) from error
            if completed.returncode != 0:
                raise ValueError(
                    f"environment task failed with exit code {completed.returncode}: "
                    f"{proof.get('state', 'UNKNOWN')}"
                )
        run = scheduler.bind_environment(args.run_id, proof)
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
            plugin_revision=args.plugin_revision,
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
        task = scheduler.complete(args.task_id, worker_id=args.worker, result=result)
        return {"command": "complete", "task": task.to_dict()}

    if args.command == "fail":
        task = scheduler.fail(args.task_id, worker_id=args.worker, error=args.error)
        return {"command": "fail", "task": task.to_dict()}

    if args.command == "resolve-diagnosis":
        result = _json_file(args.result, field="result")
        task = scheduler.complete(args.task_id, worker_id=args.worker, result=result)
        return {"command": "resolve-diagnosis", "task": task.to_dict()}

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
