"""CLI for the persistent operator adaptation scheduler.

This command is intentionally transport-neutral: real child Agents can claim a
task, run their toolchain, then submit a JSON result without coupling the
orchestrator to a particular Agent runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestration import (  # noqa: E402
    AdaptationRun,
    TaskScheduler,
    load_report,
    operator_specs_from_report,
)


def _scheduler(path: Path) -> TaskScheduler:
    return TaskScheduler(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="SQLite scheduler database")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-run")
    create.add_argument("--run-id", required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--model-revision", default="unknown")
    create.add_argument("--plugin-revision", default="unknown")
    create.add_argument("--backend", default="unknown")

    discover = sub.add_parser("discover")
    discover.add_argument("--run-id", required=True)
    discover.add_argument("--report", type=Path, required=True)
    discover.add_argument("--model", required=True)
    discover.add_argument("--backend", required=True)
    discover.add_argument("--model-revision", default="unknown")
    discover.add_argument("--plugin-revision")

    claim = sub.add_parser("claim")
    claim.add_argument("--worker", required=True)
    claim.add_argument("--stage", choices=["torch", "xpu", "integration", "diagnosis"])
    claim.add_argument("--limit", type=int, default=1)

    complete = sub.add_parser("complete")
    complete.add_argument("--task-id", required=True)
    complete.add_argument("--worker", required=True)
    complete.add_argument("--result", type=Path, required=True)

    fail = sub.add_parser("fail")
    fail.add_argument("--task-id", required=True)
    fail.add_argument("--worker", required=True)
    fail.add_argument("--error", required=True)

    diagnosis = sub.add_parser("resolve-diagnosis")
    diagnosis.add_argument("--task-id", required=True)
    diagnosis.add_argument("--worker", required=True)
    diagnosis.add_argument("--result", type=Path, required=True)

    sub.add_parser("list")
    args = parser.parse_args()
    scheduler = _scheduler(args.state)

    if args.command == "create-run":
        result = scheduler.create_run(
            AdaptationRun(
                run_id=args.run_id, model_id=args.model, model_revision=args.model_revision,
                plugin_revision=args.plugin_revision, backend=args.backend,
                status="WAITING_FOR_ENVIRONMENT",
                metadata={"environment_required": True},
            )
        ).to_dict()
    elif args.command == "discover":
        run = scheduler.store.run(args.run_id)
        if run is None:
            parser.error(f"unknown run: {args.run_id}")
        report = load_report(args.report)
        specs = operator_specs_from_report(
            report, model_id=args.model, backend=args.backend,
            model_revision=args.model_revision, plugin_revision=args.plugin_revision,
        )
        result = {
            "run_id": args.run_id,
            "tasks": [scheduler.discover_operator(args.run_id, spec).to_dict() for spec in specs],
        }
    elif args.command == "claim":
        result = {"tasks": [task.to_dict() for task in scheduler.claim_ready(args.worker, args.stage, limit=args.limit)]}
    elif args.command == "complete":
        result = scheduler.complete(args.task_id, worker_id=args.worker, result=json.loads(args.result.read_text(encoding="utf-8"))).to_dict()
    elif args.command == "fail":
        result = scheduler.fail(args.task_id, worker_id=args.worker, error=args.error).to_dict()
    elif args.command == "resolve-diagnosis":
        result = scheduler.complete(
            args.task_id,
            worker_id=args.worker,
            result=json.loads(args.result.read_text(encoding="utf-8")),
        ).to_dict()
    else:
        result = {"tasks": [task.to_dict() for task in scheduler.store.tasks()]}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
