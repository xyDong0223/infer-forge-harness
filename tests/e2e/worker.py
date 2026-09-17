"""External simulated Agent process; scheduling and acceptance use the real CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from core.paths import REPO_ROOT
from core.storage import ArtifactStore
from engine.contracts import OperatorSpec, OperatorTask
from engine.fake_agents import EvidenceGate, FakeAgentHarness
from engine.scheduler import TaskScheduler


def invoke(state: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli/adaptation.py"),
         "--state", str(state), *arguments],
        capture_output=True, text=True, check=False, timeout=30,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("torch", "xpu", "integration"), required=True)
    parser.add_argument("--fault", choices=("missing-evidence", "abandon"))
    parser.add_argument("--lease-seconds", default="300")
    args = parser.parse_args()
    worker = f"local-scenario-{args.stage}"
    scheduler = TaskScheduler(args.state)
    try:
        run = scheduler.store.run(args.run_id)
        if run is None or run.metadata.get("evidence_mode") != "simulation":
            raise ValueError("the local scenario worker only accepts explicit simulation runs")
        claim = invoke(args.state, "claim", "--run-id", args.run_id,
                       "--worker", worker, "--stage", args.stage,
                       "--lease-seconds", args.lease_seconds)
        if claim.returncode:
            print(claim.stdout, end="")
            return claim.returncode
        tasks = json.loads(claim.stdout)["tasks"]
        if len(tasks) != 1:
            raise ValueError(f"expected one ready {args.stage} task, got {len(tasks)}")
        task = OperatorTask.from_dict(tasks[0])
        if task.run_id != args.run_id:
            raise ValueError("scoped claim returned a task from another run")
        output = ArtifactStore(task.input["workspace"]["output"])
        if args.fault == "abandon":
            output.write_json("abandoned.json", {"task_id": task.task_id, "attempt": task.attempt})
            print(json.dumps({"abandoned": task.to_dict()}), flush=True)
            return 75

        spec = OperatorSpec.from_dict(task.input["operator_spec"])
        agents = FakeAgentHarness(scheduler, run.metadata["artifact_root"])
        if task.stage == "torch":
            result = agents._torch_agent(task, spec)
            EvidenceGate.torch(result, spec)
        else:
            reference = scheduler.store.get_task(
                f"{args.run_id}:{task.operator_key}:torch",
            ).output
            if task.stage == "xpu":
                result = agents._xpu_agent(task, spec, reference)
                EvidenceGate.xpu(result, spec)
            else:
                device = scheduler.store.get_task(
                    f"{args.run_id}:{task.operator_key}:xpu",
                ).output
                result = agents._integration_agent(task, spec, reference, device)
                EvidenceGate.integration(result, spec)
        envelope = agents._envelope(task, spec, result)
        if args.fault == "missing-evidence":
            key = {"torch": "focused_tests", "xpu": "device_test",
                   "integration": "service_regression"}[task.stage]
            Path(envelope["evidence"][key]).unlink()
        submission = output.write_json("submission.json", envelope)
        completed = invoke(
            args.state, "complete", "--task-id", task.task_id, "--worker", worker,
            f"--lease-token={task.lease_token}", "--result", str(submission),
        )
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode
    finally:
        scheduler.store.close()


if __name__ == "__main__":
    raise SystemExit(main())
