"""Walk a Task Graph instead of hand-feeding one Task's output to the next.

The workflow file already declared nodes and edges; nothing read it. Every run so
far was an operator pasting the previous artifact's path into the next command,
which is exactly the step where a stale path silently answers for the wrong model.

The executor resolves each node's inputs from the Journal by `consumes`, so a node
runs against facts recorded for this subject in this environment or does not run at
all. Default is `--plan`: print the resolved commands without executing, because a
graph that cannot be inspected before it touches a cluster is not safe to trust.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import journal as journal_module  # noqa: E402

# How to invoke each node, and which recorded facts it needs. `artifacts` is the
# node's own output directory; `fact:<Kind>:<file>` resolves through the Journal.
NODES: dict[str, dict] = {
    "model_intake": {
        "produces": "ModelRequest",
        "command": [
            "python3", "tools/model_intake.py",
            "--model-id", "{subject}", "--model-path", "{model_path}",
            "--attempt-id", "{attempt}", "--out", "{artifacts}",
        ],
        "state_file": "intake_status.json",
    },
    "environment_proof": {
        "produces": "EnvironmentProof",
        "command": [
            "python3", "runners/task_runner.py", "{contract_instance}",
            "--execute", "--phase", "environment", "--artifact-dir", "{artifacts}",
        ],
        "state_file": "status.json",
    },
    "model_scan": {
        "produces": "ModelSupportCard",
        "needs": {"--model-request": "fact:ModelRequest:model_request.yaml",
                  "--env-status": "fact:EnvironmentProof:status.json"},
        "command": ["python3", "tools/scan_model_support.py", "--out", "{artifacts}"],
        "state_file": "scan_status.json",
    },
    "capability_match": {
        "produces": "CapabilityMatch",
        "needs": {"--model-request": "fact:ModelRequest:model_request.yaml",
                  "--support-card": "fact:ModelSupportCard:model_support.json",
                  "--env-status": "fact:EnvironmentProof:status.json"},
        "command": ["python3", "tools/match_capabilities.py", "--out", "{artifacts}"],
        "state_file": "match_status.json",
    },
    "gap_classification": {
        "produces": "GapClassification",
        "needs": {"--support-card": "fact:ModelSupportCard:model_support.json",
                  "--capability-match": "fact:CapabilityMatch:capability_match.json"},
        "command": ["python3", "tools/classify_gaps.py", "--out", "{artifacts}"],
        "state_file": "classification_status.json",
    },
    "deployment_plan": {
        "produces": "DeploymentPlan",
        "needs": {"--model-request": "fact:ModelRequest:model_request.yaml",
                  "--classification": "fact:GapClassification:gap_classification.json"},
        # Optional, and it matters: without the placed patch the plan reports
        # enforce_eager=False, and the support matrix then records a condition the
        # deployment did not actually run under.
        "optional": {"--placed-patch": "fact:PlacedPatch:placement_report.json"},
        "command": ["python3", "tools/plan_deployment.py", "--out", "{artifacts}"],
        "state_file": "plan_status.json",
    },
    "service_proof": {
        "produces": "DeploymentProof",
        # The plan renders the instance contract, and the pod comes from the
        # environment proof; both are passed in via --set rather than guessed.
        "command": [
            "python3", "runners/task_runner.py", "{contract_instance}",
            "--execute", "--phase", "service", "--attach-pod", "{pod}",
            "--artifact-dir", "{artifacts}",
        ],
        "state_file": "status.json",
    },
    "memory_budget": {
        "produces": "MemoryBudget",
        "needs": {},
        "command": [
            "python3", "tools/memory_budget.py", "--pod", "{pod}",
            "--server-log", "{server_log}", "--out", "{artifacts}",
        ],
        "state_file": "budget_status.json",
    },
    "accuracy_differential": {
        "produces": "AccuracyDifferential",
        "needs": {"--model-request": "fact:ModelRequest:model_request.yaml"},
        "command": [
            "python3", "tools/accuracy_differential.py", "--pod", "{pod}",
            "--served-model-name", "{served_model_name}", "--port", "{port}",
            "--out", "{artifacts}",
        ],
        "state_file": "accuracy_status.json",
    },
    "support_matrix": {
        "produces": "SupportMatrixEntry",
        "needs": {"--accuracy": "fact:AccuracyDifferential:accuracy_differential.json"},
        "optional": {
            "--deployment-status": "fact:DeploymentProof:status.json",
            "--budget-status": "fact:MemoryBudget:budget_status.json",
            "--plan": "fact:DeploymentPlan:deployment_plan.json",
        },
        "command": [
            "python3", "tools/update_support_matrix.py", "--subject", "{subject}",
            "--out", "{artifacts}",
        ],
        "state_file": "matrix_status.json",
    },
    "vendor_handoff": {
        "produces": "VendorHandoff",
        "needs": {"--triage": "fact:FailureTriage:triage_report.json"},
        "command": [
            "python3", "tools/vendor_handoff.py",
            "--environment", "{environment_text}", "--out", "{artifacts}",
        ],
        "state_file": "handoff_status.json",
    },
}

# Nodes that are deliberately not single commands. Triage needs a server rerun
# between instrumenting and reading the capture; placement needs a rerun between
# installing and validating. Pretending either is one command would make the graph
# claim work it did not do, so the walk stops here and says what to run instead.
MANUAL: dict[str, str] = {
    "failure_triage": "instrument the call, rerun the service proof, read the capture, sweep in "
    "isolation, then restore — see the contract's runs_with",
    "patch_placement": "install the patch, rerun the service proof, compare numerically, then "
    "record the placement — see the contract's runs_with",
}


class Unresolved(RuntimeError):
    """Raised when a node's inputs are not in the Journal."""


def load_workflow(path: Path) -> list[dict]:
    import yaml

    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    return workflow["spec"]["nodes"]


def node_task_type(node: dict) -> str | None:
    import yaml

    task = node.get("task")
    if not task or task == "PLANNED":
        return None
    contract = yaml.safe_load((REPO_ROOT / task).read_text(encoding="utf-8"))
    return contract["metadata"]["task_type"]


def resolve(spec: dict, context: dict, journal: Path, environment: dict) -> list[str]:
    command = [part.format(**context) for part in spec["command"]]
    for flag, reference in (spec.get("needs") or {}).items():
        _, kind, filename = reference.split(":", 2)
        hit = journal_module.latest(journal, kind, subject=context["subject"], environment=environment)
        if not hit:
            raise Unresolved(
                f"{kind} for {context['subject']} is not in the Journal for this environment; "
                "run the node that produces it first"
            )
        command += [flag, str(Path(hit["artifacts"]) / filename)]
    for flag, reference in (spec.get("optional") or {}).items():
        _, kind, filename = reference.split(":", 2)
        hit = journal_module.latest(journal, kind, subject=context["subject"], environment=environment)
        if hit:
            command += [flag, str(Path(hit["artifacts"]) / filename)]
    return command


def read_state(artifacts: Path, spec: dict) -> str:
    path = artifacts / spec["state_file"]
    if not path.exists():
        return "UNKNOWN"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("state", "UNKNOWN")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=REPO_ROOT / "workflows" / "model_adaptation.yaml")
    parser.add_argument("--subject", required=True, help="e.g. Qwen3-8B")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--journal", type=Path, default=journal_module.DEFAULT_JOURNAL)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="environment fingerprint; facts are only reused within it")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="node context, e.g. model_path=/mnt/cluster/... or pod=...")
    parser.add_argument("--from-node", help="start here instead of the entry task")
    parser.add_argument("--until-node", help="stop after this node")
    parser.add_argument("--execute", action="store_true", help="actually run; default is --plan")
    args = parser.parse_args()

    environment = dict(pair.split("=", 1) for pair in args.env)
    context = dict(pair.split("=", 1) for pair in args.set)
    context.update(subject=args.subject, attempt="graph",
                   environment_text=",".join(f"{k}={v}" for k, v in sorted(environment.items())))

    nodes = load_workflow(args.workflow)
    order = [node["id"] for node in nodes]
    start = order.index(args.from_node) if args.from_node else 0
    by_id = {node["id"]: node for node in nodes}

    current = order[start]
    while current:
        node = by_id.get(current)
        if node is None or node.get("task") in (None, "PLANNED"):
            print(f"stop: {current} has no contract yet")
            break
        task_type = node_task_type(node)
        if task_type in MANUAL:
            print(f"stop: {current} is operator-driven — {MANUAL[task_type]}")
            break
        spec = NODES.get(task_type or "")
        if spec is None:
            print(f"stop: no executor registered for task_type {task_type!r} ({current})")
            break

        artifacts = args.artifact_root / current
        context["artifacts"] = str(artifacts)
        try:
            command = resolve(spec, context, args.journal, environment)
        except (Unresolved, KeyError) as error:
            print(f"NEEDS_HUMAN: {current}: {error}")
            return 2

        printable = " ".join(command)
        if not args.execute:
            print(f"[plan] {current}: {printable}")
        else:
            print(f"[run ] {current}: {printable}")
            artifacts.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, cwd=REPO_ROOT, text=True)
            state = read_state(artifacts, spec)
            print(f"[state] {current}: {state} (exit {result.returncode})")
            journal_module.record(args.journal, spec["produces"], args.subject, state,
                                 artifacts, environment)
            if result.returncode != 0:
                failure = node.get("on_failure", "NEEDS_HUMAN")
                print(f"[edge] {current} --failure--> {failure}")
                current = failure if failure in by_id else None
                continue

        if args.until_node and current == args.until_node:
            break
        nxt = node.get("on_success")
        if nxt not in by_id:
            print(f"[edge] {current} --> {nxt}")
            break
        current = nxt
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
