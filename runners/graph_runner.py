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
from tools import skill_registry  # noqa: E402
from tools import task_memory  # noqa: E402

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
    # The first node whose width is not known until an upstream artifact is read:
    # which capability dimensions are worth exercising depends on what the model
    # demands. `list` prints a JSON array, one child runs per element, and `aggregate`
    # is the fan-in — the tool owns the artifact format, the executor only walks.
    "capability_evaluation": {
        "produces": "CapabilityEvaluation",
        "needs": {"--capability-match": "fact:CapabilityMatch:capability_match.json"},
        "fan_out": {
            "var": "dimension",
            "list": ["python3", "tools/evaluate_capability.py", "--list-dimensions"],
            "aggregate": ["python3", "tools/evaluate_capability.py", "--aggregate",
                          "--out", "{artifacts}"],
        },
        "command": [
            "python3", "tools/evaluate_capability.py", "--dimension", "{dimension}",
            "--subject", "{subject}", "--pod", "{pod}", "--model-path", "{weights}",
            "--out", "{artifacts}",
        ],
        "state_file": "evaluation_status.json",
    },
    # A different question from every numerical node: whether the response has the
    # right shape. Needs a pod but not a running server, because parsers are pure text
    # transforms.
    "api_conformance": {
        "produces": "ApiConformance",
        "needs": {"--model-request": "fact:ModelRequest:model_request.yaml"},
        "command": [
            "python3", "tools/check_api_conformance.py", "--subject", "{subject}",
            "--pod", "{pod}", "--out", "{artifacts}",
        ],
        "state_file": "conformance_status.json",
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


def resolve(spec: dict, context: dict, journal: Path, environment: dict,
            command: list[str] | None = None, include_optional: bool = True) -> list[str]:
    if "contract_instance" not in context:
        # The plan renders the instance the service proof runs, so the graph can
        # supply it rather than asking an operator to copy a path.
        plan = journal_module.latest(journal, "DeploymentPlan", subject=context["subject"],
                                     environment=environment)
        if plan:
            context = {**context, "contract_instance": str(Path(plan["artifacts"]) / "kdp_instance.yaml")}
    command = [part.format(**context) for part in (command or spec["command"])]
    for flag, reference in (spec.get("needs") or {}).items():
        _, kind, filename = reference.split(":", 2)
        hit = journal_module.latest(journal, kind, subject=context["subject"], environment=environment)
        if not hit:
            raise Unresolved(
                f"{kind} for {context['subject']} is not in the Journal for this environment; "
                "run the node that produces it first"
            )
        command += [flag, str(Path(hit["artifacts"]) / filename)]
    if include_optional:
        for flag, reference in (spec.get("optional") or {}).items():
            _, kind, filename = reference.split(":", 2)
            hit = journal_module.latest(journal, kind, subject=context["subject"], environment=environment)
            if hit:
                command += [flag, str(Path(hit["artifacts"]) / filename)]
    return command


def fan_out_items(spec: dict, context: dict, journal: Path, environment: dict) -> list[str]:
    """Ask the node's own tool how wide it is.

    Run even under --plan. The list command reads a recorded artifact and touches
    nothing, and a plan that cannot say how many children will run is not a plan.
    """
    command = resolve(spec, context, journal, environment,
                      command=spec["fan_out"]["list"], include_optional=False)
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise Unresolved(
            f"the fan-out list command failed: {' '.join(command)}: {result.stderr.strip()[-300:]}"
        )
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("["):
            return json.loads(line)
    raise Unresolved(f"the fan-out list command printed no JSON array: {' '.join(command)}")


def fan_out_plan(spec: dict, context: dict, journal: Path, environment: dict,
                 artifacts: Path) -> list[tuple[Path, list[str]]]:
    """One child command per item, then the fan-in.

    Each child writes into its own subdirectory so the aggregate can point at the
    per-dimension evidence instead of summarising it away. The fan-in is a command
    the node's tool provides, not something the executor computes, so the artifact
    format stays owned by the Task.
    """
    items = fan_out_items(spec, context, journal, environment)
    if not items:
        raise Unresolved(
            "the fan-out list is empty: this model demands none of the dimensions this node "
            "evaluates, so there is nothing to exercise"
        )
    variable = spec["fan_out"]["var"]
    plan: list[tuple[Path, list[str]]] = []
    children: list[Path] = []
    for item in items:
        child = artifacts / str(item)
        children.append(child)
        child_context = {**context, variable: item, "artifacts": str(child)}
        plan.append((child, resolve(spec, child_context, journal, environment)))
    aggregate = [part.format(**{**context, "artifacts": str(artifacts)})
                 for part in spec["fan_out"]["aggregate"]]
    for child in children:
        aggregate += ["--child", str(child)]
    plan.append((artifacts, aggregate))
    return plan


def read_state(artifacts: Path, spec: dict) -> str:
    path = artifacts / spec["state_file"]
    if not path.exists():
        return "UNKNOWN"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("state", "UNKNOWN")


SUCCESS_STATES = {
    "INTAKE_READY",
    "ENVIRONMENT_READY",
    "SCAN_READY",
    "MATCH_READY",
    "CLASSIFICATION_READY",
    "EVALUATION_PASS",
    "PLAN_READY",
    "DEPLOYMENT_READY",
    "BUDGET_ACCEPTABLE",
    "CONFORMANT",
    "HANDOFF_READY",
    "PATCH_PLACED",
}


def reusable_fact(
    spec: dict, subject: str, journal: Path, environment: dict
) -> dict | None:
    """Return a prior successful fact whose artifact state is still valid."""
    kind = spec.get("produces")
    if not kind:
        return None
    hit = journal_module.latest(
        journal, kind, subject=subject, environment=environment, states=tuple(SUCCESS_STATES)
    )
    if not hit:
        return None
    artifact_dir = Path(hit["artifacts"])
    if not (artifact_dir / spec["state_file"]).exists():
        return None
    return hit


def emit_summary(summary: dict, json_output: bool) -> None:
    if json_output:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(
            f"[summary] {summary['status']} node={summary.get('node')} "
            f"next={summary.get('next_task') or '-'}"
        )


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
    parser.add_argument("--resume", action="store_true",
                        help="reuse successful Journal facts and skip completed nodes")
    parser.add_argument("--loop-state", type=Path,
                        help="Task Memory JSON path; defaults under artifact-root")
    parser.add_argument("--json", action="store_true",
                        help="emit one machine-readable summary per terminal decision")
    args = parser.parse_args()

    environment = dict(pair.split("=", 1) for pair in args.env)
    context = dict(pair.split("=", 1) for pair in args.set)
    context.update(subject=args.subject, attempt="graph",
                   environment_text=",".join(f"{k}={v}" for k, v in sorted(environment.items())))
    loop_state = args.loop_state or args.artifact_root / "task_memory.json"
    memory = task_memory.load(loop_state, args.workflow.stem, args.subject)

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
        try:
            skill = (
                skill_registry.resolve_for_context(task_type, context)
                if task_type else None
            )
        except skill_registry.SkillResolutionError:
            emit_summary(
                {"status": "NEEDS_HUMAN", "node": current, "next_task": current,
                 "reason_code": "SKILL_UNRESOLVED", "message": f"task_type={task_type}"},
                args.json,
            )
            break
        if task_type in MANUAL:
            print(f"stop: {current} is operator-driven — {MANUAL[task_type]}")
            emit_summary(
                {"status": "NEEDS_HUMAN", "node": current, "next_task": current,
                 "reason_code": "MANUAL_STEP", "skill": skill["id"],
                 "message": MANUAL[task_type]},
                args.json,
            )
            break
        spec = NODES.get(task_type or "")
        if spec is None:
            print(f"stop: no executor registered for task_type {task_type!r} ({current})")
            emit_summary(
                {"status": "NEEDS_HUMAN", "node": current, "next_task": current,
                 "reason_code": "NO_EXECUTOR", "skill": skill["id"],
                 "message": f"task_type={task_type}"},
                args.json,
            )
            break

        artifacts = args.artifact_root / current
        context["artifacts"] = str(artifacts)
        prior = reusable_fact(spec, args.subject, args.journal, environment)
        if args.resume and prior:
            next_task = node.get("on_success")
            task_memory.start_block(
                memory,
                block_id=f"{current}:reused:{len(memory['completed_loop_blocks']) + 1}",
                sub_target=current,
                exit_condition={"state_file": spec["state_file"],
                 "success_states": sorted(SUCCESS_STATES)},
                routing={"mode": "reuse_journal_fact", "skill": skill["id"],
                         "verification": skill["verification"]},
            )
            task_memory.finish_block(
                memory,
                prior["state"],
                artifacts=[prior["artifacts"]],
                next_block={"sub_target": next_task},
            )
            task_memory.save(loop_state, memory)
            emit_summary(
                {"status": "REUSED", "node": current, "next_task": next_task,
                 "reason_code": "SUCCESSFUL_FACT_REUSED",
                 "skill": skill["id"],
                 "artifacts": [prior["artifacts"]]},
                args.json,
            )
            current = next_task if next_task in by_id else None
            continue
        try:
            if "fan_out" in spec:
                commands = fan_out_plan(spec, context, args.journal, environment, artifacts)
            else:
                commands = [(artifacts, resolve(spec, context, args.journal, environment))]
        except (Unresolved, KeyError) as error:
            print(f"NEEDS_HUMAN: {current}: {error}")
            emit_summary(
                {"status": "NEEDS_HUMAN", "node": current, "next_task": current,
                 "reason_code": "INPUT_UNRESOLVED", "message": str(error)},
                args.json,
            )
            return 2

        returncode = 0
        if args.execute:
            task_memory.start_block(
                memory,
                block_id=f"{current}:{len(memory['completed_loop_blocks']) + 1}",
                sub_target=current,
                exit_condition={"state_file": spec["state_file"],
                                "success_states": sorted(SUCCESS_STATES)},
                routing={"skill": skill["id"], "verification": skill["verification"],
                         "tools": skill["tools"]},
            )
            task_memory.save(loop_state, memory)
        if not args.execute:
            for _, command in commands:
                print(f"[plan] {current}: {' '.join(command)}")
        else:
            for target, command in commands:
                print(f"[run ] {current}: {' '.join(command)}")
                target.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(command, cwd=REPO_ROOT, text=True)
                state = read_state(target, spec)
                print(f"[state] {current}: {state} (exit {result.returncode})")
                returncode = returncode or result.returncode
            journal_module.record(args.journal, spec["produces"], args.subject,
                                  read_state(artifacts, spec), artifacts, environment)
            state = read_state(artifacts, spec)
            task_memory.finish_block(
                memory,
                state,
                artifacts=[str(artifacts)],
                next_block={"sub_target": node.get("on_failure" if returncode else "on_success")},
            )
            task_memory.save(loop_state, memory)
            if returncode != 0:
                failure = node.get("on_failure", "NEEDS_HUMAN")
                print(f"[edge] {current} --failure--> {failure}")
                emit_summary(
                    {"status": "REWORK", "node": current, "next_task": failure,
                     "reason_code": "COMMAND_FAILED", "state": state,
                     "skill": skill["id"],
                     "artifacts": [str(artifacts)]},
                    args.json,
                )
                current = failure if failure in by_id else None
                continue

        if args.until_node and current == args.until_node:
            break
        nxt = node.get("on_success")
        if nxt not in by_id:
            print(f"[edge] {current} --> {nxt}")
            break
        if args.execute:
            emit_summary(
                {"status": "CONTINUE", "node": current, "next_task": nxt,
                 "reason_code": "NODE_COMPLETE", "state": read_state(artifacts, spec),
                 "skill": skill["id"],
                 "artifacts": [str(artifacts)]},
                args.json,
            )
        current = nxt
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
