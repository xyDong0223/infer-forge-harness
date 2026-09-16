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
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners import evidence  # noqa: E402
from runners import watch as watch_module  # noqa: E402
from tools import journal as journal_module  # noqa: E402
from tools import skill_registry  # noqa: E402
from tools import task_memory  # noqa: E402
from core.facade import resolve_adapters  # noqa: E402
from core.storage import ArtifactStore, RunPaths, WritePolicyError, ensure_external  # noqa: E402
from core.target import (  # noqa: E402
    bind_subject, canonical_hardware, contract_target, load_target, require_supported,
    target_environment,
)
from runners.task_runner import load_yaml  # noqa: E402
from validators.deployment_validator import (  # noqa: E402
    ENVIRONMENT_ARTIFACTS, validate_deployment_status, validate_environment_status,
)

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
    "runtime_drift_scan": {
        "produces": "RuntimeDriftReport",
        "needs": {"--env-status": "fact:EnvironmentProof:status.json"},
        "command": ["python3", "tools/scan_runtime_drift.py", "--out", "{artifacts}"],
        "state_file": "drift_status.json",
    },
    "toy_bringup": {
        "produces": "ToyBringupReport",
        "needs": {"--model-request": "fact:ModelRequest:model_request.yaml",
                  "--env-status": "fact:EnvironmentProof:status.json"},
        "command": ["python3", "tools/toy_bringup.py", "--out", "{artifacts}"],
        "state_file": "bringup_status.json",
    },
    "torch_shim_handoff": {
        "produces": "TorchShimRegistry",
        "needs": {"--env-status": "fact:EnvironmentProof:status.json"},
        "command": ["python3", "tools/scan_torch_shims.py", "--out", "{artifacts}"],
        "state_file": "shim_status.json",
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
    "operator_task_dispatch": {
        "produces": "OperatorTaskDispatch",
        "needs": {"--gaps": "fact:GapClassification:gap_classification.json"},
        "command": [
            "python3", "tools/operator_lifecycle.py", "dispatch",
            "--subject", "{subject}", "--out", "{artifacts}",
        ],
        "state_file": "dispatch_status.json",
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
    "baseline_freeze": {
        "produces": "ServingBaseline",
        "needs": {
            "--service": "fact:DeploymentProof:status.json",
            "--accuracy": "fact:AccuracyDifferential:accuracy_differential.json",
        },
        "command": [
            "python3", "tools/operator_lifecycle.py", "freeze-baseline",
            "--subject", "{subject}", "--out", "{artifacts}",
        ],
        "state_file": "baseline_status.json",
    },
    "operator_candidate_integration": {
        "produces": "OperatorIntegration",
        "needs": {"--baseline": "fact:ServingBaseline:baseline_manifest.json"},
        "command": [
            "python3", "tools/operator_lifecycle.py", "integrate",
            "--subject", "{subject}", "--out", "{artifacts}",
        ],
        "state_file": "integration_status.json",
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
    # mat-006 as one orchestrated sequence (instrument -> rerun service ->
    # capture -> isolate -> restore -> verdict). It used to stop the walk as a
    # MANUAL step, which meant every kernel failure on a new model waited for a
    # person; the sequence was always mechanical, only unowned.
    "failure_triage": {
        "produces": "FailureTriage",
        "needs": {"--env-status": "fact:EnvironmentProof:status.json"},
        "command": [
            "python3", "runners/triage_executor.py",
            "--pod", "{pod}", "--contract-instance", "{contract_instance}",
            "--out", "{artifacts}",
        ],
        "state_file": "status.json",
    },
    # mat-007 likewise: apply the reversible fallback, validate in the server,
    # compare numerically, record the placement — or reject and remove it.
    "patch_placement": {
        "produces": "PlacedPatch",
        "needs": {"--triage": "fact:FailureTriage:triage_report.json"},
        "command": [
            "python3", "runners/patch_executor.py",
            "--pod", "{pod}", "--contract-instance", "{contract_instance}",
            "--out", "{artifacts}",
        ],
        "state_file": "status.json",
    },
    # The three correctness gates, sequenced by runners/correctness_executor.py.
    # Each exercises the real path against an independent reference with a
    # discriminating control; a control that cannot fail makes the grade
    # AMBIGUOUS, never a pass.
    "platform_kernel_correctness": {
        "produces": "PlatformKernelCorrectness",
        "command": [
            "python3", "runners/correctness_executor.py", "kernel",
            "--pod", "{pod}", "--out", "{artifacts}",
        ],
        "state_file": "kernel_status.json",
    },
    "end_to_end_accuracy": {
        "produces": "EndToEndAccuracy",
        "needs": {"--model-request": "fact:ModelRequest:model_request.yaml"},
        "command": [
            "python3", "runners/correctness_executor.py", "end-to-end",
            "--pod", "{pod}", "--served-model-name", "{served_model_name}",
            "--port", "{port}", "--out", "{artifacts}",
        ],
        "state_file": "accuracy_status.json",
    },
    "long_context_sparse_correctness": {
        "produces": "LongContextSparseCorrectness",
        "command": [
            "python3", "runners/correctness_executor.py", "long-context",
            "--pod", "{pod}", "--out", "{artifacts}",
        ],
        "state_file": "long_context_status.json",
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

# Every task type in the workflow now has a sequenced executor. The MANUAL
# mechanism stays because it is the honest answer for a future node whose work
# genuinely cannot be sequenced: the walk stops and says what to run instead of
# pretending a command did work it did not do.
MANUAL: dict[str, str] = {}


class Unresolved(RuntimeError):
    """Raised when a node's inputs are not in the Journal."""


def load_workflow(path: Path) -> list[dict]:
    import yaml

    try:
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        nodes = workflow["spec"]["nodes"]
    except (yaml.YAMLError, KeyError, TypeError) as error:
        raise ValueError("workflow must declare spec.nodes") from error
    if (not isinstance(nodes, list) or not nodes or not all(
        isinstance(node, dict) and isinstance(node.get("id"), str) and node["id"].strip()
        for node in nodes
    ) or len({node["id"] for node in nodes}) != len(nodes)):
        raise ValueError("workflow nodes must have unique nonempty identities")
    return nodes


def node_task_type(node: dict) -> str | None:
    import yaml

    task = node.get("task")
    if not task or task == "PLANNED":
        return None
    contract = yaml.safe_load((REPO_ROOT / task).read_text(encoding="utf-8"))
    return contract["metadata"]["task_type"]


def resolve(spec: dict, context: dict, journal: Path, environment: dict,
            command: list[str] | None = None, include_optional: bool = True) -> list[str]:
    if spec.get("needs") and not environment:
        raise Unresolved("a nonempty environment is required to reuse Journal inputs; "
                         "provide --target/--env and a current environment proof")
    if "contract_instance" not in context:
        # The plan renders the instance the service proof runs, so the graph can
        # supply it rather than asking an operator to copy a path.
        plan = input_fact(journal, "DeploymentPlan", context["subject"], environment)
        if plan:
            context = {**context, "contract_instance": str(Path(plan["artifacts"]) / "kdp_instance.yaml")}
    command = [part.format(**context) for part in (command or spec["command"])]
    for flag, reference in (spec.get("needs") or {}).items():
        _, kind, filename = reference.split(":", 2)
        hit = input_fact(journal, kind, context["subject"], environment)
        if not hit:
            raise Unresolved(
                f"{kind} for {context['subject']} is not in the Journal for this environment; "
                "run the node that produces it first"
            )
        command += [flag, str(Path(hit["artifacts"]) / filename)]
    if include_optional:
        for flag, reference in (spec.get("optional") or {}).items():
            _, kind, filename = reference.split(":", 2)
            hit = input_fact(journal, kind, context["subject"], environment)
            if hit:
                command += [flag, str(Path(hit["artifacts"]) / filename)]
    requested = (bind_subject(load_target(context["target_file"]), context["subject"])
                 if context.get("target_file") else None)
    if "runners/task_runner.py" in command:
        path = command[command.index("runners/task_runner.py") + 1]
        require_supported(contract_target(load_yaml(Path(path)), requested))
        if requested is not None:
            command += ["--target", context["target_file"], "--subject", context["subject"]]
    elif "--contract-instance" in command:
        path = command[command.index("--contract-instance") + 1]
        require_supported(contract_target(load_yaml(Path(path)), requested))
    return command


def fan_out_items(spec: dict, context: dict, journal: Path, environment: dict) -> list[str]:
    """Ask the node's own tool how wide it is.

    Run even under --plan. The list command reads a recorded artifact and touches
    nothing, and a plan that cannot say how many children will run is not a plan.
    """
    command = resolve(spec, context, journal, environment,
                      command=spec["fan_out"]["list"], include_optional=False)
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if result.returncode != 0:
        raise Unresolved(
            f"the fan-out list command failed: {' '.join(command)}: {result.stderr.strip()[-300:]}"
        )
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("["):
            items = json.loads(line)
            if (not isinstance(items, list) or not all(
                isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item)
                for item in items
            ) or len(set(items)) != len(items)):
                raise WritePolicyError("fan-out items must be unique safe directory names")
            return items
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


def read_status(artifacts: Path, spec: dict) -> dict:
    path = artifacts / spec["state_file"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_state(artifacts: Path, spec: dict) -> str:
    state = read_status(artifacts, spec).get("state")
    return state if isinstance(state, str) else "UNKNOWN"


def node_passed(artifacts: Path, spec: dict, returncode: int) -> bool:
    payload = read_status(artifacts, spec)
    validator = payload.get("validator")
    return (
        returncode == 0 and read_state(artifacts, spec) in SUCCESS_STATES
        and (validator is None or isinstance(validator, dict)
             and validator.get("passed") is True and not validator.get("errors"))
    )


def allocate_commands(paths: RunPaths, node: str, artifacts: Path,
                      commands: list[tuple[Path, list[str]]]):
    """Bind a read-only plan to a fresh workspace only when it will execute."""
    attempt = paths.allocate_attempt(node)
    resolved = [
        (attempt.output / target.relative_to(artifacts),
         [part.replace(str(artifacts), str(attempt.output)).replace(
             "{attempt-id}", attempt.identity["attempt_id"]) for part in command])
        for target, command in commands
    ]
    (attempt.input / "commands.json").write_text(
        json.dumps([command for _, command in resolved], indent=2), encoding="utf-8",
    )
    return attempt, resolved


def command_logs(attempt, target: Path) -> Path:
    relative = target.relative_to(attempt.output)
    # The aggregate and every child have distinct log directories.
    return attempt.logs / ("aggregate" if relative == Path(".") else "children") / relative


def validate_run_identity(paths: RunPaths) -> None:
    """Check existing ownership without turning a plan into a write."""
    marker = paths.root / "run.json"
    if marker.is_symlink():
        raise WritePolicyError("run identity cannot be a symlink")
    if not marker.exists():
        return
    try:
        identity = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise WritePolicyError(f"cannot read run identity: {marker}") from error
    if (not isinstance(identity, dict) or identity.get("schema_version") != 1
            or not isinstance(identity.get("run_id"), str) or not identity["run_id"].strip()
            or paths.run_id is not None and identity["run_id"] != paths.run_id):
        raise WritePolicyError(f"run identity conflicts with {marker}")


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
    "TRIAGE_READY",
    "PATCH_PLACED",
    "KERNEL_PASS",
    "ACCURACY_PASS",
    "LONG_CONTEXT_PASS",
    "DISPATCHED",
    "DISPATCH_SKIPPED",
    "BASELINE_FROZEN",
    "WAITING_FOR_CANDIDATE",
    "READY_FOR_INTEGRATION",
}


def reusable_fact(
    spec: dict, subject: str, journal: Path, environment: dict
) -> dict | None:
    """Return a prior successful fact whose artifact state is still valid."""
    kind = spec.get("produces")
    if not kind or not environment:
        return None
    hit = journal_module.latest(
        journal, kind, subject=subject, environment=fact_environment(kind, environment)
    )
    if not hit or hit.get("state") not in SUCCESS_STATES:
        return None
    artifact_dir = Path(hit["artifacts"])
    payload = read_status(artifact_dir, spec)
    if payload.get("state") != hit["state"]:
        return None
    validator = payload.get("validator")
    if validator is not None and (
        not isinstance(validator, dict) or validator.get("passed") is not True
        or validator.get("errors")
    ):
        return None
    detail = hit.get("detail") or {}
    if detail.get("returncode", 0) != 0:
        return None
    if detail.get("status_sha256") and detail["status_sha256"] != file_digest(
        artifact_dir / spec["state_file"]
    ):
        return None
    if kind in ("EnvironmentProof", "DeploymentProof"):
        validate = validate_environment_status if kind == "EnvironmentProof" else validate_deployment_status
        try:
            if validate(payload):
                return None
            if any(not (artifact_dir / name).exists() for name in payload["artifacts"]):
                return None
            if kind == "EnvironmentProof" and any(
                not (artifact_dir / name).is_file() for name in ENVIRONMENT_ARTIFACTS
            ):
                return None
        except (TypeError, AttributeError):
            return None
    if kind == "EnvironmentProof":
        proven = proof_fingerprint(artifact_dir)
        if not proven or (detail.get("environment_fingerprint")
                          and proven != detail["environment_fingerprint"]):
            return None
        if environment.get("environment_fingerprint") not in (None, proven):
            return None
    return hit


def fact_environment(kind: str, environment: dict) -> dict:
    # Intake is model identity, and the proof establishes the runtime scope.
    # Neither depends on a fingerprint that is only produced by that proof.
    if kind in ("ModelRequest", "EnvironmentProof"):
        return {key: value for key, value in environment.items()
                if key not in ("environment_fingerprint", "environment_pod")}
    return environment


def input_fact(journal: Path, kind: str, subject: str, environment: dict) -> dict | None:
    spec = next((spec for spec in NODES.values() if spec.get("produces") == kind), None)
    if spec is None:
        return None
    return reusable_fact(spec, subject, journal, environment)


def file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def proof_fingerprint(artifacts: Path) -> str | None:
    path = artifacts / "environment_fingerprint.txt"
    try:
        if not path.read_text(encoding="utf-8").strip():
            return None
    except (OSError, ValueError):
        return None
    return file_digest(path)


def bind_proven_environment(context: dict, environment: dict, journal: Path) -> None:
    proof = reusable_fact(NODES["environment_proof"], context["subject"], journal,
                          fact_environment("EnvironmentProof", environment))
    if not proof:
        return
    artifacts = Path(proof["artifacts"])
    status = read_status(artifacts, NODES["environment_proof"])
    proven = proof_fingerprint(artifacts)
    if environment.get("environment_fingerprint") not in (None, proven):
        raise Unresolved("current environment proof fingerprint does not match --env")
    if (context.get("pod") not in (None, status["pod"])
            and context.get("pod") != context.get("_environment_pod")):
        raise Unresolved("pod does not match the current environment proof")
    environment.update(environment_fingerprint=proven, environment_pod=status["pod"])
    context["pod"] = status["pod"]
    context["_environment_pod"] = status["pod"]


def record_fact(journal: Path, spec: dict, subject: str, artifacts: Path,
                environment: dict, returncode: int = 0) -> dict:
    detail = {"status_sha256": file_digest(artifacts / spec["state_file"]),
              "returncode": returncode}
    if spec["produces"] == "EnvironmentProof":
        detail["environment_fingerprint"] = proof_fingerprint(artifacts)
    return journal_module.record(
        journal, spec["produces"], subject, read_state(artifacts, spec), artifacts,
        fact_environment(spec["produces"], environment), extra=detail,
    )


def node_watch(args, node: str, artifacts: Path):
    """The node's heartbeat journal, or None when disabled.

    A long node whose console prints nothing until it finishes (a weight
    load behind task_runner is exactly that shape) leaves no durable trace
    that the walk is alive — run glm52-int-w8a8-p800-001's four "any
    progress?" interruptions had no on-disk answer. The journal beats
    regardless of console silence.
    """
    if not getattr(args, "watch_interval", 0):
        return None
    return watch_module.LogWatch(
        node, artifacts / "node_console.log",
        artifacts / "watch_journal.jsonl",
        interval=args.watch_interval,
    ).start()


def emit_summary(summary: dict, json_output: bool) -> None:
    if json_output:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(
            f"[summary] {summary['status']} node={summary.get('node')} "
            f"next={summary.get('next_task') or '-'}"
        )
        if summary.get("message"):
            print(summary["message"])


def _failure_reason(artifacts: Path, spec: dict, state: str) -> str:
    """Read what the node itself said, before falling back to the bare state."""
    payload = read_status(artifacts, spec)
    for key in ("reason", "reason_code", "validation_errors"):
        if payload.get(key):
            return str(payload[key])
    return f"node ended in state {state}"


def attempt_recovery(args, *, node: str, spec: dict, context: dict, artifacts: Path,
                     environment: dict, state: str) -> object:
    """Engage the brain on a failed node; None-handling is the caller's.

    The controller owns the loop; this function only supplies the two things
    it cannot know: how this node reruns (its command, rebuilt with the
    decision's parameter overrides) and where the run's artifacts live.
    """
    from engine.brain import DecisionRequest, FailureEvidence, brain_from_config
    from engine.recovery import RecoveryController, default_actions

    paths = getattr(args, "run_paths", None) or RunPaths(args.artifact_root)
    recovery_attempt = paths.allocate_attempt(f"{node}:recovery")
    run_dir = recovery_attempt.output
    failure_evidence = FailureEvidence(
        node=node, state=state,
        reason=_failure_reason(artifacts, spec, state),
        artifacts=[str(artifacts)], environment=dict(environment),
    )
    request = DecisionRequest(
        model=args.subject,
        backend=environment.get("hardware", "p800"),
        failure=failure_evidence, attempts_remaining=args.recovery_budget,
        context={k: str(v) for k, v in context.items()},
    )
    brain = brain_from_config(
        {"brain": args.brain, "decide_command": args.decide_command,
         "decide_timeout": args.decide_timeout},
        run_dir,
    )

    final_artifacts = ""

    def rerun(decision) -> tuple[bool, str]:
        nonlocal final_artifacts
        protected = {"artifacts", "attempt", "subject", "target_file", "contract_instance",
                     "_environment_pod", "environment_text", "run_id", "journal", "loop_state"}
        if any(key not in context or key in protected for key in decision.params):
            return False, "BLOCKED: retry cannot override managed paths or undeclared context"
        planned = paths.root / "{attempt-output}"
        merged = {**context, **{str(key): str(value)
                                for key, value in decision.params.items()},
                  "artifacts": str(planned), "attempt": "{attempt-id}"}
        try:
            if "fan_out" in spec:
                commands = fan_out_plan(spec, merged, args.journal, environment, planned)
            else:
                commands = [(planned, resolve(spec, merged, args.journal, environment))]
            attempt, commands = allocate_commands(paths, node, planned, commands)
        except WritePolicyError:
            raise
        except (Unresolved, KeyError, ValueError, OSError) as error:
            return False, f"UNRESOLVED: {error}"
        returncode = 0
        for target, command in commands:
            print(f"[rerun ] {node}: {' '.join(command)}")
            target.mkdir(parents=True, exist_ok=True)
            # Same crash-first contract as the main walk: a recovery rerun
            # writes the live node_console.log, and a dead child is snapshotted
            # before the next attempt can replace it.
            logs = command_logs(attempt, target)
            watch = node_watch(args, node, logs)
            try:
                result = evidence.run_logged(
                    command, cwd=REPO_ROOT, log_path=logs / "node_console.log",
                    crash_tag="recovery", watch=watch,
                )
            except (OSError, ValueError):
                ArtifactStore(attempt.root).register(identity=attempt.identity, outcome="BLOCKED")
                raise
            finally:
                if watch is not None:
                    watch.stop("rerun finished")
            returncode = returncode or result.returncode
        new_state = read_state(attempt.output, spec)
        passed = node_passed(attempt.output, spec, returncode)
        ArtifactStore(attempt.root).register(
            identity=attempt.identity, outcome=new_state,
            required=(f"output/{spec['state_file']}",) if passed else (),
        )
        record_fact(args.journal, spec, args.subject, attempt.output, environment, returncode)
        failure_evidence.artifacts.append(str(attempt.output))
        if passed:
            final_artifacts = str(attempt.output)
        return passed, new_state

    actions = default_actions(
        REPO_ROOT, {"subject": args.subject, "pod": context.get("pod", "")}, run_dir
    )
    controller = RecoveryController(brain, rerun, actions, budget=args.recovery_budget)
    outcome = controller.recover(request)
    outcome.final_artifacts = final_artifacts
    outcome.recovery_artifacts = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "recovery_outcome.json").write_text(
        json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ArtifactStore(recovery_attempt.root).register(
        identity=recovery_attempt.identity, outcome=outcome.status,
        required=("output/recovery_outcome.json",),
    )
    return outcome


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=REPO_ROOT / "workflows" / "model_adaptation.yaml")
    parser.add_argument("--subject", required=True, help="e.g. Qwen3-8B")
    parser.add_argument(
        "--target",
        type=Path,
        help="platform target YAML; applies the compatibility gate before planning",
    )
    parser.add_argument("--artifact-root", type=Path,
                        help="external run root override; otherwise --run-id is required")
    parser.add_argument("--run-id", help="explicit durable run identity")
    parser.add_argument("--journal", type=Path)
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
    parser.add_argument("--auto-recover", action="store_true",
                        help="on a failed node, consult the brain before following the "
                             "failure edge: decide -> act -> rerun, bounded by --recovery-budget")
    parser.add_argument("--recovery-budget", type=int, default=3,
                        help="repair attempts per failed node before the failure edge applies")
    parser.add_argument("--brain", choices=("rule", "agent"), default="agent",
                        help="decision source: 'agent' delegates to an external decider "
                             "(LLM) through decision_request/decision files; 'rule' is the "
                             "deterministic safety net")
    parser.add_argument("--decide-command", default=None,
                        help="decider command for --brain agent; receives the request and "
                             "response paths. Without it the runner waits for decision.json "
                             "to appear next to the request.")
    parser.add_argument("--decide-timeout", type=float, default=600.0,
                        help="seconds to wait for one decision in agent mode")
    parser.add_argument("--watch-interval", type=float, default=30.0,
                        help="heartbeat seconds for the node watch journal; "
                             "0 disables the watch (a silent long node is "
                             "expected — the journal is what keeps it "
                             "observable)")
    args = parser.parse_args()

    try:
        if args.artifact_root is None and not args.run_id:
            raise WritePolicyError("provide --run-id or an external --artifact-root")
        args.run_paths = (RunPaths(args.artifact_root, args.run_id) if args.artifact_root
                          else RunPaths.for_run(args.run_id))
        validate_run_identity(args.run_paths)
        args.artifact_root = args.run_paths.root
        args.journal = ensure_external(args.journal or args.run_paths.journal)
        args.loop_state = ensure_external(args.loop_state or args.run_paths.memory)
        for path in (args.journal, args.loop_state):
            if path.exists() and not path.is_file():
                raise WritePolicyError(f"coordination state must be a file: {path}")
            if path == args.artifact_root / "run.json":
                raise WritePolicyError("coordination state cannot overwrite run identity")
        if args.journal == args.loop_state:
            raise WritePolicyError("Journal and Task Memory must have distinct paths")
        environment = dict(pair.split("=", 1) for pair in args.env)
        context = dict(pair.split("=", 1) for pair in args.set)
        if any(key in context for key in ("artifacts", "attempt", "run_id", "journal",
                                          "loop_state", "subject")):
            raise WritePolicyError("--set cannot override managed identity or write paths")
        if args.recovery_budget < 1:
            raise ValueError("recovery budget must be at least 1")
    except (ValueError, OSError) as error:
        emit_summary({"status": "BLOCKED", "reason_code": "WRITE_POLICY",
                      "message": str(error)}, args.json)
        return 2
    try:
        requested_target = load_target(args.target) if args.target else None
        if requested_target is not None:
            resolve_adapters(requested_target, require_supported=True)
            requested_target = bind_subject(requested_target, args.subject)
            target_env = target_environment(requested_target)
            for key, value in target_env.items():
                actual = environment.get(key)
                if key == "hardware" and actual is not None:
                    actual = canonical_hardware(actual)
                if key in environment and actual != value:
                    raise ValueError(f"--env {key} conflicts with --target")
            environment.update(target_env, compatibility_status="supported")
            context["target_file"] = str(args.target.resolve())
        if context.get("contract_instance"):
            require_supported(contract_target(
                load_yaml(Path(context["contract_instance"])), requested_target
            ))
    except (ValueError, OSError) as error:
        print(f"blocked: {error}")
        emit_summary({"status": "BLOCKED", "reason_code": "TARGET_MISMATCH",
                      "message": str(error)}, args.json)
        return 2
    context.update(subject=args.subject, attempt="{attempt-id}",
                   environment_text=",".join(f"{k}={v}" for k, v in sorted(environment.items())))
    try:
        bind_proven_environment(context, environment, args.journal)
    except Unresolved as error:
        emit_summary({"status": "BLOCKED", "reason_code": "ENVIRONMENT_MISMATCH",
                      "message": str(error)}, args.json)
        return 2
    context["environment_text"] = ",".join(f"{k}={v}" for k, v in sorted(environment.items()))
    loop_state = args.loop_state
    memory = task_memory.load(loop_state, args.workflow.stem, args.subject)

    nodes = load_workflow(args.workflow)
    order = [node["id"] for node in nodes]
    start = order.index(args.from_node) if args.from_node else 0
    by_id = {node["id"]: node for node in nodes}

    current = order[start]
    visits: dict[str, int] = {}
    while current:
        node = by_id.get(current)
        if node is None or node.get("task") in (None, "PLANNED"):
            print(f"stop: {current} has no contract yet")
            break
        # Failure edges can declare a cycle (mat-006 --failure--> mat-020
        # --failure--> mat-006) that is legitimate once — a second upstream
        # failure may re-enter triage — but never four times: without this
        # guard the walk looped those two nodes forever while both failed
        # (run glm52-int-w8a8-p800-001, 2026-09-14, dozens of iterations).
        visits[current] = visits.get(current, 0) + 1
        if visits[current] > 3:
            print(
                f"stop: {current} re-entered {visits[current]} times in one walk "
                "(failure-edge cycle with no progress)"
            )
            emit_summary(
                {"status": "NEEDS_HUMAN", "node": current, "next_task": current,
                 "reason_code": "FAILURE_EDGE_CYCLE",
                 "message": f"{current} keeps failing and re-entering; the failure "
                 "edges form a cycle — a human decision is required"},
                args.json,
            )
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

        artifacts = args.artifact_root / "{attempt-output}"
        context.update(artifacts=str(artifacts), attempt="{attempt-id}")
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
            if args.execute:
                task_memory.save(loop_state, memory)
            emit_summary(
                {"status": "REUSED", "node": current, "next_task": next_task,
                 "reason_code": "SUCCESSFUL_FACT_REUSED",
                 "skill": skill["id"],
                 "artifacts": [prior["artifacts"]]},
                args.json,
            )
            if args.until_node and current == args.until_node:
                break
            current = next_task if next_task in by_id else None
            continue
        try:
            if "fan_out" in spec:
                commands = fan_out_plan(spec, context, args.journal, environment, artifacts)
            else:
                commands = [(artifacts, resolve(spec, context, args.journal, environment))]
        except WritePolicyError:
            raise
        except (Unresolved, KeyError, ValueError, OSError) as error:
            print(f"NEEDS_HUMAN: {current}: {error}")
            emit_summary(
                {"status": "NEEDS_HUMAN", "node": current, "next_task": current,
                 "reason_code": "INPUT_UNRESOLVED", "message": str(error)},
                args.json,
            )
            return 2

        returncode = 0
        crash_logs: list[str] = []
        if args.execute:
            attempt, commands = allocate_commands(
                args.run_paths, current, artifacts, commands,
            )
            artifacts = attempt.output
            context.update(artifacts=str(artifacts), attempt=attempt.identity["attempt_id"])
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
                # A node's console output is evidence, not noise: it is teed
                # live to the terminal (a long bring-up must show progress) and
                # to node_console.log, and a non-zero exit is snapshotted to a
                # non-overwritable crash/<node>.crash.log before the failure
                # edge reruns anything into this directory. Before this, a
                # crashing node's traceback scrolled past unrecorded (run
                # glm52-int-w8a8-p800-001, 2026-09-14).
                logs = command_logs(attempt, target)
                watch = node_watch(args, current, logs)
                try:
                    result = evidence.run_logged(
                        command, cwd=REPO_ROOT, log_path=logs / "node_console.log",
                        crash_tag="node", watch=watch,
                    )
                except (OSError, ValueError):
                    ArtifactStore(attempt.root).register(
                        identity=attempt.identity, outcome="BLOCKED",
                    )
                    raise
                finally:
                    if watch is not None:
                        watch.stop("node finished")
                state = read_state(target, spec)
                print(f"[state] {current}: {state} (exit {result.returncode})")
                if result.crash_log:
                    print(f"[crash] {current}: evidence snapshotted to {result.crash_log}")
                    crash_logs.append(str(result.crash_log))
                returncode = returncode or result.returncode
            state = read_state(artifacts, spec)
            passed = node_passed(artifacts, spec, returncode)
            ArtifactStore(attempt.root).register(
                identity=attempt.identity, outcome=state,
                required=(f"output/{spec['state_file']}",) if passed else (),
            )
            record_fact(args.journal, spec, args.subject, artifacts, environment, returncode)
            task_memory.finish_block(
                memory,
                state,
                artifacts=[str(artifacts)],
                next_block={"sub_target": node.get("on_success" if passed else "on_failure")},
            )
            task_memory.save(loop_state, memory)
            # Exit code 0 is not success either: the node's own state file
            # is the contract, and a state outside SUCCESS_STATES riding the
            # success edge would skip triage for a failure that already
            # happened — the failure edge must be selected by EITHER signal.
            state_mismatch = returncode == 0 and not passed
            if returncode != 0 or state_mismatch:
                reason_code = "STATE_NOT_SUCCESS" if state_mismatch else "COMMAND_FAILED"
                recovered = False
                if args.auto_recover:
                    outcome = attempt_recovery(
                        args, node=current, spec=spec, context=context,
                        artifacts=artifacts, environment=environment, state=state,
                    )
                    if outcome.status == "RECOVERED":
                        recovered = True
                        artifacts = Path(outcome.final_artifacts)
                        context["artifacts"] = str(artifacts)
                        state = outcome.final_state or read_state(artifacts, spec)
                        record_fact(args.journal, spec, args.subject, artifacts, environment)
                        task_memory.start_block(
                            memory,
                            block_id=f"{current}:recovered:{len(memory['completed_loop_blocks']) + 1}",
                            sub_target=current,
                            exit_condition={"success_states": sorted(SUCCESS_STATES)},
                            routing={"mode": "recovery", "skill": skill["id"]},
                        )
                        task_memory.finish_block(
                            memory, state, artifacts=[str(artifacts)],
                            next_block={"sub_target": node.get("on_success")},
                        )
                        task_memory.save(loop_state, memory)
                        emit_summary(
                            {"status": "RECOVERED", "node": current,
                             "next_task": node.get("on_success"),
                             "reason_code": "AUTO_RECOVERY",
                             "state": state, "skill": skill["id"],
                             "artifacts": [str(artifacts)],
                             "recovery": outcome.recovery_artifacts},
                            args.json,
                        )
                if not recovered:
                    task_memory.record_observed_issue(
                        memory,
                        issue="command_failure",
                        evidence=[str(artifacts)],
                        environment=environment,
                        source=current,
                    )
                    task_memory.save(loop_state, memory)
                    failure = node.get("on_failure", "NEEDS_HUMAN")
                    print(f"[edge] {current} --failure--> {failure}")
                    emit_summary(
                        {"status": "REWORK", "node": current, "next_task": failure,
                         "reason_code": reason_code, "state": state,
                         "skill": skill["id"],
                         "artifacts": [str(artifacts)] + crash_logs},
                        args.json,
                    )
                    current = failure if failure in by_id else None
                    continue
            if spec["produces"] == "EnvironmentProof":
                # A new proof replaces any prior runtime scope, never copies its
                # fingerprint onto facts produced in this different environment.
                environment.pop("environment_fingerprint", None)
                environment.pop("environment_pod", None)
                try:
                    bind_proven_environment(context, environment, args.journal)
                except Unresolved as error:
                    emit_summary({"status": "BLOCKED", "reason_code": "ENVIRONMENT_MISMATCH",
                                  "message": str(error)}, args.json)
                    return 2
                context["environment_text"] = ",".join(
                    f"{key}={value}" for key, value in sorted(environment.items())
                )

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


def main() -> int:
    try:
        return _main()
    except WritePolicyError as error:
        emit_summary({"status": "BLOCKED", "reason_code": "WRITE_POLICY",
                      "message": str(error)}, "--json" in sys.argv)
        return 2
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        emit_summary({"status": "BLOCKED", "reason_code": "INVALID_INPUT",
                      "message": str(error)}, "--json" in sys.argv)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
