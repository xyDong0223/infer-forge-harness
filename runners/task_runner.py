"""Plan-first task runner for infer-forge-harness."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import REPO_ROOT
from core.user_identity import resolve_user_id
from operations.deployment.environment_contract import build_environment_contract
from validators.contract_validator import (
    find_placeholders,
    validate_task_contract,
    validate_executable,
)
from validators.deployment_validator import (
    validate_deployment_status,
    validate_environment_status,
    validate_environment_identity,
)
from core.contracts import TargetContext
from core.target import bind_subject, contract_target, load_target, require_supported
from core.storage import (
    ArtifactStore, RunPaths, WritePolicyError, default_state_root,
    ensure_external, locate_attempt, safe_component,
)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to run tasks") from exc
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def proof_phase(contract: dict[str, Any], phase: str) -> str:
    return {"environment_proof": "environment", "service_proof": "service"}.get(
        contract.get("metadata", {}).get("task_type"), phase,
    )


def build_plan(
    contract: dict[str, Any], target: TargetContext | None = None,
    phase: str = "environment",
) -> dict[str, Any]:
    resolved = contract_target(contract, target)
    require_supported(resolved)
    phase = proof_phase(contract, phase)
    actions = contract.get("actions", [])
    phase_task = {"environment": "kdp-001a-environment-proof",
                  "service": "kdp-001b-service-proof"}.get(phase)
    if phase_task:
        actions = load_yaml(REPO_ROOT / "tasks" / phase_task / "task.yaml")["actions"]
    return {
        "task_id": contract.get("metadata", {}).get("name"),
        "task_type": {"environment": "environment_proof", "service": "service_proof"}.get(
            phase, contract.get("metadata", {}).get("task_type"),
        ),
        "mode": "PLAN_ONLY",
        "phase": phase,
        "actions": actions,
        "requires": {
            "explicit_execute": True,
            "cluster_access": True,
            "environment_adapter": f"adapters/{resolved.hardware.replace('/', '_')}",
        },
    }


def execute(
    contract: dict[str, Any],
    contract_path: Path | None,
    artifact_dir: Path | None,
    attach_pod: str | None = None,
    phase: str = "environment",
    target: TargetContext | None = None,
    run_id: str | None = None,
    user_id: str | None = None,
) -> int:
    """Run the task against the real cluster and let a Validator decide."""
    task_id = contract["metadata"]["name"]
    try:
        requested = artifact_dir or (contract.get("artifacts") or {}).get("directory")
        root = ensure_external(
            requested if requested is not None
            else default_state_root() / "runs" / safe_component(run_id or task_id)
        )
        attempt = locate_attempt(root)
        if attempt is not None:
            if root != attempt.output and attempt.output not in root.parents:
                raise WritePolicyError("task output must reside inside its attempt output/")
            if run_id is not None and run_id != attempt.identity["run_id"]:
                raise WritePolicyError("requested run_id does not own the attempt")
            if (attempt.root / "manifest.json").exists():
                raise WritePolicyError("attempt already has a formal result; allocate a fresh attempt")
            target_dir = root
        else:
            attempt = RunPaths(root, run_id).initialize().allocate_attempt(task_id)
            target_dir = attempt.output
        store = ArtifactStore(target_dir)
        if store.path("status.json").exists() or store.path("manifest.json").exists():
            raise WritePolicyError("task output already contains a formal result; allocate a fresh attempt")
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, indent=2))
        return 2

    inventory = ArtifactStore(attempt.root)

    def finish(status: dict[str, Any], code: int) -> int:
        status.setdefault("task_id", task_id)
        status.setdefault("state", status.get("status", "BLOCKED"))
        status.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        if contract.get("execution", {}).get("user_id"):
            status["user_id"] = contract["execution"]["user_id"]
        status["evidence_mode"] = contract.get("metadata", {}).get("evidence_mode", "real")
        status["artifact_root"] = str(target_dir)
        status["manifest_path"] = str(attempt.root / "manifest.json")
        status["workspace_identity"] = attempt.identity
        required = ["status.json", *status.get("artifacts", [])]
        try:
            declared_paths = [store.path(name) for name in required]
            required_paths = [
                path.relative_to(attempt.root).as_posix()
                for path in declared_paths if not path.is_dir()
            ]
            store.write_json("status.json", status, overwrite=True)
            outcome = status.get("state", status.get("status", "BLOCKED"))
            if status.get("validator", {}).get("passed") is False:
                outcome = "REWORK"
            inventory.register(
                identity=attempt.identity, outcome=outcome,
                required=required_paths,
            )
        except (ValueError, OSError) as error:
            code = 2
            if status.get("message") or status.get("reason"):
                status["execution_error"] = status.get("message") or status["reason"]
            status.update(state="BLOCKED", status="BLOCKED", message=str(error))
            status["validator"] = {"passed": False, "errors": [str(error)]}
            try:
                inventory.path("manifest.json").unlink(missing_ok=True)
                store.write_json("status.json", status, overwrite=True)
                inventory.register(
                    identity=attempt.identity, outcome="BLOCKED",
                    required=[store.path("status.json").relative_to(attempt.root).as_posix()],
                )
            except (ValueError, OSError) as publication_error:
                status["manifest_error"] = str(publication_error)
                status["manifest_path"] = None
                try:
                    store.write_json("status.json", status, overwrite=True)
                except (ValueError, OSError) as status_error:
                    status["status_error"] = str(status_error)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return code

    try:
        owner = resolve_user_id(user_id, contract.get("execution", {}).get("user_id"))
        recorded_owner = contract.get("execution", {}).get("user_id")
        if (proof_phase(contract, phase) != "environment" and recorded_owner
                and owner != recorded_owner):
            raise ValueError("user_id does not match the generated service contract owner; "
                             "use the recorded owner or generate a plan for a new run")
        contract.setdefault("execution", {})["user_id"] = owner
        ArtifactStore(attempt.input).write_json("requested_task_contract.json", {
            "requested_phase": phase,
            "effective_phase": proof_phase(contract, phase),
            "contract": contract,
        })
        return _execute(contract, target_dir, attach_pod, phase, target, finish)
    except (ValueError, OSError, RuntimeError) as error:
        return finish({"status": "BLOCKED", "message": str(error), "task_id": task_id}, 2)


def _execute(contract, target_dir, attach_pod, phase, target, finish) -> int:
    from adapters import ClusterConfig
    from core.facade import resolve_adapters

    from runners.deployment_proof import DeploymentProofRunner

    task_type = contract["metadata"]["task_type"]
    # environment_proof and service_proof are the two halves of a deployment
    # proof: the same executor, stopped at a different exit criterion.
    phase = proof_phase(contract, phase)
    if task_type not in ("deployment_proof", "environment_proof", "service_proof"):
        return finish({"status": "EXECUTION_NOT_CONFIGURED", "message": "no executor for this task_type"}, 4)

    try:
        target_context = contract_target(contract, target)
        bundle = resolve_adapters(target_context, require_supported=True)
    except ValueError as error:
        return finish({"status": "BLOCKED", "message": str(error)}, 2)

    owner = resolve_user_id(recorded=contract.get("execution", {}).get("user_id"))
    if not owner:
        return finish({
            "state": "INPUT_REQUIRED", "status": "INPUT_REQUIRED",
            "paths": ["$.execution.user_id"],
            "message": "Ask the user for their user ID and pass --user-id <USER_ID>. "
                       "Do not infer it from the host login or an example resource name.",
        }, 3)
    if phase == "environment":
        generated = build_environment_contract(
            owner, previous=contract,
        )
        contract.clear()
        contract.update(generated)
        contract["artifacts"]["directory"] = str(target_dir)
    phase_task = {"environment": "kdp-001a-environment-proof",
                  "service": "kdp-001b-service-proof"}.get(phase)
    if phase_task:
        # Normalize the phase contract while preserving the target launch
        # settings imported from the deployment plan for service execution.
        phase_contract = load_yaml(REPO_ROOT / "tasks" / phase_task / "task.yaml")
        for field in ("actions", "acceptance", "exit_states"):
            contract[field] = phase_contract[field]
        contract["artifacts"] = {
            "directory": str(target_dir),
            "collect": list(dict.fromkeys([
                *phase_contract["artifacts"], "task_contract", "reproduce_command",
            ])),
        }
    contract.setdefault("context", {})["resolved_target"] = {
        "model": target.model if target else target_context.model,
        "hardware": target_context.hardware,
        "engine": target_context.engine,
        "backend": target_context.backend,
        "plugin": target_context.plugin,
        "revisions": target_context.revisions,
        "compatibility": bundle.compatibility,
    }
    contract["metadata"]["task_type"] = {
        "environment": "environment_proof", "service": "service_proof",
    }.get(phase, task_type)
    attempt = locate_attempt(target_dir)
    if attempt is not None:
        # Keep the request for diagnosis, but replay only the effective contract
        # that is also passed to the runner and saved in output/task_contract.yaml.
        ArtifactStore(attempt.input).write_json("task_contract.json", contract)
    errors = validate_executable(contract)
    if errors:
        return finish({"status": "CONTRACT_INVALID", "errors": errors}, 2)
    config = ClusterConfig.load(user_id=contract.get("execution", {}).get("user_id"))
    if contract["execution"]["namespace"] != config.namespace:
        return finish({
            "status": "NEEDS_HUMAN",
            "message": "contract namespace does not match the harness config",
        }, 5)
    runner = DeploymentProofRunner(
        contract=contract,
        adapter=bundle.hardware(config),
        repo_root=REPO_ROOT,
        artifact_dir=target_dir,
        attach_pod=attach_pod,
        phase=phase,
        runtime=bundle.runtime,
    )
    status = runner.run()
    gate = (
        validate_environment_status(status) + validate_environment_identity(target_dir)
        if phase == "environment"
        else validate_deployment_status(status)
    )
    status["validator"] = {"passed": not gate, "errors": gate}
    status["target"] = contract["context"]["resolved_target"]
    return finish(status, 0 if not gate else 6)


def run(args) -> int:

    try:
        target = load_target(args.target) if args.target else None
        if args.subject:
            if target is None:
                raise ValueError("--subject requires --target")
            target = bind_subject(target, args.subject)
        if target is not None:
            require_supported(target)
        if args.contract is None:
            if args.phase != "environment":
                raise ValueError("service/all requires an external contract generated by MAT-005")
            contract = build_environment_contract(
                getattr(args, "user_id", None),
                evidence_mode=getattr(args, "evidence_mode", None),
                health_interval_seconds=getattr(args, "health_interval_seconds", None),
            )
        else:
            contract = load_yaml(args.contract)
        if getattr(args, "evidence_mode", None) == "simulation":
            contract.setdefault("metadata", {})["evidence_mode"] = "simulation"
        errors = validate_task_contract(contract)
        if errors:
            print(json.dumps({"status": "CONTRACT_INVALID", "errors": errors}, indent=2))
            return 2
        require_supported(contract_target(contract, target))
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, indent=2))
        return 2
    if args.server_log:
        contract.setdefault("execution", {})["server_log"] = args.server_log
    interval = getattr(args, "health_interval_seconds", None)
    if interval is not None:
        if interval < 0:
            print(json.dumps({"status": "CONTRACT_INVALID", "message": "health interval must be nonnegative"}))
            return 2
        contract.setdefault("execution", {})["health_interval_seconds"] = interval
    phase = proof_phase(contract, args.phase)
    environment_phase = phase == "environment"
    placeholders = [] if args.execute and environment_phase else find_placeholders(contract)
    if placeholders:
        print(json.dumps({"status": "INPUT_REQUIRED", "paths": placeholders}, indent=2))
        return 3
    if args.execute:
        return execute(contract, args.contract, args.artifact_dir, args.attach_pod, args.phase,
                       target, args.run_id, getattr(args, "user_id", None))

    plan = build_plan(contract, target, phase)
    rendered = json.dumps(plan, indent=2)
    if args.output:
        try:
            output = ensure_external(args.output)
            ArtifactStore(output.parent).write_text(output.name, rendered + "\n")
        except (ValueError, OSError) as error:
            print(json.dumps({"status": "BLOCKED", "message": str(error)}, indent=2))
            return 2
    print(rendered)
    return 0
