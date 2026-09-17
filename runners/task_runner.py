"""Plan-first task runner for infer-forge-harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.paths import REPO_ROOT
from core.user_identity import resolve_user_id
from validators.contract_validator import (
    find_placeholders,
    validate_contract_file,
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


def build_plan(contract: dict[str, Any], target: TargetContext | None = None) -> dict[str, Any]:
    resolved = contract_target(contract, target)
    require_supported(resolved)
    return {
        "task_id": contract.get("metadata", {}).get("name"),
        "task_type": contract.get("metadata", {}).get("task_type"),
        "mode": "PLAN_ONLY",
        "actions": contract.get("actions", []),
        "requires": {
            "explicit_execute": True,
            "cluster_access": True,
            "environment_adapter": f"adapters/{resolved.hardware.replace('/', '_')}",
        },
    }


def execute(
    contract: dict[str, Any],
    contract_path: Path,
    artifact_dir: Path | None,
    attach_pod: str | None = None,
    phase: str = "all",
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
        contract.setdefault("execution", {})["user_id"] = owner
        ArtifactStore(attempt.input).write_json("task_contract.json", contract)
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
    phase = {"environment_proof": "environment", "service_proof": "service"}.get(task_type, phase)
    if task_type not in ("deployment_proof", "environment_proof", "service_proof"):
        return finish({"status": "EXECUTION_NOT_CONFIGURED", "message": "no executor for this task_type"}, 4)

    try:
        target_context = contract_target(contract, target)
        bundle = resolve_adapters(target_context, require_supported=True)
    except ValueError as error:
        return finish({"status": "BLOCKED", "message": str(error)}, 2)

    if phase == "environment":
        profile = load_yaml(REPO_ROOT / "config" / "clusters" / "p800-cluster.yaml")
        base = profile.get("validation", {}).get("base_model", {})
        deployment = profile.get("deployment", {})
        cluster = profile.get("cluster", {})
        user_id = resolve_user_id(recorded=contract.get("execution", {}).get("user_id"))
        if not user_id:
            return finish({
                "status": "INPUT_REQUIRED", "paths": ["$.execution.user_id"],
                "message": "Ask the user for their user ID and pass --user-id <USER_ID> "
                           "(or execution.user_id in the contract; legacy USER_ID is supported). "
                           "Do not infer it from the host login or an example resource name.",
            }, 3)
        if not base.get("required") or not base.get("path"):
            return finish({"status": "CONTRACT_INVALID", "message": "base model is not configured"}, 2)
        contract["context"]["model"] = {"name": base["name"], "path": base["path"], "pvc": deployment["model_pvc"]}
        contract["context"]["server"] = {"host": "0.0.0.0", "port": 8356, **base}
        contract["context"]["target"] = {"hardware": "Kunlunxin-3-P800", "device_count": deployment["xpu_count"], "namespace": cluster["namespace"], "volcano_queue": deployment["queue"], "dedicated_pool": deployment["node_pool"]}
        contract["context"]["software"] = {"image": deployment["image"]}
        common_setup = profile.get("runtime", {}).get("common_setup", [])
        serve_config = {
            "port": 8356,
            "path": base["path"],
            **base,
        }
        previous_execution = contract.get("execution", {})
        contract["execution"] = {
            "mode": "execute", "namespace": cluster["namespace"],
            "user_id": user_id,
            "resource_name": f"{user_id}-environment-base",
            "manifest": deployment["base_manifest"],
            "startup_timeout_seconds": 1800,
            "health_interval_seconds": previous_execution.get("health_interval_seconds", 10),
            "health_successes_required": previous_execution.get("health_successes_required", 3),
            "retain_on_failure": True,
            "commands": {
                "install": ["bash /workspace/install_vllm_kunlun.sh"],
                "setup": common_setup,
                "serve": [bundle.runtime.build_serve_command(serve_config)],
            },
        }
        if previous_execution.get("server_log"):
            contract["execution"]["server_log"] = previous_execution["server_log"]
        contract["checks"] = {"health": {"path": "/health", "expected_status": 200}, "chat": {"path": "/v1/chat/completions", "method": "POST", "expected_non_empty_text": True, "payload": {"model": base["served_model_name"], "messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 16}}, "backend": {"expected": "kunlun", "reject_unexpected_fallback": True}}
    contract.setdefault("context", {})["resolved_target"] = {
        "model": target.model if target else target_context.model,
        "hardware": target_context.hardware,
        "engine": target_context.engine,
        "backend": target_context.backend,
        "plugin": target_context.plugin,
        "revisions": target_context.revisions,
        "compatibility": bundle.compatibility,
    }
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

    errors = validate_contract_file(args.contract)
    contract = load_yaml(args.contract)
    if errors:
        print(json.dumps({"status": "CONTRACT_INVALID", "errors": errors}, indent=2))
        return 2
    try:
        target = load_target(args.target) if args.target else None
        if args.subject:
            if target is None:
                raise ValueError("--subject requires --target")
            target = bind_subject(target, args.subject)
        require_supported(contract_target(contract, target))
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, indent=2))
        return 2
    if args.server_log:
        contract.setdefault("execution", {})["server_log"] = args.server_log
    environment_phase = (args.phase == "environment" or
                         contract.get("metadata", {}).get("task_type") == "environment_proof")
    placeholders = [] if args.execute and environment_phase else find_placeholders(contract)
    if placeholders:
        print(json.dumps({"status": "INPUT_REQUIRED", "paths": placeholders}, indent=2))
        return 3
    if args.execute:
        return execute(contract, args.contract, args.artifact_dir, args.attach_pod, args.phase,
                       target, args.run_id, getattr(args, "user_id", None))

    plan = build_plan(contract, target)
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
