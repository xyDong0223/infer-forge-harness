"""Plan-first task runner for infer-forge-harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow `python3 runners/task_runner.py` from anywhere in the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validators.contract_validator import (
    find_placeholders,
    validate_contract_file,
    validate_executable,
)
from validators.deployment_validator import (
    validate_deployment_status,
    validate_environment_status,
)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to run tasks") from exc
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_plan(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": contract.get("metadata", {}).get("name"),
        "task_type": contract.get("metadata", {}).get("task_type"),
        "mode": "PLAN_ONLY",
        "actions": contract.get("actions", []),
        "requires": {
            "explicit_execute": True,
            "cluster_access": True,
            "environment_adapter": "adapters/kunlun_p800",
        },
    }


def execute(
    contract: dict[str, Any],
    contract_path: Path,
    artifact_dir: Path | None,
    attach_pod: str | None = None,
    phase: str = "all",
) -> int:
    """Run the task against the real cluster and let a Validator decide."""
    from adapters import ClusterConfig, get_hardware

    KunlunP800Adapter = get_hardware()

    from runners.deployment_proof import DeploymentProofRunner

    task_type = contract["metadata"]["task_type"]
    # environment_proof and service_proof are the two halves of a deployment
    # proof: the same executor, stopped at a different exit criterion.
    phase = {"environment_proof": "environment", "service_proof": "service"}.get(task_type, phase)
    if task_type not in ("deployment_proof", "environment_proof", "service_proof"):
        print(json.dumps({"status": "EXECUTION_NOT_CONFIGURED", "message": "no executor for this task_type"}, indent=2))
        return 4

    repo_root = Path(__file__).resolve().parents[1]
    if task_type == "environment_proof":
        profile = load_yaml(repo_root / "config" / "clusters" / "p800-cluster.yaml")
        base = profile.get("validation", {}).get("base_model", {})
        deployment = profile.get("deployment", {})
        cluster = profile.get("cluster", {})
        user_id = os.environ.get("USER_ID", "").strip()
        if not user_id:
            print(json.dumps({"status": "INPUT_REQUIRED", "paths": ["$env.USER_ID"]}, indent=2))
            return 3
        if not base.get("required") or not base.get("path"):
            print(json.dumps({"status": "CONTRACT_INVALID", "message": "base model is not configured"}, indent=2))
            return 2
        contract["context"]["model"] = {"name": base["name"], "path": base["path"], "pvc": deployment["model_pvc"]}
        contract["context"]["server"] = {"host": "0.0.0.0", "port": 8356, **base}
        contract["context"]["target"] = {"hardware": "Kunlunxin-3-P800", "device_count": deployment["xpu_count"], "namespace": cluster["namespace"], "volcano_queue": deployment["queue"], "dedicated_pool": deployment["node_pool"]}
        contract["context"]["software"] = {"image": deployment["image"]}
        common_setup = profile.get("runtime", {}).get("common_setup", [])
        optional = f" --max-num-batched-tokens {base['max_num_batched_tokens']} --block-size {base['block_size']} --gpu-memory-utilization {base['gpu_memory_utilization']}" if base.get("max_num_batched_tokens") else ""
        contract["execution"] = {"mode": "execute", "namespace": cluster["namespace"], "resource_name": f"{user_id}-environment-base", "manifest": deployment["base_manifest"], "startup_timeout_seconds": 1800, "health_interval_seconds": 10, "health_successes_required": 3, "retain_on_failure": True, "commands": {"install": ["bash /workspace/install_vllm_kunlun.sh"], "setup": common_setup, "serve": [f"python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8356 --model {base['path']} --trust-remote-code --max-model-len {base['max_model_len']} --max-num-seqs {base['max_num_seqs']} --tensor-parallel-size {base['tensor_parallel_size']} --dtype {base['dtype']} --served-model-name {base['served_model_name']}{optional}"]}}
        contract["checks"] = {"health": {"path": "/health", "expected_status": 200}, "chat": {"path": "/v1/chat/completions", "method": "POST", "expected_non_empty_text": True, "payload": {"model": base["served_model_name"], "messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 16}}, "backend": {"expected": "kunlun", "reject_unexpected_fallback": True}}
    errors = validate_executable(contract)
    if errors:
        print(json.dumps({"status": "CONTRACT_INVALID", "errors": errors}, indent=2))
        return 2
    config = ClusterConfig.load()
    if contract["execution"]["namespace"] != config.namespace:
        print(
            json.dumps(
                {
                    "status": "NEEDS_HUMAN",
                    "message": "contract namespace does not match the harness config",
                },
                indent=2,
            )
        )
        return 5
    artifacts = contract.get("artifacts") or {}
    target_dir = artifact_dir or Path(artifacts.get("directory", repo_root / "artifacts"))
    runner = DeploymentProofRunner(
        contract=contract,
        adapter=KunlunP800Adapter(config),
        repo_root=repo_root,
        artifact_dir=target_dir,
        attach_pod=attach_pod,
        phase=phase,
    )
    status = runner.run()
    gate = (
        validate_environment_status(status)
        if phase == "environment"
        else validate_deployment_status(status)
    )
    status["validator"] = {"passed": not gate, "errors": gate}
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if not gate else 6


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or plan an inference engineering task")
    parser.add_argument("contract", type=Path)
    parser.add_argument("--execute", action="store_true", help="Run against the real cluster")
    parser.add_argument("--output", type=Path, help="Where to write the plan (plan mode)")
    parser.add_argument("--artifact-dir", type=Path, help="Override the contract artifact directory")
    parser.add_argument(
        "--phase",
        choices=["all", "environment", "service"],
        default="all",
        help="Which half of the proof to run; a contract's task_type overrides this",
    )
    parser.add_argument(
        "--attach-pod",
        help="Prove against an already prepared Pod instead of creating one (Imported Context)",
    )
    parser.add_argument(
        "--server-log",
        help="Override execution.server_log with a path of this attempt's own. "
             "A reproof (e.g. the MAT-006 triage rerun) must not truncate the "
             "log of the attempt it exists to explain.",
    )
    args = parser.parse_args()

    errors = validate_contract_file(args.contract)
    contract = load_yaml(args.contract)
    if errors:
        print(json.dumps({"status": "CONTRACT_INVALID", "errors": errors}, indent=2))
        return 2
    if args.server_log:
        contract.setdefault("execution", {})["server_log"] = args.server_log
    placeholders = [] if args.execute and contract.get("metadata", {}).get("task_type") == "environment_proof" else find_placeholders(contract)
    if placeholders:
        print(json.dumps({"status": "INPUT_REQUIRED", "paths": placeholders}, indent=2))
        return 3
    if args.execute:
        return execute(contract, args.contract, args.artifact_dir, args.attach_pod, args.phase)

    plan = build_plan(contract)
    rendered = json.dumps(plan, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
