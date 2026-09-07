"""Plan-first task runner for kunlun-inference-agent."""

from __future__ import annotations

import argparse
import json
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
from validators.deployment_validator import validate_deployment_status


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
) -> int:
    """Run the task against the real cluster and let a Validator decide."""
    from adapters.kunlun_p800 import ClusterConfig, KunlunP800Adapter

    from runners.deployment_proof import DeploymentProofRunner

    errors = validate_executable(contract)
    if errors:
        print(json.dumps({"status": "CONTRACT_INVALID", "errors": errors}, indent=2))
        return 2
    if contract["metadata"]["task_type"] != "deployment_proof":
        print(json.dumps({"status": "EXECUTION_NOT_CONFIGURED", "message": "no executor for this task_type"}, indent=2))
        return 4

    repo_root = Path(__file__).resolve().parents[1]
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
    )
    status = runner.run()
    gate = validate_deployment_status(status)
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
        "--attach-pod",
        help="Prove against an already prepared Pod instead of creating one (Imported Context)",
    )
    args = parser.parse_args()

    errors = validate_contract_file(args.contract)
    contract = load_yaml(args.contract)
    if errors:
        print(json.dumps({"status": "CONTRACT_INVALID", "errors": errors}, indent=2))
        return 2
    placeholders = find_placeholders(contract)
    if placeholders:
        print(json.dumps({"status": "INPUT_REQUIRED", "paths": placeholders}, indent=2))
        return 3
    if args.execute:
        return execute(contract, args.contract, args.artifact_dir, args.attach_pod)

    plan = build_plan(contract)
    rendered = json.dumps(plan, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
