"""Plan-first task runner for kunlun-inference-agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validators.contract_validator import find_placeholders, validate_contract_file


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or plan an inference engineering task")
    parser.add_argument("contract", type=Path)
    parser.add_argument("--execute", action="store_true", help="Reserved for an authorized adapter")
    parser.add_argument("--output", type=Path)
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
        print(json.dumps({"status": "EXECUTION_NOT_CONFIGURED", "message": "Use an environment adapter."}, indent=2))
        return 4

    plan = build_plan(contract)
    rendered = json.dumps(plan, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
