"""Small, dependency-free validators for the v0.1 task contract shape."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_PLACEHOLDER = re.compile(r"\$\{[^}]+\}")


def validate_task_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("api_version", "kind", "metadata", "context", "actions", "acceptance"):
        if field not in contract:
            errors.append(f"missing required field: {field}")
    if contract.get("kind") != "Task":
        errors.append("kind must be Task")
    metadata = contract.get("metadata", {})
    for field in ("name", "task_type"):
        if not metadata.get(field):
            errors.append(f"metadata.{field} is required")
    if not isinstance(contract.get("actions"), list) or not contract.get("actions"):
        errors.append("actions must be a non-empty list")
    return errors


def find_placeholders(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, str):
        return [path] if _PLACEHOLDER.search(value) else []
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.extend(find_placeholders(item, f"{path}.{key}"))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(find_placeholders(item, f"{path}[{index}]"))
        return result
    return []


def validate_contract_file(path: str | Path) -> list[str]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - clear operator guidance
        return [f"PyYAML is required to load contracts: {exc}"]
    with Path(path).open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle) or {}
    return validate_task_contract(contract)
