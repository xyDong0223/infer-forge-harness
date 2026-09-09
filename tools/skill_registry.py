"""Machine-readable Skill registry used by Agents and runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "catalog" / "skill_catalog.yaml"
DEFAULT_TOOL_CATALOG = REPO_ROOT / "catalog" / "tool_catalog.yaml"


class SkillResolutionError(ValueError):
    """Raised when a task cannot be mapped to a usable Skill."""


def load_catalog(path: Path = DEFAULT_CATALOG) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("kind") != "SkillCatalog":
        raise SkillResolutionError(f"{path} is not a SkillCatalog")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SkillResolutionError(f"{path} has no Skill entries")
    return entries


def resolve(task_type: str, path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    matches = [
        skill for skill in load_catalog(path)
        if task_type in skill.get("task_types", [])
        and not skill.get("when")
    ]
    if len(matches) != 1:
        raise SkillResolutionError(
            f"expected exactly one Skill for task_type={task_type!r}, found {len(matches)}"
        )
    skill = matches[0]
    required = ("id", "task_types", "tools", "verification", "exit_conditions")
    missing = [field for field in required if not skill.get(field)]
    if missing:
        raise SkillResolutionError(
            f"Skill {skill.get('id', '<unknown>')} is missing: {', '.join(missing)}"
        )
    return skill


def resolve_for_context(
    task_type: str,
    context: dict[str, Any] | None = None,
    path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    """Prefer a narrowly targeted Skill when its activation facts are present."""
    context = context or {}
    candidates = [
        skill
        for skill in load_catalog(path)
        if task_type in skill.get("task_types", [])
        and all(
            context.get(key) == value
            for key, value in (skill.get("when") or {}).items()
        )
    ]
    if not candidates:
        raise SkillResolutionError(f"no Skill matches task_type={task_type!r} and context")
    return max(candidates, key=lambda skill: len(skill.get("when") or {}))


def index_by_task_type(path: Path = DEFAULT_CATALOG) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for skill in load_catalog(path):
        if skill.get("when"):
            continue
        for task_type in skill.get("task_types", []):
            if task_type in index:
                raise SkillResolutionError(f"duplicate task_type registration: {task_type}")
            index[task_type] = skill
    return index


def validate_tool_references(
    path: Path = DEFAULT_CATALOG, tool_path: Path = DEFAULT_TOOL_CATALOG
) -> list[str]:
    """Catch stale tool names before an Agent starts execution."""
    tool_payload = yaml.safe_load(tool_path.read_text(encoding="utf-8")) or {}
    known_tools = {entry.get("id") for entry in tool_payload.get("entries", [])}
    errors = []
    for skill in load_catalog(path):
        for tool in skill.get("tools", []):
            if tool not in known_tools:
                errors.append(f"{skill.get('id')}: unknown tool {tool}")
    return errors
