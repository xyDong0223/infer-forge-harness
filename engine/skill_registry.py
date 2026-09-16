"""Machine-readable Skill registry used by Agents and runners."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from core.paths import REPO_ROOT
DEFAULT_CATALOG = REPO_ROOT / "catalog" / "skill_catalog.yaml"
DEFAULT_TOOL_CATALOG = REPO_ROOT / "catalog" / "tool_catalog.yaml"
DEFAULT_SKILLS_ROOT = REPO_ROOT / "skills"


class SkillResolutionError(ValueError):
    """Raised when a task cannot be mapped to a usable Skill."""


def load_catalog(path: Path = DEFAULT_CATALOG) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise SkillResolutionError(f"{path} must contain a YAML mapping")
    if payload.get("kind") != "SkillCatalog":
        raise SkillResolutionError(f"{path} is not a SkillCatalog")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SkillResolutionError(f"{path} has no Skill entries")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SkillResolutionError(f"{path} entry {index} must be a mapping")
        if not isinstance(entry.get("id"), str) or not entry["id"]:
            raise SkillResolutionError(f"{path} entry {index} has no valid id")
        task_types = entry.get("task_types")
        if (
            not isinstance(task_types, list) or not task_types
            or not all(isinstance(item, str) and item for item in task_types)
        ):
            raise SkillResolutionError(
                f"{path} entry {entry['id']!r} has invalid task_types"
            )
        when = entry.get("when")
        if when is not None and not isinstance(when, dict):
            raise SkillResolutionError(
                f"{path} entry {entry['id']!r} has invalid when conditions"
            )
        method_package = entry.get("method_package")
        if method_package is not None and (
            not isinstance(method_package, str) or not method_package
        ):
            raise SkillResolutionError(
                f"{path} entry {entry['id']!r} has invalid method_package"
            )
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
    specificity = max(len(skill.get("when") or {}) for skill in candidates)
    matches = [
        skill for skill in candidates
        if len(skill.get("when") or {}) == specificity
    ]
    if len(matches) != 1:
        ids = ", ".join(str(skill.get("id", "<unknown>")) for skill in matches)
        raise SkillResolutionError(
            f"ambiguous Skill match for task_type={task_type!r}: {ids}"
        )
    return matches[0]


def load_packages(skills_root: Path = DEFAULT_SKILLS_ROOT) -> dict[str, dict[str, Any]]:
    """Load installed Skill packages by id without relying on directory naming."""
    packages: dict[str, dict[str, Any]] = {}
    for descriptor_path in sorted(skills_root.glob("*/skill.yaml")):
        payload = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise SkillResolutionError(
                f"{descriptor_path} must contain a YAML mapping"
            )
        if payload.get("kind") != "Skill" or not payload.get("id"):
            raise SkillResolutionError(f"{descriptor_path} is not a valid Skill package")
        if not isinstance(payload["id"], str):
            raise SkillResolutionError(f"{descriptor_path} has an invalid Skill id")
        task_types = payload.get("task_types")
        if (
            not isinstance(task_types, list) or not task_types
            or not all(isinstance(item, str) and item for item in task_types)
        ):
            raise SkillResolutionError(f"{descriptor_path} has invalid task_types")
        for field in ("catalog", "method_document"):
            if not isinstance(payload.get(field), str) or not payload[field]:
                raise SkillResolutionError(f"{descriptor_path} has invalid {field}")
        package_id = str(payload["id"])
        if package_id in packages:
            raise SkillResolutionError(f"duplicate Skill package id: {package_id}")
        packages[package_id] = {
            **payload,
            "_descriptor_path": descriptor_path,
        }
    return packages


def execution_contract(
    skill: dict[str, Any],
    task_type: str,
    *,
    skills_root: Path = DEFAULT_SKILLS_ROOT,
) -> dict[str, Any]:
    """Build the exact method packet an executor or Agent must consume."""
    contract = {
        key: value for key, value in skill.items()
        if key in {
            "id", "task_types", "purpose", "tools", "verification",
            "preconditions", "exit_conditions", "rules", "when",
        }
    }
    contract["task_type"] = task_type
    package_id = skill.get("method_package")
    if not package_id:
        contract["method"] = None
        return contract

    packages = load_packages(skills_root)
    package = packages.get(str(package_id))
    if package is None:
        raise SkillResolutionError(
            f"Skill {skill.get('id', '<unknown>')} references unknown method package "
            f"{package_id!r}"
        )
    if task_type not in package.get("task_types", []):
        raise SkillResolutionError(
            f"method package {package_id!r} does not support task_type={task_type!r}"
        )
    if package.get("catalog") != str(DEFAULT_CATALOG.relative_to(REPO_ROOT)):
        raise SkillResolutionError(
            f"method package {package_id!r} does not reference "
            f"{DEFAULT_CATALOG.relative_to(REPO_ROOT)}"
        )
    method_value = package.get("method_document")
    if not isinstance(method_value, str) or not method_value:
        raise SkillResolutionError(f"method package {package_id!r} has no method_document")
    method_path = (REPO_ROOT / method_value).resolve()
    skills_root_resolved = skills_root.resolve()
    if method_path != skills_root_resolved and skills_root_resolved not in method_path.parents:
        raise SkillResolutionError(
            f"method package {package_id!r} points outside skills/: {method_value}"
        )
    descriptor_path = package["_descriptor_path"].resolve()
    if method_path.parent != descriptor_path.parent:
        raise SkillResolutionError(
            f"method package {package_id!r} document is outside its package directory"
        )
    if not method_path.is_file():
        raise SkillResolutionError(
            f"method package {package_id!r} document does not exist: {method_value}"
        )
    content = method_path.read_text(encoding="utf-8")
    if not content.strip():
        raise SkillResolutionError(f"method package {package_id!r} document is empty")
    if not content.startswith("---\n") or "\n---\n" not in content[4:]:
        raise SkillResolutionError(
            f"method package {package_id!r} document has no YAML front matter"
        )
    front_matter = yaml.safe_load(content.split("\n---\n", 1)[0][4:])
    if front_matter is None:
        front_matter = {}
    if not isinstance(front_matter, dict):
        raise SkillResolutionError(
            f"method package {package_id!r} front matter must be a YAML mapping"
        )
    if front_matter.get("name") != package_id:
        raise SkillResolutionError(
            f"method package {package_id!r} document name does not match its id"
        )
    contract["method"] = {
        "package_id": package_id,
        "document": method_value,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content": content,
    }
    return contract


def validate_method_references(
    path: Path = DEFAULT_CATALOG,
    skills_root: Path = DEFAULT_SKILLS_ROOT,
) -> list[str]:
    """Validate all package links before a workflow performs any work."""
    errors: list[str] = []
    try:
        packages = load_packages(skills_root)
        catalog = load_catalog(path)
    except (OSError, yaml.YAMLError, SkillResolutionError) as error:
        return [str(error)]
    referenced: set[str] = set()
    for skill in catalog:
        package_id = skill.get("method_package")
        if not package_id:
            continue
        referenced.add(str(package_id))
        package = packages.get(str(package_id))
        if package is None:
            errors.append(f"{skill.get('id')}: unknown method package {package_id}")
            continue
        missing_types = sorted(
            set(skill.get("task_types", [])) - set(package.get("task_types", []))
        )
        if missing_types:
            errors.append(
                f"{skill.get('id')}: method package {package_id} does not support "
                f"{', '.join(missing_types)}"
            )
            continue
        try:
            for task_type in skill.get("task_types", []):
                execution_contract(skill, task_type, skills_root=skills_root)
        except (OSError, yaml.YAMLError, SkillResolutionError) as error:
            errors.append(f"{skill.get('id')}: {error}")
    for package_id in sorted(set(packages) - referenced):
        errors.append(f"{package_id}: method package is not referenced by the catalog")
    return errors


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
