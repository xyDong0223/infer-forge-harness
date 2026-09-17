"""Check repository source ownership and references to canonical commands."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path, PurePosixPath
import re
import shlex
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from core.task_execution import load_task  # noqa: E402

IGNORED = {".git", ".venv", ".pytest_cache", "__pycache__", "artifacts", "openwiki"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".json"}
SOURCE_REFERENCE = re.compile(
    r"(?:cli|operations|core|engine|runners|validators|adapters|runtimes|tools)"
    r"/(?:[A-Za-z_][\w-]*/)*[A-Za-z_][\w-]*\.(?:py|sh)\b"
)
LEGACY_RUNNER_COMMAND = re.compile(
    r"(?:python(?:3)?\s+|[\"'])runners/"
    r"(?:graph_runner|task_runner|triage_executor|patch_executor|correctness_executor)\.py"
)
LEGACY_IMPORT = re.compile(r"\b(?:from|import)\s+(?:patches|implementations)\b")
LIBRARY_ROOTS = {"core", "engine", "operations", "runners", "adapters", "runtimes", "validators"}


def check_tool_catalog(root: Path) -> list[str]:
    """Validate indirection through the same Task reader used by the Graph."""
    catalog = root / "catalog" / "tool_catalog.yaml"
    if not catalog.is_file():
        return []
    try:
        payload = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("kind") != "ToolCatalog":
            raise ValueError("expected a ToolCatalog mapping")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("entries must be a nonempty list")
    except (OSError, ValueError, yaml.YAMLError) as error:
        return [f"catalog/tool_catalog.yaml: {error}"]
    errors, ids = [], set()
    for index, entry in enumerate(entries):
        label = f"catalog/tool_catalog.yaml entry {index}"
        try:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"].strip():
                raise ValueError("tool entry requires a nonempty id")
            label = f"catalog/tool_catalog.yaml {entry['id']}"
            if entry["id"] in ids:
                raise ValueError("duplicate tool id")
            ids.add(entry["id"])
            modes = {"command", "task_definition", "task_definitions"} & entry.keys()
            if len(modes) != 1:
                raise ValueError("provide exactly one command or Task definition reference form")
            if "command" in modes:
                command = entry["command"]
                if not isinstance(command, str):
                    raise ValueError("standalone command must be a string")
                argv = shlex.split(command)
                if len(argv) < 2 or argv[0] != "python3":
                    raise ValueError("standalone command must name a python3 repository script")
                path = PurePosixPath(argv[1])
                if (path.is_absolute() or ".." in path.parts or path.suffix != ".py"
                        or not (root / path).is_file()):
                    raise ValueError(f"missing or invalid standalone command script: {argv[1]}")
                continue
            references = ([entry["task_definition"]] if "task_definition" in modes
                          else entry["task_definitions"])
            if (not isinstance(references, list) or not references
                    or any(not isinstance(value, str) or not value for value in references)):
                raise ValueError("Task references must be a nonempty list of paths")
            if len(references) != len(set(references)):
                raise ValueError("Task references must not contain duplicates")
            for reference in references:
                path = PurePosixPath(reference)
                if (path.is_absolute() or ".." in path.parts or str(path) != reference
                        or len(path.parts) != 3 or path.parts[0] != "tasks" or path.name != "task.yaml"):
                    raise ValueError(f"invalid Task definition reference: {reference}")
                descriptor = load_task(root / path)
                if not descriptor.executable:
                    raise ValueError(f"referenced Task has no execution descriptor: {reference}")
                if not (root / descriptor.argv[1]).is_file():
                    raise ValueError(f"Task entrypoint does not exist: {descriptor.argv[1]}")
        except (OSError, ValueError, yaml.YAMLError) as error:
            errors.append(f"{label}: {error}")
    return errors


def scan(root: Path) -> list[str]:
    errors: list[str] = check_tool_catalog(root)
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (not path.is_file() or path.suffix not in TEXT_SUFFIXES
                or set(relative.parts) & IGNORED
                or any(part.endswith(".egg-info") for part in relative.parts)
                or path.name.endswith("-findings.yaml")):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for reference in sorted(set(SOURCE_REFERENCE.findall(text))):
            if not (root / reference).is_file():
                errors.append(f"{relative}: missing source reference {reference}")
        if LEGACY_RUNNER_COMMAND.search(text):
            errors.append(f"{relative}: runner script invocation must use its cli entry point")
        if LEGACY_IMPORT.search(text):
            errors.append(f"{relative}: removed top-level implementation import")
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(relative))
            except SyntaxError as error:
                errors.append(f"{relative}: invalid Python: {error}")
                continue
            for node in ast.walk(tree):
                modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                if isinstance(node, ast.ImportFrom) and node.module == "tools":
                    modules = [f"tools.{alias.name}" for alias in node.names]
                if any(module.startswith("tools.") and module.split(".")[1] not in
                       {"probe", "patches", "torch"} for module in modules):
                    errors.append(f"{relative}:{node.lineno}: removed host tool import")
                if relative.parts[0] not in LIBRARY_ROOTS:
                    continue
                if any(module == "cli" or module.startswith("cli.") for module in modules):
                    errors.append(f"{relative}:{node.lineno}: library cannot import cli")
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"parse_args", "parse_known_args",
                                              "parse_intermixed_args", "parse_known_intermixed_args"}):
                    errors.append(f"{relative}:{node.lineno}: argument parsing belongs in cli")
                if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) and "__main__" in ast.unparse(node.test):
                    errors.append(f"{relative}:{node.lineno}: executable entry point belongs in cli")
    for path in (root / "tools").glob("*"):
        if path.suffix in {".py", ".sh"} and path.name != "__init__.py":
            errors.append(f"tools/{path.name}: host commands belong in cli, not tools/")
    if (root / "tools").is_dir():
        for path in (root / "tools").iterdir():
            if path.is_dir() and path.name not in {"probe", "patches", "torch", "__pycache__"}:
                errors.append(f"tools/{path.name}: unclassified tool directory")
    for directory in ("patches", "implementations"):
        if (root / directory).exists():
            errors.append(f"removed directory still exists: {directory}/")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    errors = scan(args.root)
    if errors:
        print("\n".join(errors))
        return 1
    print("repository reference check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
