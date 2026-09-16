"""Check repository source ownership and references to canonical commands."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

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


def scan(root: Path) -> list[str]:
    errors: list[str] = []
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
