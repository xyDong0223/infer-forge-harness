"""Source ownership, canonical entry points, and independent CLI startup."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from cli.maintenance.check_repo_references import scan
from core.paths import REPO_ROOT


COMMANDS = sorted(
    path.relative_to(REPO_ROOT).as_posix()
    for path in (REPO_ROOT / "cli").rglob("*.py")
    if path.name not in {"__init__.py", "common.py"}
)


def test_live_repository_references_and_layer_boundaries():
    assert scan(REPO_ROOT) == []


def test_host_tools_are_not_left_as_compatibility_wrappers():
    assert sorted(path.name for path in (REPO_ROOT / "tools").iterdir()
                  if path.suffix in {".py", ".sh"}) == ["__init__.py"]


@pytest.mark.parametrize("command", COMMANDS)
def test_cli_help_starts_outside_checkout_without_runtime_writes(command, tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / command), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=20,
        env={
            **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
            "INFER_FORGE_STATE_ROOT": str(tmp_path / "state"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
    assert not list(tmp_path.iterdir())


def test_reference_checker_rejects_missing_command(tmp_path):
    (tmp_path / "workflow.yaml").write_text(
        "command: python3 cli/" + "missing.py\n", encoding="utf-8",
    )
    assert any("missing source reference" in error for error in scan(tmp_path))


@pytest.mark.parametrize("statement", [
    "from tools import journal\n",
    "from tools.journal import record\n",
    "import tools.journal\n",
])
def test_reference_checker_rejects_removed_python_imports(tmp_path, statement):
    (tmp_path / "caller.py").write_text(statement, encoding="utf-8")
    assert any("removed host tool import" in error for error in scan(tmp_path))


def test_reference_checker_rejects_library_entrypoints_and_reverse_imports(tmp_path):
    library = tmp_path / "operations"
    library.mkdir()
    (library / "example.py").write_text(
        "from cli import common\n"
        "import argparse\n"
        "def execute():\n"
        "    return argparse.ArgumentParser().parse_args()\n"
        'if __name__ == "__main__":\n'
        "    execute()\n",
        encoding="utf-8",
    )
    errors = scan(tmp_path)
    assert any("library cannot import cli" in error for error in errors)
    assert any("argument parsing belongs in cli" in error for error in errors)
    assert any("executable entry point belongs in cli" in error for error in errors)


def test_portable_patch_directory_is_not_a_forbidden_top_level_patch(tmp_path):
    patch = tmp_path / "tools" / "patches" / "repair.py"
    patch.parent.mkdir(parents=True)
    patch.write_text('"""Portable replayable patch."""\n', encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "Run python tools/patches/" + "repair.py\n", encoding="utf-8",
    )
    assert scan(tmp_path) == []
    (tmp_path / "patches").mkdir()
    assert any("removed directory still exists" in error for error in scan(tmp_path))
