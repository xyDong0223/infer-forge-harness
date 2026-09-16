"""Validate the public v0.1 repository scaffold."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        import yaml
    except ImportError:
        print("PyYAML is required for scaffold validation", file=sys.stderr)
        return 2

    root = args.root.resolve()
    sys.path.insert(0, str(root))
    yaml_files = sorted(path for path in root.rglob("*.yaml")
                        if not set(path.relative_to(root).parts) &
                        {".git", ".venv", "__pycache__", "artifacts"})
    for path in yaml_files:
        with path.open(encoding="utf-8") as handle:
            yaml.safe_load(handle)
        print(f"YAML OK: {path.relative_to(root)}")

    try:
        for directory in ("core", "engine", "operations", "cli", "runners",
                          "validators", "adapters", "runtimes", "tools"):
            for path in (root / directory).rglob("*.py"):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except SyntaxError as error:
        print(str(error), file=sys.stderr)
        return 3

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(root / "tests")],
        cwd=root,
        check=False,
    )
    return 0 if result.returncode == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
