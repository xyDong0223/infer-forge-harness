"""Validate the public v0.1 repository scaffold."""

from __future__ import annotations

import compileall
import sys
import unittest
from pathlib import Path


def main() -> int:
    try:
        import yaml
    except ImportError:
        print("PyYAML is required for scaffold validation", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    yaml_files = sorted(root.rglob("*.yaml"))
    for path in yaml_files:
        with path.open(encoding="utf-8") as handle:
            yaml.safe_load(handle)
        print(f"YAML OK: {path.relative_to(root)}")

    if not compileall.compile_dir(str(root / "runners"), quiet=1):
        return 3
    if not compileall.compile_dir(str(root / "validators"), quiet=1):
        return 3

    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 4


if __name__ == "__main__":
    raise SystemExit(main())
