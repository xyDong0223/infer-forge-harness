"""Fail when repository files reference removed or forbidden runtime paths."""
from __future__ import annotations

import argparse
from pathlib import Path

FORBIDDEN = ("patches/", "implementations/", "from patches", "import patches")


def scan(root: Path) -> list[str]:
    errors: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or "__pycache__" in path.parts or "openwiki" in path.parts or path.name == "check_repo_references.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in FORBIDDEN:
            if marker in text:
                errors.append(f"{path}: forbidden reference {marker!r}")
    for directory in ("patches", "implementations"):
        if (root / directory).exists():
            errors.append(f"removed directory still exists: {directory}/")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = scan(args.root)
    if errors:
        print("\n".join(errors))
        return 1
    print("repository reference check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
