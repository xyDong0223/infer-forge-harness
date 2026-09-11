"""Build the known-operators index dispatch deduplicates against.

The GLM-5.2 lesson (skill catalog, operator-task-dispatch): before dispatching
a new operator, search the operator repo — two of three dispatched shim kernels
already existed in XSpeedGate, so one dispatch produced a duplicate and two
produced work an op-gen agent correctly answered "already exists".

The index is generated from the operator repo's Python stubs (.pyi), the same
surface the plugin imports, and is committed so dispatch never depends on the
repo checkout being present at dispatch time.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPERATOR_REPO = Path("/ssd1/dongxinyu03/baidu/hac-aiacc/XSpeedGate")
DEFAULT_OUTPUT = REPO_ROOT / "catalog" / "known_operators.json"

DEFINITION = re.compile(r"^def\s+([a-z0-9_]+)\s*\(", re.MULTILINE)


def collect_operators(repo: Path) -> dict[str, list[str]]:
    """One bucket per stub module, so the index can say where a name came from."""
    buckets: dict[str, list[str]] = {}
    for stub in sorted(repo.glob("xspeedgate_ops/*.pyi")):
        names = DEFINITION.findall(stub.read_text(encoding="utf-8"))
        if names:
            buckets[stub.stem.lstrip("_")] = sorted(set(names))
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_OPERATOR_REPO,
                        help="operator repo with xspeedgate_ops/*.pyi stubs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    buckets = collect_operators(args.repo)
    if not buckets:
        print(f"no stubs found under {args.repo}/xspeedgate_ops — refusing to write "
              "an empty index that would silently disable dedup")
        return 1
    operators = sorted({name for names in buckets.values() for name in names})
    payload = {
        "source_repo": str(args.repo),
        "modules": buckets,
        "operators": operators,
        "operator_count": len(operators),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"{args.out}: {len(operators)} operators across {len(buckets)} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
