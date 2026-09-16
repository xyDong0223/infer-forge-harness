"""Journal: make evidence retrievable across Tasks.

Until now each Task wrote an isolated directory under the artifact root, so the
only way to feed one Task's output into the next was to hand it a path. That makes
two things impossible: knowing whether a fact already exists, and knowing whether
it is still about the current world.

A fact is (kind, subject, environment fingerprint) -> artifact bundle. The
fingerprint is what makes a hit trustworthy: the same model on the same stack
commit is the same fact, and a different stack commit is a different one, even if
the model is identical. Without it, a cached fact would silently answer for an
environment nobody validated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from engine.state.journal import DEFAULT_JOURNAL, KINDS, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("record")
    add.add_argument("--kind", required=True, choices=sorted(set(KINDS.values())))
    add.add_argument("--subject", required=True, help="e.g. Qwen3-8B or a pod name")
    add.add_argument("--state", required=True)
    add.add_argument("--artifacts", type=Path, required=True)
    add.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")

    find = sub.add_parser("query")
    find.add_argument("--kind", required=True)
    find.add_argument("--subject")
    find.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    find.add_argument("--state", action="append", default=[])

    sub.add_parser("list")
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
