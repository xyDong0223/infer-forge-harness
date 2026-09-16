"""Small, dependency-light tensor diff primitives and JSON CLI.

The comparator is intentionally independent of any accelerator framework. It
accepts JSON arrays so probes can dump tensors from PyTorch, XPU, or a vendor
binding and grade them with one implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from operations.validation.tensor_diff import execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--max-relative-l2", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
