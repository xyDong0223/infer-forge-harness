"""Fixed managed worker measurement entrypoint; never accepts a PASS report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from operations.validation.worker_probe import execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        observation = execute(args.request, args.output)
    except Exception as error:
        print(json.dumps({"status": "MEASUREMENT_FAILED", "error_type": type(error).__name__,
                          "error": str(error)}), file=sys.stderr)
        return 6
    print(json.dumps({"status": "MEASURED", "role": observation["role"],
                      "observations": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
