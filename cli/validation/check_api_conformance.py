"""MAT-009 API Conformance: the response shape, not the numbers.

A different failure mode from every other Task here. The model can compute correctly,
the text can be right, and the API response can still be wrong: `reasoning_content`
empty, `tool_calls` never parsed, `finish_reason` plain. No differential over logits
sees that, which is why this is its own family rather than another MAT-008 dimension.

Parsers are pure text transforms, so this needs a pod but not a running server. That
is deliberate: it can run before a deployment is configured with `--reasoning-parser`
and still say whether the pair the deployment is about to use actually works.

Samples live in the contract, not here. Adding a model means writing down its parser
pair and a sample of its output, which is reviewable — and the probe refuses the
sample unless the marker it uses appears in that model's own chat template.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from cli.common import run_managed_tool
from operations.validation.check_api_conformance import CONTRACT, ConformanceFailed, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="model id, e.g. Qwen3-30B-A3B")
    parser.add_argument("--model-path", help="weights; defaults to the ModelRequest's source")
    parser.add_argument("--model-request", help="model_request.yaml from MAT-001")
    parser.add_argument("--pod", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except ConformanceFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
