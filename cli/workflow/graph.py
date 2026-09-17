"""Walk a Task Graph instead of hand-feeding one Task's output to the next.

The workflow file already declared nodes and edges; nothing read it. Every run so
far was an operator pasting the previous artifact's path into the next command,
which is exactly the step where a stale path silently answers for the wrong model.

The executor resolves each node's inputs from the Journal by `consumes`, so a node
runs against facts recorded for this subject in this environment or does not run at
all. Default is `--plan`: print the resolved commands without executing, because a
graph that cannot be inspected before it touches a cluster is not safe to trust.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from core.storage import WritePolicyError
from runners.graph_runner import emit_summary, run


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=REPO_ROOT / "workflows" / "model_adaptation.yaml")
    parser.add_argument("--subject", required=True, help="e.g. Qwen3-8B")
    parser.add_argument(
        "--target",
        type=Path,
        help="platform target YAML; applies the compatibility gate before planning",
    )
    parser.add_argument("--artifact-root", type=Path,
                        help="external run root override; otherwise --run-id is required")
    parser.add_argument("--run-id", help="explicit durable run identity")
    parser.add_argument(
        "--scheduler-state", type=Path,
        help="connect an existing adaptation run to the persistent operator scheduler",
    )
    parser.add_argument(
        "--operator-report", type=Path,
        help="measured operator contracts supplementing static gaps (requires --scheduler-state)",
    )
    parser.add_argument("--shim-registry", type=Path,
                        help="declared shim contracts consumed by the runtime shim handoff")
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="environment fingerprint; facts are only reused within it")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="node context, e.g. model_path=/mnt/cluster/... or pod=...")
    parser.add_argument("--from-node", help="start here instead of the entry task")
    parser.add_argument("--until-node", help="stop after this node")
    parser.add_argument("--execute", action="store_true", help="actually run; default is --plan")
    parser.add_argument("--resume", action="store_true",
                        help="reuse successful Journal facts and skip completed nodes")
    parser.add_argument("--loop-state", type=Path,
                        help="Task Memory JSON path; defaults under artifact-root")
    parser.add_argument("--json", action="store_true",
                        help="emit one machine-readable summary per terminal decision")
    parser.add_argument("--auto-recover", action="store_true",
                        help="on a failed node, consult the brain before following the "
                             "failure edge: decide -> act -> rerun, bounded by --recovery-budget")
    parser.add_argument("--interaction-mode", choices=("headless", "codex"), default="headless",
                        help="codex returns a durable decision handoff instead of waiting for a brain")
    parser.add_argument("--recovery-budget", type=int, default=3,
                        help="repair attempts per failed node before the failure edge applies")
    parser.add_argument("--brain", choices=("rule", "agent"), default="agent",
                        help="decision source: 'agent' delegates to an external decider "
                             "(LLM) through decision_request/decision files; 'rule' is the "
                             "deterministic safety net")
    parser.add_argument("--decide-command", default=None,
                        help="decider command for --brain agent; receives the request and "
                             "response paths. Without it the runner waits for decision.json "
                             "to appear next to the request.")
    parser.add_argument("--decide-timeout", type=float, default=600.0,
                        help="seconds to wait for one decision in agent mode")
    parser.add_argument("--watch-interval", type=float, default=30.0,
                        help="heartbeat seconds for the node watch journal; "
                             "0 disables the watch (a silent long node is "
                             "expected — the journal is what keeps it "
                             "observable)")
    args = parser.parse_args()
    return run(args)


def main() -> int:
    try:
        return _main()
    except WritePolicyError as error:
        emit_summary({"status": "BLOCKED", "reason_code": "WRITE_POLICY",
                      "message": str(error)}, "--json" in sys.argv)
        return 2
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        emit_summary({"status": "BLOCKED", "reason_code": "INVALID_INPUT",
                      "message": str(error)}, "--json" in sys.argv)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
