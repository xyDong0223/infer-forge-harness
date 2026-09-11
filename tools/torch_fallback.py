"""MAT-030 Torch Fallback: a kernel that will not run gets a torch shim, not a meeting.

The operator chain used to stop for a human at every failure: triage was
operator-driven, patch placement paused for a review, gap classification
failures went to NEEDS_HUMAN. The record says none of that ever helped —
Qwen3-8B, MiniMax-M3 and GLM-5.2 were all brought up by hand-writing torch
shims while the graph waited. So this node makes the practiced move the
default one: whatever failed, the answer is "write a torch shim for these
operators, keep bring-up moving, and record the operator debt".

This tool is offline and reversible. It turns failure artifacts into a shim
work list, dispatches one durable operator request per operator through the
same MAT-024 machinery the shim handoff uses, and exits FALLBACK_APPLIED.
Writing the shims themselves is fanned out to subagents (one per operator).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.operator_lifecycle import dispatch as dispatch_requests  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_operators(payload: dict) -> list[str]:
    """Pull operator names out of any failure/gap artifact shape we produce.

    Handles gap classifications (operator/symbol/name/axis keys), triage and
    status artifacts that carry an `operators` list, and shim registries.
    """
    if not isinstance(payload, dict):
        return []
    names: list[str] = []
    for key in ("operator", "symbol", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            names.append(value)
    if payload.get("class") == "CAPABILITY_MISSING" and payload.get("axis"):
        names.append(payload["axis"])
    for key in ("operators", "gaps", "findings", "failed_kernels"):
        for item in payload.get(key) or []:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.extend(extract_operators(item))
    # De-duplicate while keeping order: one request per operator, not per mention.
    return list(dict.fromkeys(name for name in names if name))


def write_brief(out: Path, operator: str, subject: str, request_id: str | None) -> None:
    """The one-page task a shim-authoring subagent consumes."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "shim_brief.md").write_text(
        f"# Torch shim task — {operator}\n"
        f"\n"
        f"- subject: {subject}\n"
        f"- request: `{request_id or '—'}`\n"
        f"\n"
        f"Replace `{operator}` with a pure-torch implementation. P800 has no\n"
        f"triton path — do not attempt one. Keep the shim reversible (env-var\n"
        f"switch), register it in the MAT-029 shim registry in the same pass, and\n"
        f"compare numerically against the kernel it replaces before claiming it.\n",
        encoding="utf-8",
    )


def prepare(gaps: list[dict], out: Path, subject: str, baseline_id: str | None = None,
            dispatch_subject: str | None = None) -> dict:
    """Dispatch one durable request per operator and record the shim work list.

    `dispatch_subject` keeps request ids unique when several children dispatch
    concurrently: each names its operator, so `Model-foo-op-001` cannot collide
    with `Model-bar-op-001`.
    """
    dispatch_input = out / "fallback_gaps.json"
    dispatch_input.parent.mkdir(parents=True, exist_ok=True)
    dispatch_input.write_text(
        json.dumps({"gaps": gaps}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    record = dispatch_requests(dispatch_input, out, dispatch_subject or subject, baseline_id)
    operators = [gap_operator(gap) for gap in gaps]
    request_ids = record.get("requests") or []
    for operator, request_id in zip(operators, request_ids):
        write_brief(out, operator, subject, request_id)
    report = {
        "state": "FALLBACK_APPLIED",
        "subject": subject,
        "operators": operators,
        "request_ids": request_ids,
        "dispatch": record,
        "instruction": "replace each listed operator with a torch shim; register the shim "
                       "in the MAT-029 registry in the same pass",
    }
    (out / "fallback_status.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def gap_operator(gap: dict) -> str | None:
    """The operator a gap names, under any of the keys our artifacts use."""
    for key in ("operator", "symbol", "name"):
        value = gap.get(key)
        if isinstance(value, str) and value:
            return value
    if gap.get("class") == "CAPABILITY_MISSING":
        return gap.get("axis")
    return None


def collect_gaps(args: argparse.Namespace) -> list[dict]:
    """Gather shim targets from every input shape the apply action accepts."""
    gaps: list[dict] = []
    if getattr(args, "gaps", None):
        payload = _load(args.gaps)
        items = payload.get("gaps", payload.get("findings", [])) if isinstance(payload, dict) else payload
        gaps.extend(item for item in (items or []) if isinstance(item, dict))
    for failure_path in getattr(args, "failure", None) or []:
        for name in extract_operators(_load(failure_path)):
            gaps.append({"name": name, "class": "TORCH_SHIM", "source": "torch_fallback"})
    for name in getattr(args, "operator", None) or []:
        gaps.append({"name": name, "class": "TORCH_SHIM", "source": "torch_fallback"})
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action")

    apply_parser = sub.add_parser("apply", help="turn failure evidence into shim work")
    apply_parser.add_argument("--gaps", type=Path, help="gap classification or any gaps JSON")
    apply_parser.add_argument("--failure", type=Path, action="append", default=[],
                              help="any status/triage/registry JSON; repeatable")
    apply_parser.add_argument("--operator", action="append", default=[],
                              help="operator name; repeatable, wins over artifacts")
    apply_parser.add_argument("--subject", required=True)
    apply_parser.add_argument("--baseline-id")
    apply_parser.add_argument("--out", type=Path, required=True)

    # The fan-out list command: prints one JSON array line, one operator per
    # element, so the graph runner can spawn a subagent per shim.
    list_parser = sub.add_parser("list-operators", help="print the shim work list as a JSON array")
    list_parser.add_argument("--gaps", type=Path)
    list_parser.add_argument("--failure", type=Path, action="append", default=[])
    list_parser.add_argument("--subject", required=True)

    # One fan-out child per operator: dispatches that operator's request and
    # writes the brief a shim-authoring subagent consumes.
    shim_parser = sub.add_parser("shim", help="prepare one operator's torch shim task")
    shim_parser.add_argument("--operator", required=True)
    shim_parser.add_argument("--subject", required=True)
    shim_parser.add_argument("--baseline-id")
    shim_parser.add_argument("--out", type=Path, required=True)

    # The fan-in: merge the per-operator children into one node-level status.
    aggregate_parser = sub.add_parser("aggregate", help="merge per-operator fallback children")
    aggregate_parser.add_argument("--child", type=Path, action="append", default=[])
    aggregate_parser.add_argument("--subject")
    aggregate_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.action == "apply":
        gaps = collect_gaps(args)
        if not gaps:
            report = {
                "state": "FALLBACK_SKIPPED",
                "subject": args.subject,
                "operators": [],
                "reason": "no operator named in the inputs: nothing to shim",
            }
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "fallback_status.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print("FALLBACK_SKIPPED no operators")
            return 0
        report = prepare(gaps, args.out, args.subject, args.baseline_id)
        print(
            f"{report['state']} operators={len(report['operators'])} "
            f"requests={len(report['request_ids'])}"
        )
        for operator in report["operators"]:
            print(f"  {operator}")
        return 0
    if args.action == "list-operators":
        operators = [gap_operator(gap) for gap in collect_gaps(args)]
        operators = [name for name in operators if name]
        print(json.dumps(operators))
        return 0
    if args.action == "shim":
        gap = {"name": args.operator, "class": "TORCH_SHIM", "source": "torch_fallback"}
        report = prepare([gap], args.out, args.subject, args.baseline_id,
                         dispatch_subject=f"{args.subject}-{args.operator}")
        print(f"{report['state']} operator={args.operator}")
        return 0
    if args.action == "aggregate":
        operators: list[str] = []
        request_ids: list[str] = []
        for child in args.child:
            status = child / "fallback_status.json"
            if not status.exists():
                continue
            payload = _load(status)
            operators.extend(payload.get("operators") or [])
            request_ids.extend(payload.get("request_ids") or [])
        operators = list(dict.fromkeys(operators))
        request_ids = list(dict.fromkeys(request_ids))
        report = {
            "state": "FALLBACK_APPLIED" if operators else "FALLBACK_SKIPPED",
            "subject": args.subject,
            "operators": operators,
            "request_ids": request_ids,
            "children": [str(child) for child in args.child],
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "fallback_status.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{report['state']} operators={len(operators)}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
