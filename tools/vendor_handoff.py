"""MAT-020 Vendor Handoff: produce a ticket the vendor can act on.

A terminal state, not a failure. When a triage lands on a binary layer there is
nothing further to fix here, and the deliverable is a package someone else can
work from: what was called, with which arguments, in which environment, what was
expected, what happened, and — the part usually missing — whether it reproduces
outside the server.

That last field is why this tool exists rather than a wiki page. Qwen3-8B does not
reproduce in isolation, and a ticket that implies it does wastes the vendor's time
and comes back rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validators.handoff_validator import validate_handoff  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-020-vendor-handoff" / "task.yaml"
BINARY_LAYERS = {"kunlun_ops_vendor", "torch_xmlir_vendor"}


def build(triage: dict, environment: str, workaround: dict | None) -> dict:
    isolated = triage.get("isolated_reproduction") or {}
    reproduces = bool(isolated.get("reproduced"))
    return {
        "state": "HANDOFF_READY",
        "layer": triage.get("layer"),
        "symbol": triage.get("failing_symbol"),
        "error_text": triage.get("error_text"),
        "arguments": triage.get("captured_arguments"),
        "environment_fingerprint": environment,
        "reproduces_in_isolation": reproduces,
        # Stated in the ticket rather than left to the reader: without it the
        # vendor starts by trying to reproduce standalone and closes the ticket.
        "reproduction": (
            "the captured arguments fail when called directly"
            if reproduces
            else "the identical arguments succeed when called directly; the failure only occurs "
            "inside the server process, so reproduction requires the full serving run"
        ),
        "expected": f"{triage.get('failing_symbol')} returns 0 for the recorded arguments",
        "observed": triage.get("error_text"),
        "ruled_out": (isolated.get("hard_constraints") or []) + [
            f"{isolated.get('passed', 0)} of {isolated.get('cases', 0)} swept cases pass in isolation"
        ],
        "local_workaround": (workaround or triage.get("workaround") or {}).get("description"),
        "owner": "vendor",
    }


def render(ticket: dict) -> str:
    lines = [
        f"# Vendor handoff — {ticket['symbol']}",
        "",
        f"- layer: `{ticket['layer']}`",
        f"- environment: `{ticket['environment_fingerprint']}`",
        f"- reproduces in isolation: **{ticket['reproduces_in_isolation']}**",
        "",
        "## What happens",
        "",
        f"```\n{ticket['observed']}\n```",
        "",
        f"Expected: {ticket['expected']}",
        "",
        "## How to reproduce",
        "",
        ticket["reproduction"],
        "",
        "## Arguments recorded at the failure",
        "",
        "```json",
        json.dumps(ticket["arguments"], indent=2),
        "```",
        "",
        "## Already ruled out",
        "",
    ]
    lines += [f"- {item}" for item in ticket["ruled_out"]]
    if ticket.get("local_workaround"):
        lines += ["", "## Local workaround in place", "", ticket["local_workaround"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", required=True, help="mat-006 triage_report.json")
    parser.add_argument("--environment", required=True, help="fingerprint or its digest")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    triage = json.loads(Path(args.triage).read_text(encoding="utf-8"))
    if triage.get("state") != "TRIAGE_READY":
        print(f"NEEDS_HUMAN: triage is {triage.get('state')}", file=sys.stderr)
        return 1
    if triage.get("layer") not in BINARY_LAYERS:
        print(
            f"CONTRACT_INVALID: {triage.get('layer')} is modifiable here, so this is not a handoff",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ticket = build(triage, args.environment, None)
    (out / "vendor_ticket.json").write_text(json.dumps(ticket, indent=2), encoding="utf-8")
    (out / "vendor_ticket.md").write_text(render(ticket), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_handoff(ticket, contract)
    (out / "handoff_status.json").write_text(
        json.dumps({"state": ticket["state"], "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    if gate:
        print("CONTRACT_INVALID: " + "; ".join(gate), file=sys.stderr)
        return 1
    print(f"HANDOFF_READY  {ticket['symbol']}  reproduces_in_isolation={ticket['reproduces_in_isolation']}")
    print(f"artifacts: {out}/vendor_ticket.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
