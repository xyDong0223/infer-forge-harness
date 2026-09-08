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
from validators.handoff_validator import validate_handoff_package  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-020-vendor-handoff" / "task.yaml"
BINARY_LAYERS = {"kunlun_ops_vendor", "torch_xmlir_vendor"}
OWNER_LABEL = {
    "us": "this project (plugin or xpu variant)",
    "upstream": "vLLM upstream",
    "vendor": "the vendor (binary layer)",
}


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


def build_package(findings_doc: dict) -> dict:
    """Route each finding to an owner and keep the package from reading as finished."""
    findings = findings_doc.get("findings") or []
    by_owner: dict[str, list[str]] = {}
    for finding in findings:
        by_owner.setdefault(str(finding.get("owner")), []).append(str(finding.get("id")))
    return {
        "state": "HANDOFF_READY",
        "form": "package",
        "subject": findings_doc.get("subject"),
        "subject_status": findings_doc.get("subject_status") or {},
        "environment_fingerprint": findings_doc.get("environment_fingerprint"),
        "ruled_out": findings_doc.get("ruled_out") or [],
        "findings": findings,
        "routing": by_owner,
        "observed": sum(1 for f in findings if f.get("kind") == "observed_failure"),
        "unreached": sum(1 for f in findings if f.get("kind") == "unreached_dependency"),
        "file_separately": [f.get("id") for f in findings if f.get("file_separately")],
    }


def render_package(package: dict) -> str:
    status = package.get("subject_status") or {}
    lines = [
        f"# Adaptation handoff — {package['subject']}",
        "",
        f"- environment: `{package['environment_fingerprint']}`",
        f"- serves requests: **{status.get('serves_requests')}**",
        f"- stopped at: {status.get('stopped_at')}",
        f"- findings: {package['observed']} observed on hardware, "
        f"{package['unreached']} named from reading the code and never reached",
        "",
        "## Who owns what",
        "",
    ]
    for owner, ids in sorted(package["routing"].items()):
        lines.append(f"- {OWNER_LABEL.get(owner, owner)}: {', '.join(ids)}")
    if package.get("file_separately"):
        lines += [
            "",
            "Worth filing on its own, independent of this model: "
            + ", ".join(package["file_separately"]),
        ]
    if status.get("reached_with"):
        lines += ["", "## How the run got that far", "", status["reached_with"]]
    lines += ["", "## Already ruled out", ""]
    lines += [f"- {item}" for item in package.get("ruled_out", [])]

    for finding in package["findings"]:
        lines += [
            "",
            f"## {finding['id']} — {finding['symbol']}",
            "",
            f"- layer: `{finding['layer']}` · owner: **{finding['owner']}** · {finding['kind']}",
            f"- site: `{finding['site']}`",
        ]
        for key in ("also_called_at",):
            for site in finding.get(key) or []:
                lines.append(f"  - also: `{site}`")
        if finding.get("error_text"):
            lines += ["", "```", str(finding["error_text"]).strip(), "```"]
        else:
            lines += ["", f"Never reached. Blocked by: {finding.get('blocked_by')}"]
        for label, key in (
            ("Cause", "cause"),
            ("What was established", "finding"),
            ("Who else this hits", "generality"),
            ("Minimal fix", "minimal_fix"),
            ("A fix is accepted against", "verified_against"),
            ("Note", "note"),
        ):
            if finding.get(key):
                lines += ["", f"{label}: {str(finding[key]).strip()}"]
        if finding.get("stand_in"):
            lines += [
                "",
                f"Stand-in in place at `{finding['stand_in']}` — a probe, not a fix.",
            ]
    return "\n".join(lines) + "\n"


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", help="mat-006 triage_report.json (single-ticket form)")
    parser.add_argument("--findings", help="findings yaml (package form)")
    parser.add_argument("--environment", help="fingerprint or its digest; taken from the yaml in package form")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if bool(args.triage) == bool(args.findings):
        print("CONTRACT_INVALID: pass exactly one of --triage or --findings", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    if args.findings:
        doc = yaml.safe_load(Path(args.findings).read_text(encoding="utf-8"))
        if args.environment:
            doc["environment_fingerprint"] = args.environment
        package = build_package(doc)
        (out / "handoff_package.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
        (out / "handoff_package.md").write_text(render_package(package), encoding="utf-8")
        gate = validate_handoff_package(package, contract)
        (out / "handoff_status.json").write_text(
            json.dumps(
                {"state": package["state"], "form": "package", "subject": package["subject"],
                 "validator": {"passed": not gate, "errors": gate}},
                indent=2,
            ),
            encoding="utf-8",
        )
        if gate:
            print("CONTRACT_INVALID: " + "; ".join(gate), file=sys.stderr)
            return 1
        print(
            f"HANDOFF_READY  {package['subject']}  "
            f"observed={package['observed']} unreached={package['unreached']}"
        )
        for owner, ids in sorted(package["routing"].items()):
            print(f"  {owner:<9} {', '.join(ids)}")
        print(f"artifacts: {out}/handoff_package.md")
        return 0

    if not args.environment:
        print("CONTRACT_INVALID: --environment is required in the single-ticket form", file=sys.stderr)
        return 1
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

    ticket = build(triage, args.environment, None)
    (out / "vendor_ticket.json").write_text(json.dumps(ticket, indent=2), encoding="utf-8")
    (out / "vendor_ticket.md").write_text(render(ticket), encoding="utf-8")
    gate = validate_handoff(ticket, contract)
    (out / "handoff_status.json").write_text(
        json.dumps({"state": ticket["state"], "form": "ticket",
                    "validator": {"passed": not gate, "errors": gate}}, indent=2),
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
