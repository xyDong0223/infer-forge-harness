"""MAT-004 Gap Classification: turn two static findings into a next action.

Consumes MAT-002's support verdict and MAT-003's capability match. Produces a
class, because different classes have completely different next steps: a missing
registration means writing a model, a version lag means upgrading or
cherry-picking, a missing capability means an operator, and finding nothing
statically means the answer is only obtainable by running the thing.

That last class is the one worth naming. Qwen3-8B has no static gap at all and
still fails at runtime, so `NO_STATIC_GAP` explicitly routes to a deployment
attempt and then to triage — it is not a claim that the model works.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validators.gap_validator import validate_gap_classification  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-004-gap-classification" / "task.yaml"

# Scan verdict -> (class, next action). Ordered by how much work the answer implies.
SCAN_CLASSES = {
    "ABSENT": ("REGISTRATION_MISSING", "write a model implementation"),
    "PR_PENDING": ("VERSION_LAG", "cherry-pick the open pull request or wait for merge"),
    "MAIN_ONLY": ("VERSION_LAG", "upgrade the installed vLLM or cherry-pick from main"),
    "UNKNOWN_UPSTREAM": ("UNVERIFIED", "reach upstream and re-run the scan before deciding"),
    "KUNLUN_OOT": (None, None),
    "UPSTREAM_GENERIC": (None, None),
}


def classify(support: dict, match: dict) -> dict:
    gaps: list[dict] = []

    for entry in support.get("results", []):
        verdict = entry.get("verdict")
        gap_class, action = SCAN_CLASSES.get(verdict, ("UNVERIFIED", "re-run the scan"))
        if gap_class:
            gaps.append(
                {
                    "class": gap_class,
                    "axis": "networking",
                    "detail": f"{entry.get('architecture')} is {verdict}",
                    "next_action": action,
                    "owner": "us",
                    "evidence": "mat-002 model_support.json",
                }
            )

    for axis in match.get("axes", []):
        if axis.get("verdict") == "NOT_PROVIDED":
            gaps.append(
                {
                    "class": "CAPABILITY_MISSING",
                    "axis": axis["axis"],
                    "detail": f"{axis['axis']} requires {axis.get('required')} and the installation "
                    "does not provide it",
                    "next_action": f"implement or route around the {axis['axis']} path",
                    "owner": "us",
                    "evidence": "mat-003 capability_match.json",
                }
            )
        elif axis.get("verdict") == "UNKNOWN":
            gaps.append(
                {
                    "class": "UNVERIFIED",
                    "axis": axis["axis"],
                    "detail": f"{axis['axis']} was not introspected",
                    "next_action": f"introspect the {axis['axis']} surface before relying on it",
                    "owner": "us",
                    "evidence": "mat-003 capability_match.json",
                }
            )

    blocking = [gap for gap in gaps if gap["class"] in ("REGISTRATION_MISSING", "CAPABILITY_MISSING")]
    if not gaps:
        classification = "NO_STATIC_GAP"
        action = "attempt a deployment; a failure then belongs to MAT-006 triage, not to networking"
    elif blocking:
        classification = blocking[0]["class"]
        action = blocking[0]["next_action"]
    else:
        classification = gaps[0]["class"]
        action = gaps[0]["next_action"]

    return {
        "state": "CLASSIFICATION_READY",
        # Static only. Named in the payload so no consumer can read a class as a
        # statement about runtime behaviour.
        "runtime_verified": False,
        "classification": classification,
        "next_action": action,
        "gaps": gaps,
        "blocking": [gap["axis"] for gap in blocking],
        "inputs": {
            "scan_state": support.get("state"),
            "match_verdict": match.get("verdict"),
            "scanned_in": support.get("scanned_in"),
            "matched_in": match.get("matched_in"),
        },
    }


def render(report: dict, subject: str) -> str:
    lines = [
        f"# Gap classification — {subject}",
        "",
        f"- classification: **{report['classification']}**",
        f"- next action: {report['next_action']}",
        f"- runtime_verified: {report['runtime_verified']}",
        "",
    ]
    if report["gaps"]:
        lines += ["| Class | Axis | Detail | Next action |", "| --- | --- | --- | --- |"]
        lines += [
            f"| {gap['class']} | {gap['axis']} | {gap['detail']} | {gap['next_action']} |"
            for gap in report["gaps"]
        ]
    else:
        lines.append("No static gap was found. That is not a statement that the model runs:")
        lines.append("it means nothing further can be learned without deploying it.")
    return "\n".join(lines) + "\n"


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-card", required=True, help="mat-002 model_support.json")
    parser.add_argument("--capability-match", required=True, help="mat-003 capability_match.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    support = json.loads(Path(args.support_card).read_text(encoding="utf-8"))
    match = json.loads(Path(args.capability_match).read_text(encoding="utf-8"))
    for name, payload, wanted in (
        ("model scan", support, "SCAN_READY"),
        ("capability match", match, "MATCH_READY"),
    ):
        if payload.get("state") != wanted:
            print(f"NEEDS_HUMAN: the {name} is {payload.get('state')}, not {wanted}", file=sys.stderr)
            return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = classify(support, match)
    subject = (match.get("model") or {}).get("id") or (support.get("model") or {}).get("id") or "unknown"
    report["subject"] = subject
    (out / "gap_classification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "gap_classification.md").write_text(render(report, subject), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_gap_classification(report, contract)
    (out / "classification_status.json").write_text(
        json.dumps({"state": report["state"], "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    if gate:
        print("CONTRACT_INVALID: " + "; ".join(gate), file=sys.stderr)
        return 1
    print(f"{report['classification']}  ->  {report['next_action']}")
    for gap in report["gaps"]:
        print(f"  {gap['class']:22} {gap['axis']:14} {gap['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
