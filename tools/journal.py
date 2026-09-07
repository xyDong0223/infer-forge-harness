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

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_JOURNAL = Path("/ssd1/dongxinyu03/kunlun-agent-artifacts/journal.jsonl")
# Which produced fact each task_type contributes, mirroring the contracts'
# `spec.produces`.
KINDS = {
    "model_intake": "ModelRequest",
    "environment_proof": "EnvironmentProof",
    "model_scan": "ModelSupportCard",
    "capability_match": "CapabilityMatch",
    "gap_classification": "GapClassification",
    "deployment_plan": "DeploymentPlan",
    "service_proof": "DeploymentProof",
    "memory_budget": "MemoryBudget",
    "failure_triage": "FailureTriage",
    "patch_placement": "PlacedPatch",
    "vendor_handoff": "VendorHandoff",
}


def fingerprint(environment: dict[str, str]) -> str:
    """Stable digest of the facts a conclusion is only true of."""
    canonical = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def record(
    journal: Path,
    kind: str,
    subject: str,
    state: str,
    artifacts: Path,
    environment: dict[str, str],
    extra: dict | None = None,
) -> dict:
    entry = {
        "kind": kind,
        "subject": subject,
        "state": state,
        "artifacts": str(artifacts),
        "environment": environment,
        "fingerprint": fingerprint(environment),
    }
    if extra:
        entry["detail"] = extra
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load(journal: Path) -> list[dict]:
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]


def query(
    journal: Path,
    kind: str,
    subject: str | None = None,
    environment: dict[str, str] | None = None,
    states: tuple[str, ...] = (),
) -> list[dict]:
    """Most recent first. An environment argument restricts to matching facts.

    A fact recorded under a different fingerprint is not returned even when the
    subject matches: reusing it would answer a question about this environment
    with evidence from another one.
    """
    wanted = fingerprint(environment) if environment else None
    hits = [
        entry
        for entry in load(journal)
        if entry["kind"] == kind
        and (subject is None or entry["subject"] == subject)
        and (wanted is None or entry["fingerprint"] == wanted)
        and (not states or entry["state"] in states)
    ]
    return list(reversed(hits))


def latest(journal: Path, kind: str, **kwargs) -> dict | None:
    hits = query(journal, kind, **kwargs)
    return hits[0] if hits else None


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

    def environment(pairs: list[str]) -> dict[str, str]:
        return dict(pair.split("=", 1) for pair in pairs)

    if args.command == "record":
        print(json.dumps(record(args.journal, args.kind, args.subject, args.state,
                                args.artifacts, environment(args.env)), indent=2))
        return 0
    if args.command == "query":
        hits = query(args.journal, args.kind, args.subject,
                     environment(args.env) or None, tuple(args.state))
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0 if hits else 1
    for entry in load(args.journal):
        print(f"{entry['state']:22} {entry['kind']:18} {entry['subject']:34} {entry['artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
