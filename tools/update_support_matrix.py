"""MAT-016 Support Matrix: change the claim only when evidence changes.

The matrix is the one artifact people read instead of reading evidence, so the
entry has to carry what it is true of: which checkpoint revision, which stack
commit, and which conditions the result depended on. An entry that says
"supported" without the launch conditions is how a table becomes something nobody
trusts.

Nothing is inferred. The status comes from recorded facts — a deployment proof and
an accuracy differential — and a missing fact keeps the status where it was.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import journal as journal_module  # noqa: E402
from validators.matrix_validator import validate_matrix_entry  # noqa: E402

MATRIX = REPO_ROOT / "catalog" / "support_matrix.yaml"
CONTRACT = REPO_ROOT / "tasks" / "mat-016-support-matrix" / "task.yaml"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_entry(subject: str, hardware: str, deployment: dict, accuracy: dict,
                budget: dict, plan: dict) -> dict:
    served = deployment.get("state") == "DEPLOYMENT_READY"
    correct = accuracy.get("status") == "ACCURACY_PASS"
    if served and correct:
        status = "validated"
    elif served:
        status = "runs_unverified_accuracy"
    else:
        status = "not_yet_validated"

    conditions = []
    for item in plan.get("parameters") or []:
        if item["parameter"] in ("dtype", "tensor_parallel_size", "max_model_len", "enforce_eager"):
            conditions.append(f"{item['parameter']}={item['value']}")
    limitations = []
    if any(item["parameter"] == "enforce_eager" and item["value"] for item in plan.get("parameters") or []):
        limitations.append(
            "eager execution only: the decode fallback is shape-dynamic and cannot be captured in a "
            "FULL graph"
        )
    if accuracy.get("cases") is not None and len(accuracy.get("cases") or []) < 10:
        limitations.append(
            f"accuracy is a {len(accuracy.get('cases') or [])}-prompt differential, not a dataset score"
        )

    return {
        "model": subject,
        "hardware": hardware,
        "status": status,
        # What the claim is true of. Without these the row is not falsifiable.
        "revision": (accuracy.get("revision") or plan.get("revision")),
        "stack_commit": plan.get("stack_commit"),
        "conditions": conditions,
        "evidence": {
            "deployment_proof": deployment.get("state"),
            "accuracy_differential": accuracy.get("status"),
            "memory_budget": budget.get("state"),
        },
        "limitations": limitations,
    }


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--hardware", default="Kunlunxin-3-P800")
    parser.add_argument("--deployment-status", type=Path, help="service proof status.json")
    parser.add_argument("--accuracy", type=Path, help="mat-013 accuracy_differential.json")
    parser.add_argument("--budget-status", type=Path, help="mem-001 budget_status.json")
    parser.add_argument("--plan", type=Path, help="mat-005 deployment_plan.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--apply", action="store_true", help="write catalog/support_matrix.yaml")
    args = parser.parse_args()

    entry = build_entry(
        args.subject,
        args.hardware,
        read_json(args.deployment_status) if args.deployment_status else {},
        read_json(args.accuracy) if args.accuracy else {},
        read_json(args.budget_status) if args.budget_status else {},
        read_json(args.plan) if args.plan else {},
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_matrix_entry(entry, contract)
    (out / "matrix_entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    (out / "matrix_status.json").write_text(
        json.dumps({"state": "MATRIX_READY" if not gate else "MATRIX_REJECTED",
                    "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    print(f"{entry['model']} / {entry['hardware']}: {entry['status']}")
    for condition in entry["conditions"]:
        print(f"  condition: {condition}")
    for limitation in entry["limitations"]:
        print(f"  limitation: {limitation}")
    if gate:
        print("MATRIX_REJECTED: " + "; ".join(gate), file=sys.stderr)
        return 1

    if args.apply:
        matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
        entries = [
            existing
            for existing in matrix.get("entries", [])
            if not (existing.get("model") == entry["model"] and existing.get("hardware") == entry["hardware"])
        ]
        entries.append(entry)
        matrix["entries"] = entries
        MATRIX.write_text(yaml.safe_dump(matrix, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"applied to {MATRIX.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
