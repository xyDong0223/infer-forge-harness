"""Preserve pre-plan failures without inventing a service or kernel attribution."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.storage import ArtifactStore, ensure_external, locate_attempt
from validators.triage_validator import validate_failure_evidence


def collect_failure(status_path: Path, out: Path) -> dict:
    status_path = ensure_external(status_path)
    source = json.loads(status_path.read_text())
    attempt = locate_attempt(status_path)
    files = [(status_path, "failure_evidence/source_status.json")]
    if attempt is not None:
        files.extend((path, "failure_evidence/logs/" + path.relative_to(attempt.logs).as_posix())
                     for path in sorted(attempt.logs.rglob("*.log")) if path.is_file())
    store = ArtifactStore(out)
    evidence = []
    for path, relative in files:
        data = path.read_bytes()
        target = store.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
        evidence.append({"path": relative, "source": str(path),
                         "sha256": hashlib.sha256(data).hexdigest()})
    report = {
        "task_id": "mat-006-failure-triage", "state": "NEEDS_HUMAN",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "UNKNOWN", "next_action": "DIAGNOSE_TASK",
        "reason": "The upstream task failed before a valid DeploymentPlan exists. "
                  "Inspect the preserved status and command logs before choosing a repair.",
        "source_status": source, "source_path": str(status_path),
        "source_workspace": attempt.identity if attempt else None,
        "evidence": evidence,
        "artifacts": [entry["path"] for entry in evidence] + ["triage_report.json", "status.json"],
    }
    errors = validate_failure_evidence(report, out)
    report["validator"] = {"passed": not errors, "errors": errors}
    if errors:
        report.update(state="TRIAGE_FAILED", reason="; ".join(errors))
    store.write_json("triage_report.json", report)
    store.write_json("status.json", report)
    return report
