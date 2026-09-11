"""CLI contract tests for the unified adaptation entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "run_adaptation.py"


def _run(state: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--state", str(state), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_create_discover_claim_and_status_emit_json(tmp_path: Path) -> None:
    state = tmp_path / "adaptation.db"
    report = tmp_path / "gaps.json"
    report.write_text(
        json.dumps(
            {
                "plugin_revision": "kunlun-1",
                "entries": [
                    {
                        "operator_id": "missing_op",
                        "inputs": [
                            {"name": "x", "dtype": "float32", "shape": [1], "layout": "contiguous"}
                        ],
                        "outputs": [
                            {"name": "y", "dtype": "float32", "shape": [1], "layout": "contiguous"}
                        ],
                        "semantics": {"reference": "torch.add"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    created = _run(
        state,
        "create",
        "--run-id",
        "run-1",
        "--model",
        "DeepSeek-V4.1",
        "--backend",
        "kunlun-p800",
    )
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["run"]["run_id"] == "run-1"

    proof = tmp_path / "environment-status.json"
    proof.write_text(
        json.dumps(
            {
                "state": "ENVIRONMENT_READY",
                "pod": "prepared-pod",
                "checks": {
                    "pod_ready": True,
                    "runtime_importable": True,
                    "code_ready": True,
                    "device_ready": True,
                },
                "artifacts": [
                    "environment_fingerprint.txt",
                    "runtime_import.txt",
                    "code_readiness.json",
                    "device_readiness.json",
                ],
            }
        ),
        encoding="utf-8",
    )
    environment = _run(
        state,
        "environment",
        "--run-id",
        "run-1",
        "--status",
        str(proof),
    )
    assert environment.returncode == 0, environment.stdout
    assert json.loads(environment.stdout)["run"]["status"] == "ENVIRONMENT_READY"

    discovered = _run(state, "discover", "--run-id", "run-1", "--report", str(report))
    assert discovered.returncode == 0, discovered.stdout
    payload = json.loads(discovered.stdout)
    assert len(payload["specs"]) == 1
    assert payload["tasks"][0]["stage"] == "torch"

    claimed = _run(state, "claim", "--worker", "torch-agent", "--stage", "torch")
    assert claimed.returncode == 0, claimed.stdout
    assert json.loads(claimed.stdout)["tasks"][0]["status"] == "running"

    status = _run(state, "status", "--run-id", "run-1", "--events")
    assert status.returncode == 0, status.stdout
    status_payload = json.loads(status.stdout)
    assert status_payload["run"]["model_id"] == "DeepSeek-V4.1"
    assert status_payload["tasks"][0]["status"] == "running"
    assert status_payload["tasks"][0]["lease_token"]
    assert [event["event_type"] for event in status_payload["events"]] == [
        "run_created",
        "environment_bound",
        "operator_discovered",
        "task_created",
        "task_claimed",
    ]

    result = tmp_path / "torch-result.json"
    result.write_text(json.dumps({"status": "pass", "artifact": "reference.py"}), encoding="utf-8")
    completed = _run(
        state,
        "complete",
        "--task-id",
        payload["tasks"][0]["task_id"],
        "--worker",
        "torch-agent",
        "--result",
        str(result),
    )
    assert completed.returncode == 0, completed.stdout
    assert json.loads(completed.stdout)["task"]["status"] == "succeeded"


def test_unknown_run_is_machine_readable_error(tmp_path: Path) -> None:
    result = _run(tmp_path / "state.db", "status", "--run-id", "missing")
    assert result.returncode == 2
    assert json.loads(result.stdout) == {"error": "unknown run: missing", "command": "status"}


def test_discover_requires_environment_proof(tmp_path: Path) -> None:
    state = tmp_path / "adaptation.db"
    report = tmp_path / "gaps.json"
    report.write_text(json.dumps({"entries": []}), encoding="utf-8")
    created = _run(
        state,
        "create",
        "--run-id",
        "run-guard",
        "--model",
        "DeepSeek-V4.1",
        "--backend",
        "kunlun-p800",
    )
    assert created.returncode == 0
    discovered = _run(
        state,
        "discover",
        "--run-id",
        "run-guard",
        "--report",
        str(report),
    )
    assert discovered.returncode == 2
    assert "environment proof is required" in json.loads(discovered.stdout)["error"]


def test_create_run_compatibility_alias(tmp_path: Path) -> None:
    result = _run(
        tmp_path / "state.db",
        "create-run",
        "--run-id",
        "compat",
        "--model",
        "m",
        "--backend",
        "xpu",
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["run"]["run_id"] == "compat"
