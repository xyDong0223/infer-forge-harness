"""Execute MAT-006 failure triage as one orchestrated sequence.

The graph runner stopped here with "operator-driven" because the task is not a
single command: instrument the failing call, rerun the service proof so it
fails under instrumentation, pull the real arguments out of the pod, replay
them in isolation to find the boundary, then restore. Each step already had a
tool; what was missing was the sequence — which meant every kernel failure on
a new model ended in a MANUAL stop.

The sequence is fixed by the contract, not chosen here. The only decision this
runner makes is the verdict, and even that is mechanical: the isolated call
failing too is VENDOR_KERNEL_GAP, passing is RUNTIME_STATE_DEPENDENT. The
independent triage validator has the final word before TRIAGE_READY is written.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import REPO_ROOT

import yaml  # noqa: E402

from adapters import get_hardware, push_snippet  # noqa: E402
from runners.evidence import task_status  # noqa: E402

KunlunP800Adapter = get_hardware()
from validators.triage_validator import validate_triage  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-006-failure-triage" / "task.yaml"
REPLAY_TOOL = REPO_ROOT / "tools" / "probe" / "kernel_ut_replay.py"
TRACE_PATH_IN_POD = "/tmp/kdp_kernel_failure.jsonl"
REPLAY_PATH_IN_POD = "/tmp/kdp_kernel_ut_replay.py"


class PodOps:
    """Everything that touches the pod or a subprocess, in one injectable seam."""

    def __init__(self, adapter: KunlunP800Adapter):
        self.adapter = adapter

    def instrument(self, pod: str, call: str) -> str:
        import subprocess

        result = subprocess.run(
            ["python3", "cli/operators/instrument_kernel_trace.py", "--pod", pod, "--call", call],
            cwd=REPO_ROOT, text=True, capture_output=True,
        )
        return (result.stdout + result.stderr).strip() or "INSTRUMENT_EXIT_%d" % result.returncode

    def restore(self, pod: str) -> str:
        import subprocess

        result = subprocess.run(
            ["python3", "cli/operators/instrument_kernel_trace.py", "--pod", pod, "--restore"],
            cwd=REPO_ROOT, text=True, capture_output=True,
        )
        return (result.stdout + result.stderr).strip()

    def rerun_service(self, pod: str, contract_instance: Path, out: Path) -> dict[str, Any]:
        # The reproof relaunches the server in the same pod. If it wrote the
        # original attempt's server.log, the triage itself would truncate the
        # crash evidence it exists to explain (run glm52-int-w8a8-p800-001,
        # 2026-09-14: the log was truncated 40 s after the crash). The rerun
        # gets its own timestamped log path instead, recorded in its status.
        command = [
            "python3", "cli/deployment/proof.py", str(contract_instance),
            "--execute", "--phase", "service", "--attach-pod", pod,
            "--artifact-dir", str(out),
            "--server-log", self.rerun_server_log(contract_instance),
        ]
        import subprocess

        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        return task_status(result)

    @staticmethod
    def rerun_server_log(contract_instance: Path | None) -> str:
        """The reproof's own log path: the contract's, plus a rerun stamp."""
        base = "/workspace/server.log"
        if contract_instance:
            try:
                contract = yaml.safe_load(contract_instance.read_text(encoding="utf-8"))
                base = contract.get("execution", {}).get("server_log") or base
            except (OSError, ValueError, yaml.YAMLError):
                pass
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{base}.rerun-{stamp}"

    def fetch_trace(self, pod: str) -> str:
        result = self.adapter.exec(pod, f"cat {TRACE_PATH_IN_POD} 2>/dev/null", timeout=60)
        return result.stdout

    def replay_in_pod(self, pod: str, call: str) -> dict[str, Any]:
        """Run the isolation sweep where kunlun_ops actually exists: the pod."""
        command = (
            push_snippet(REPLAY_TOOL, REPLAY_PATH_IN_POD) + " && "
            f"python3 {REPLAY_PATH_IN_POD} --from-trace {TRACE_PATH_IN_POD} "
            f"--kernel {call} --json"
        )
        result = self.adapter.exec(pod, command, timeout=600)
        text = result.stdout + result.stderr
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
        return {"error": text.strip()[-500:] or "replay produced no output"}


class TriageExecutor:
    def __init__(self, pod: str, contract_instance: Path | None, call: str,
                 ops: PodOps, out: Path):
        self.pod = pod
        self.contract_instance = contract_instance
        self.call = call
        self.ops = ops
        self.out = out

    def run(self) -> dict[str, Any]:
        status: dict[str, Any] = {"task_id": "mat-006-failure-triage", "pod": self.pod,
                                  "call": self.call, "state": "TRIAGE_FAILED"}
        try:
            report = self._triage()
        except Exception as error:  # noqa: BLE001 - the failure is the report's input
            status["reason"] = f"{type(error).__name__}: {error}"
            self._write(status_path := self.out / "status.json", status)
            return status
        status.update(report)
        status["state"] = report.get("state", "TRIAGE_FAILED")
        self._write(self.out / "status.json", status)
        return status

    def _triage(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        # 1. Instrument. The wrapper re-raises unchanged, so the failure itself
        #    is not altered — only observed.
        self.ops.instrument(self.pod, self.call)
        try:
            # 2. Make the server fail under instrumentation and capture the
            #    service side of the story.
            if not self.contract_instance:
                raise RuntimeError(
                    "triage reruns the service proof to reproduce the failure: "
                    "pass --contract-instance"
                )
            server = self.ops.rerun_service(self.pod, self.contract_instance,
                                            self.out / "service_rerun")
            error_text = server.get("reason") or "service proof failed under instrumentation"
            # 3. The real arguments of the failing call.
            captured = self.ops.fetch_trace(self.pod)
            (self.out / "kernel_failure_arguments.jsonl").write_text(captured, encoding="utf-8")
            entries = [json.loads(line) for line in captured.splitlines()
                       if line.strip().startswith("{")]
            if not entries:
                return {"state": "TRIAGE_FAILED",
                        "reason": "no captured arguments: the failing call was not reached"}
            # 4. Replay in isolation: same arguments, no server state.
            replay = self.ops.replay_in_pod(self.pod, self.call)
            (self.out / "kernel_unit_test_matrix.txt").write_text(
                json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            # 5. Instrumentation is a debugging state, never a deployment state.
            self.ops.restore(self.pod)

        cases = replay.get("results") or replay.get("cases") or []
        captured_case = next((case for case in cases if case.get("case") == "captured_case"), None)
        if captured_case is None:
            return {"state": "TRIAGE_FAILED",
                    "reason": "the isolation sweep never ran the captured case",
                    "error_text": error_text, "failing_symbol": self.call}
        reproduced = captured_case.get("ok") is False
        verdict = "VENDOR_KERNEL_GAP" if reproduced else "RUNTIME_STATE_DEPENDENT"
        layer = "kunlun_ops_vendor" if reproduced else "vllm_kunlun_plugin"
        report = {
            "state": "TRIAGE_READY",
            "failing_symbol": self.call,
            "error_text": entries[-1].get("error", error_text) if entries else error_text,
            "layer": layer,
            "verdict": verdict,
            "owner": "vendor_with_local_workaround" if reproduced else "us",
            "isolated_reproduction": {
                "attempted": True,
                "reproduced": reproduced,
                "cases": cases,
            },
            "captured_arguments": entries[-1] if entries else {},
            "server_rerun": server,
            "artifacts": ["kernel_failure_arguments.jsonl", "kernel_unit_test_matrix.txt",
                          "triage_report.json", "status.json"],
        }
        # 6. The independent gate decides, not this runner.
        contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
        errors = validate_triage({**report, "state": "TRIAGE_READY"}, contract)
        if errors:
            report["state"] = "TRIAGE_FAILED"
            report["validation_errors"] = errors
        self._write(self.out / "triage_report.json", report)
        return report

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args) -> int:

    pod = args.pod
    if not pod and args.env_status and args.env_status.exists():
        pod = json.loads(args.env_status.read_text(encoding="utf-8")).get("pod")
    if not pod:
        parser.error("a pod is required: pass --pod or --env-status")

    executor = TriageExecutor(pod, args.contract_instance, args.call,
                              PodOps(KunlunP800Adapter()), args.out)
    status = executor.run()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status.get("state") == "TRIAGE_READY" else 1
