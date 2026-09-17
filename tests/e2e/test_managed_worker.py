"""Managed-v2 production CLI: measured completion, rejection and interrupted submit.

Only the implementation producer and target runtime are local simulation fixtures.
The independent probe, scheduler, leases, frozen tree and receipt gates are real.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from core.paths import REPO_ROOT
from core.storage import ArtifactStore
from engine.result_validation import STAGE_EVIDENCE
from tests.e2e.test_model_adaptation import source_snapshot


pytestmark = pytest.mark.local_e2e
CANDIDATE = "def run_case(inputs):\n    return {'y': [value * 2.0 for value in inputs['x']]}\n"
REFERENCE = "def run_case(inputs):\n    return {'y': [value + value for value in inputs['x']]}\n"


class ManagedScenario:
    def __init__(self, root):
        self.root = root
        self.state = root / "state.sqlite"
        self.run_id = "managed-worker-simulation"
        self.store = ArtifactStore(root)
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                    "INFER_FORGE_STATE_ROOT": str(root / "state")}
        self.step = 0
        reference = self.store.write_text("oracle/reference.py", REFERENCE)
        geometry = {"shape": [4], "dtype": "float64", "layout": "contiguous"}
        contract = {
            "schema_version": 1, "candidate_entry": "candidate.py",
            "reference": {"entry": str(reference), "files": {
                str(reference): hashlib.sha256(reference.read_bytes()).hexdigest()},
                "provenance": "Independently written addition oracle for multiplication fixture"},
            "cases": [{"id": "signed", "inputs": {"x": {
                **geometry, "values": [-1.0, 0.0, 1.0, 2.0]}}, "outputs": {"y": geometry}}],
            "thresholds": {"y": {"max_relative_l2": 1e-8}},
            "negative_control": {"kind": "offset", "value": 1.0},
            "expected_dispatch": {"symbol": "run_case", "device": "simulation-cpu", "ranks": [0]},
            "fallback": {"allowed_devices": ["simulation-cpu"]},
            "service": {"require_http_status": 200},
        }
        metadata = self.store.write_json("metadata.json", {
            "evidence_mode": "simulation", "environment_required": False})
        self.cli("create-run", "--run-id", self.run_id, "--model", "measured-local-scale",
                 "--model-revision", "fixture-model-v1", "--plugin-revision", "fixture-plugin-v1",
                 "--backend", "simulation-cpu", "--metadata", str(metadata),
                 "--artifact-root", str(root / "run"), "--worker-protocol", "managed-v2")
        report = self.store.write_json("observed-gap.json", {"entries": [{
            "operator_id": "scale", "inputs": [{"name": "x", **geometry}],
            "outputs": [{"name": "y", **geometry}],
            "semantics": {"formula": "y = 2 * x", "validation": contract},
            "evidence": {"mode": "simulation", "call_site": "candidate.run_case"},
        }]})
        self.cli("discover", "--run-id", self.run_id, "--report", str(report))

    def argv(self, *args):
        return [sys.executable, str(REPO_ROOT / "cli/adaptation.py"),
                "--state", str(self.state), *args]

    def process(self, argv, expected=0):
        result = subprocess.run(argv, cwd=self.root, env=self.env, capture_output=True,
                                text=True, check=False, timeout=45)
        self.step += 1
        path = self.store.write_text(f"driver/{self.step:03d}.log",
                                     f"exit={result.returncode}\n{result.stdout}\n{result.stderr}")
        if expected is not None:
            assert result.returncode == expected, f"{path}\n{result.stdout}\n{result.stderr}"
        return result

    def cli(self, *args, expected=0):
        return json.loads(self.process(self.argv(*args), expected).stdout)

    def claim(self, stage="torch"):
        tasks = self.cli("claim", "--run-id", self.run_id, "--worker", "controller",
                         "--stage", stage, "--limit", "1")["tasks"]
        assert len(tasks) == 1
        return tasks[0]

    def credentials(self, task):
        return ["--task-id", task["task_id"], "--worker", "controller",
                f"--lease-token={task['lease_token']}"]

    def produce(self, task, source=CANDIDATE):
        output = ArtifactStore(task["input"]["workspace"]["output"])
        candidate = output.write_text("candidate/candidate.py", source)
        # External implementation Agent replacement compiles and actually runs its
        # candidate. It emits raw observations, not an acceptance/PASS report.
        program = (
            "import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); "
            "ns={'__file__':str(p),'__name__':'fixture_producer'}; exec(compile(p.read_text(),str(p),'exec'),ns); "
            "print(json.dumps({'evidence_mode':'simulation','entry':str(p),"
            "'observed':ns['run_case']({'x':[-1.0,0.0,1.0,2.0]}),"
            "'registered_symbol':ns['run_case'].__name__}))"
        )
        measured = self.process([sys.executable, "-c", program, str(candidate)])
        producer_output = output.write_text("producer-observations.json", measured.stdout)
        evidence = {
            kind: str(candidate if kind == "reference_artifact" else producer_output)
            for kind in STAGE_EVIDENCE[task["stage"]] if kind != "independent_validation"
        }
        evidence_file = output.write_json("producer-evidence.json", evidence)
        frozen = self.cli("freeze-candidate", *self.credentials(task),
                          "--producer", "implementation-agent", "--candidate-root", str(candidate.parent),
                          "--base-revision", "fixture-plugin-v1")["candidate"]
        return frozen, evidence_file

    def validation_args(self, task, candidate, evidence, validation_id=None):
        return ["validate-worker", *self.credentials(task), "--validator", "independent-validator",
                "--candidate-id", candidate["candidate_id"], "--validation-id",
                validation_id or f"validation-{task['stage']}", "--evidence", str(evidence),
                "--timeout", "20"]

    def complete(self, task, result_path, expected=0):
        return self.cli("complete", *self.credentials(task), "--result", str(result_path), expected=expected)

    def status(self):
        return self.cli("status", "--run-id", self.run_id, "--events")


@pytest.fixture
def managed_scenario(tmp_path, request):
    before = source_snapshot()
    case = ManagedScenario(tmp_path)
    request.node.user_properties.append(("artifact_root", str(tmp_path)))
    yield case
    case.store.register(identity={"scenario": "managed_worker", "case": request.node.name},
                        outcome="RECORDED")
    assert source_snapshot() == before


def test_managed_worker_three_stage_measured_completion(managed_scenario):
    case = managed_scenario
    for stage in ("torch", "xpu", "integration"):
        task = case.claim(stage)
        candidate, evidence = case.produce(task)
        validation = case.cli(*case.validation_args(task, candidate, evidence))
        assert validation["status"] == "PASS"
        assert validation["evidence_mode"] == "simulation"
        result = validation["result"]
        raw = json.loads(Path(result["evidence"]["measurement_candidate"]).read_text())
        assert raw["cases"][0]["outputs"]["y"]["values"] == [-2.0, 0.0, 2.0, 4.0]
        if stage == "integration":
            service = raw["cases"][0]["service"]
            assert service["http_status"] == 200
            assert service["transport"] == "http-loopback-simulation"
        assert case.complete(task, validation["result_path"])["task"]["status"] == "succeeded"
    status = case.status()
    assert [task["status"] for task in status["tasks"]] == ["succeeded"] * 3
    assert len(status["run"]["metadata"]["execution_records"]) == 9
    assert all(record["termination_confirmed"] and record["state"] == "SUCCEEDED"
               for record in status["run"]["metadata"]["execution_records"].values())


def test_managed_worker_wrong_candidate_is_rejected_without_execution(managed_scenario):
    case = managed_scenario
    task = case.claim()
    candidate, evidence = case.produce(task)
    wrong = {**candidate, "candidate_id": "not-the-frozen-candidate"}
    rejected = case.cli(*case.validation_args(task, wrong, evidence), expected=2)
    assert "candidate" in rejected["error"].lower()
    metadata = case.status()["run"]["metadata"]
    assert not metadata.get("execution_records")
    assert not metadata.get("managed_validations")
    # Rejection did not consume the original assignment or invent a new attempt.
    accepted = case.cli(*case.validation_args(task, candidate, evidence))
    assert case.complete(task, accepted["result_path"])["task"]["status"] == "succeeded"


def test_managed_worker_forged_receipt_does_not_promote(managed_scenario):
    case = managed_scenario
    task = case.claim()
    candidate, evidence = case.produce(task)
    validation = case.cli(*case.validation_args(task, candidate, evidence))
    forged = deepcopy(validation["result"])
    forged["managed_validation_id"] = "producer-invented-receipt"
    path = ArtifactStore(task["input"]["workspace"]["output"]).write_json("forged-result.json", forged)
    rejected = case.complete(task, path, expected=6)
    assert rejected["task"]["status"] == "failed"
    assert "receipt" in json.dumps(rejected).lower()
    assert not [item for item in case.status()["tasks"] if item["stage"] == "xpu"]


def test_managed_worker_wrong_numerics_fail_despite_successful_processes(managed_scenario):
    case = managed_scenario
    task = case.claim()
    candidate, evidence = case.produce(task, CANDIDATE.replace("2.0", "3.0"))
    validation = case.cli(*case.validation_args(task, candidate, evidence), expected=6)
    assert validation["status"] == "FAIL"
    assert validation["receipt"]["state"] == "failed"
    executions = case.status()["run"]["metadata"]["execution_records"]
    assert len(executions) == 3 and all(value["state"] == "SUCCEEDED" for value in executions.values())
    assert not [item for item in case.status()["tasks"] if item["stage"] == "xpu"]


def test_managed_worker_crash_after_validation_replays_before_submit(managed_scenario):
    case = managed_scenario
    task = case.claim()
    candidate, evidence = case.produce(task)
    arguments = case.validation_args(task, candidate, evidence)
    # The controlling Agent process really exits abruptly after receiving the
    # durable measured receipt, before it can submit complete. No hooks or DB edits.
    program = (
        "import os,subprocess,sys; p=subprocess.run(sys.argv[1:],capture_output=True,text=True); "
        "sys.stdout.write(p.stdout); sys.stderr.write(p.stderr); "
        "sys.stdout.flush(); sys.stderr.flush(); os._exit(17 if p.returncode == 0 else 18)"
    )
    interrupted = case.process([sys.executable, "-c", program, *case.argv(*arguments)], expected=17)
    validated = json.loads(interrupted.stdout)
    before = case.status()["run"]["metadata"]["execution_records"]
    replay = case.cli(*arguments)
    assert replay["replayed"] is True
    assert replay["result"] == validated["result"]
    assert case.status()["run"]["metadata"]["execution_records"] == before
    done = case.complete(task, validated["result_path"])
    assert done["task"]["status"] == "succeeded"
    assert len(case.status()["run"]["metadata"]["execution_records"]) == 3


def test_managed_worker_dead_coordinator_requires_process_and_receipt_reconciliation(managed_scenario):
    case = managed_scenario
    task = case.claim()
    marker = Path(task["input"]["workspace"]["scratch"]) / "probe-started.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    source = (
        "import json, os, time\nfrom pathlib import Path\n"
        "def run_case(inputs):\n"
        "    if __name__ == 'infer_forge_frozen_entry':\n"
        f"        Path({str(marker)!r}).write_text(json.dumps({{'pid': os.getpid()}}))\n"
        "        time.sleep(2.0)\n"
        "    return {'y': [value * 2.0 for value in inputs['x']]}\n"
    )
    candidate, evidence = case.produce(task, source)
    validation_id = "interrupted-controller"
    args = case.validation_args(task, candidate, evidence, validation_id)
    log = case.root / "interrupted-controller.log"
    with log.open("w") as stream:
        controller = subprocess.Popen(case.argv(*args), cwd=case.root, env=case.env,
                                      stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 12
            while not marker.exists() and time.monotonic() < deadline and controller.poll() is None:
                time.sleep(0.02)
            assert marker.exists(), log.read_text()
            # Wait for the durable birth record as well as the child-side marker;
            # killing in the spawn-before-record gap is a different UNKNOWN case.
            records = case.cli("execution-status", "--run-id", case.run_id)["executions"]
            assert len(records) == 1 and records[0]["state"] == "RUNNING", records
            assert records[0]["pid"] == json.loads(marker.read_text())["pid"]
            # The child is in its own managed group, so killing the coordinator
            # does not certify termination or free the task's resource occupancy.
            controller.kill()
            controller.wait(timeout=5)
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait(timeout=5)

    pending = case.cli("reconcile-validation", "--run-id", case.run_id,
                       "--validation-id", validation_id, expected=6)
    assert pending["receipt"]["state"] == "executing"
    case.cli("renew-lease", *case.credentials(task), "--lease-seconds", "0.05")
    time.sleep(0.08)
    assert case.cli("claim", "--run-id", case.run_id, "--worker", "replacement",
                    "--stage", "torch", "--limit", "1")["tasks"] == []

    execution_id = f"validation:{validation_id}:candidate"
    deadline = time.monotonic() + 12
    observed = None
    while time.monotonic() < deadline:
        observed = case.cli("reconcile-execution", "--run-id", case.run_id,
                            "--execution-id", execution_id, expected=None)
        if observed["execution"]["termination_confirmed"]:
            break
        time.sleep(0.1)
    assert observed and observed["execution"]["termination_confirmed"], observed
    assert observed["execution"]["state"] == "FAILED"
    assert observed["execution"]["pid"] == json.loads(marker.read_text())["pid"]
    reconciled = case.cli("reconcile-validation", "--run-id", case.run_id,
                          "--validation-id", validation_id)
    assert reconciled["receipt"]["state"] == "failed"
    assert reconciled["receipt"]["result"] is None
    assert reconciled["receipt"]["reconciliation"]["observation"]["controller_absent"] is True

    replacement = case.claim()
    assert replacement["attempt"] == task["attempt"] + 1
    assert replacement["input"]["workspace"]["root"] != task["input"]["workspace"]["root"]
    assert marker.is_file() and Path(candidate["candidate_root"]).is_dir()
    new_candidate, new_evidence = case.produce(replacement)
    validation = case.cli(*case.validation_args(replacement, new_candidate, new_evidence, "recovered-controller"))
    assert case.complete(replacement, validation["result_path"])["task"]["status"] == "succeeded"
