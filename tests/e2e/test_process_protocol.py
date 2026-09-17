"""Process protocol used as a local operator fixture, not a remote task bypass."""

import json
from pathlib import Path
import sys
import time

import pytest

from core.paths import REPO_ROOT
from core.storage import ArtifactStore
from tests.e2e.test_managed_worker import ManagedScenario
from tests.e2e.test_model_adaptation import source_snapshot


pytestmark = pytest.mark.local_e2e
HELPER = REPO_ROOT / "cli/deployment/managed_process.py"


@pytest.fixture
def process_scenario(tmp_path, request):
    before = source_snapshot()
    case = ManagedScenario(tmp_path)
    request.node.user_properties.append(("artifact_root", str(tmp_path)))
    yield case
    case.store.register(identity={"scenario": "process_protocol", "case": request.node.name},
                        outcome="RECORDED")
    assert source_snapshot() == before


def invoke(case, action, root, request, *, expected=0):
    return json.loads(case.process([sys.executable, str(HELPER), action,
                                   "--root", str(root), "--request", str(request)], expected).stdout)


def await_terminal(case, root, request):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        observed = invoke(case, "inspect", root, request)
        if observed["terminal"]:
            return observed
        time.sleep(0.05)
    invoke(case, "cancel", root, request)
    pytest.fail(f"protocol did not settle: {observed}")


def request_file(case, task, program):
    output = ArtifactStore(task["input"]["workspace"]["output"])
    request = output.write_json("request.json", {
        "schema_version": 1, "execution_id": "fixture-child", "nonce": "fixture-unique-nonce-001",
        "argv": [sys.executable, "-c", program],
        "cwd": str(case.root.resolve()), "timeout_seconds": 10,
    })
    return request, output.path("process")


def test_process_protocol_measured_worker_completion(process_scenario):
    case = process_scenario
    task = case.claim()
    output = str(Path(task["input"]["workspace"]["output"]).resolve())
    # The simulated external operator really computes in a detached subprocess.
    # Only the production independent validator may decide the torch task passes.
    source = f'''import json, subprocess, sys, time, uuid
from pathlib import Path
def run_case(inputs):
    root = Path({output!r}) / ('protocol-' + uuid.uuid4().hex)
    request = root.with_suffix('.request.json')
    request.write_text(json.dumps({{
        'schema_version': 1, 'execution_id': root.name, 'nonce': uuid.uuid4().hex,
        'argv': [sys.executable, '-c',
                 "import json,sys; print(json.dumps({{'y':[2.0*x for x in json.loads(sys.argv[1])]}}))",
                 json.dumps(inputs['x'])],
        'cwd': {output!r}, 'timeout_seconds': 10}}))
    def call(action):
        result = subprocess.run([sys.executable, {str(HELPER)!r}, action,
                                 '--root', str(root), '--request', str(request)],
                                check=True, capture_output=True, text=True, timeout=10)
        return json.loads(result.stdout)
    call('start')
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        observed = call('inspect')
        if observed['terminal']:
            assert observed['state'] == 'EXITED' and observed['returncode'] == 0
            assert observed['task_verdict'] is None and not observed['ledger_integrated']
            return json.loads(Path(observed['log_path']).read_text())
        time.sleep(0.05)
    call('cancel')
    raise RuntimeError('process protocol did not terminate')
'''
    candidate, evidence = case.produce(task, source)
    assert case.status()["tasks"][0]["status"] == "running"
    measured = case.cli(*case.validation_args(task, candidate, evidence))
    assert measured["status"] == "PASS" and measured["evidence_mode"] == "simulation"
    assert case.complete(task, measured["result_path"])["task"]["status"] == "succeeded"
    receipts = list(Path(output).glob("protocol-*/exit_receipt.json"))
    assert len(receipts) == 2  # producer observation + independent candidate execution
    assert all(json.loads(path.read_text())["termination_confirmed"] for path in receipts)


def test_process_protocol_cancel_blocks_late_start_without_promoting_task(process_scenario):
    case = process_scenario
    task = case.claim()
    marker = case.root / "must-not-run"
    request, root = request_file(case, task, f"from pathlib import Path; Path({str(marker)!r}).touch()")
    cancelled = invoke(case, "cancel", root, request)
    assert cancelled["state"] == "CANCELLED" and cancelled["termination_confirmed"]
    replay = invoke(case, "start", root, request)
    assert replay["state"] == "CANCELLED" and replay["task_verdict"] is None
    assert not marker.exists()
    conflict = json.loads(request.read_text())
    conflict["nonce"] = "different-request-nonce-002"
    path = ArtifactStore(case.root).write_json("conflict.json", conflict)
    assert invoke(case, "start", root, path, expected=2)["protocol_status"] == "REJECTED"
    assert case.status()["tasks"][0]["status"] == "running"
    assert not case.status()["run"]["metadata"].get("execution_records")


def test_process_protocol_new_controller_reuses_single_execution(process_scenario):
    case = process_scenario
    task = case.claim()
    marker = case.root / "actual-starts.txt"
    request, root = request_file(case, task,
        f"import time; f=open({str(marker)!r},'a'); f.write('started\\n'); f.close(); time.sleep(0.4)")
    first = invoke(case, "start", root, request)
    # The launch CLI has exited. Its monitor continues; another CLI must not spawn again.
    second = invoke(case, "start", root, request)
    assert second["request_sha256"] == first["request_sha256"]
    result = await_terminal(case, root, request)
    assert result["state"] == "EXITED" and result["returncode"] == 0
    assert marker.read_text().splitlines() == ["started"]
    assert invoke(case, "inspect", root, request) == result
    assert invoke(case, "start", root, request) == result
    assert case.status()["tasks"][0]["status"] == "running"
