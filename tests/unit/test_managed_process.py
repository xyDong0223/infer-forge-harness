"""Standalone supervisor tests use real processes, not uploaded exit/PASS reports."""

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from core.paths import REPO_ROOT
from operations.deployment import managed_process as protocol


pytestmark = pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="POSIX process protocol")
CLI = REPO_ROOT / "cli/deployment/managed_process.py"


def request(tmp_path, source="print('observed stdout')", timeout=10):
    return {"schema_version": 1, "execution_id": "unit:process", "nonce": uuid4().hex,
            "argv": [sys.executable, "-c", source], "cwd": str(tmp_path.resolve()),
            "timeout_seconds": timeout}


def wait_for(root, req, predicate=lambda observation: observation["terminal"], seconds=8):
    until = time.monotonic() + seconds
    result = None
    while time.monotonic() < until:
        result = protocol.inspect(root, req)
        if predicate(result):
            return result
        time.sleep(0.05)
    pytest.fail(f"process observation did not reach expected state: {result}")


def cli(command, root, req_path):
    child = subprocess.run([sys.executable, "-B", str(CLI), command, "--root", str(root),
                            "--request", str(req_path)], capture_output=True, text=True,
                           timeout=10, check=False)
    return child, json.loads(child.stdout)


def test_completion_and_new_cli_process_replay_never_execute_twice(tmp_path):
    root = tmp_path / "process"
    req = request(tmp_path, "from pathlib import Path; Path('once').open('x').write('only'); print(6 * 7)")
    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(req))
    child, started = cli("start", root, req_path)
    assert child.returncode == 0 and started["task_verdict"] is None
    finished = wait_for(root, req)
    assert finished["state"] == "EXITED" and finished["returncode"] == 0
    assert finished["termination_confirmed"] is True and finished["ledger_integrated"] is False
    assert (root / "console.log").read_text().strip() == "42"
    assert finished["process"]["pid"] == finished["process"]["sid"] == finished["process"]["pgid"]
    receipt = json.loads((root / "exit_receipt.json").read_text())
    assert receipt["request_sha256"] == finished["request_sha256"]
    assert receipt["log_sha256"] == hashlib.sha256((root / "console.log").read_bytes()).hexdigest()
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    replay_child, replay = cli("start", root, req_path)
    assert replay_child.returncode == 0 and replay == finished
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert (tmp_path / "once").read_text() == "only"


def test_nonzero_exit_is_a_process_receipt_not_a_pass(tmp_path):
    root, req = tmp_path / "process", request(tmp_path, "import sys; print('original failure'); sys.exit(7)")
    protocol.start(root, req)
    finished = wait_for(root, req)
    assert finished["state"] == "EXITED" and finished["returncode"] == 7
    assert finished["task_verdict"] is None and finished["terminal"]
    assert "original failure" in (root / "console.log").read_text()


def test_concurrent_fresh_cli_starts_spawn_once(tmp_path):
    root = tmp_path / "process"
    req = request(tmp_path, "from pathlib import Path; Path('once').open('x').write('one'); print('done')")
    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(req))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: cli("start", root, req_path), range(2)))
    assert all(child.returncode == 0 for child, _ in results)
    finished = wait_for(root, req)
    assert finished["returncode"] == 0
    assert (root / "console.log").read_text().strip() == "done"


def test_durable_exit_receipt_survives_missing_mutable_state(tmp_path):
    root, req = tmp_path / "process", request(tmp_path)
    first = protocol.cancel(root, req)
    (root / "state.json").unlink()
    assert protocol.inspect(root, req) == first
    assert protocol.start(root, req) == first
    assert protocol.cancel(root, req) == first
    assert not (root / "state.json").exists()


def test_symlink_receipt_is_not_an_observation(tmp_path):
    root, req = tmp_path / "process", request(tmp_path)
    protocol.cancel(root, req)
    receipt = root / "exit_receipt.json"
    copied = tmp_path / "receipt-copy.json"
    copied.write_bytes(receipt.read_bytes())
    receipt.unlink()
    receipt.symlink_to(copied)
    with pytest.raises(OSError):
        protocol.inspect(root, req)


@pytest.mark.parametrize("field,value", [("nonce", "different-nonce-12345"),
                                         ("execution_id", "other"), ("argv", ["false"])])
def test_conflicting_identity_or_payload_cannot_use_a_reserved_root(tmp_path, field, value):
    root, req = tmp_path / "process", request(tmp_path)
    protocol.cancel(root, req)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    for operation in (protocol.start, protocol.inspect, protocol.cancel):
        with pytest.raises(ValueError, match="different request"):
            operation(root, {**req, field: value})
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_cancel_before_start_tombstone_blocks_late_start(tmp_path):
    root, req = tmp_path / "process", request(tmp_path, "from pathlib import Path; Path('forbidden').touch()")
    cancelled = protocol.cancel(root, req)
    assert cancelled["state"] == "CANCELLED" and cancelled["termination_confirmed"]
    assert (root / "cancel.json").is_file()
    assert protocol.start(root, req) == cancelled
    assert not (tmp_path / "forbidden").exists()
    assert not (root / "monitor.log").exists()


def test_timeout_cancels_owned_process_group(tmp_path):
    root, req = tmp_path / "process", request(tmp_path, "import time; time.sleep(30)", timeout=0.2)
    protocol.start(root, req)
    finished = wait_for(root, req)
    assert finished["state"] == "CANCELLED" and finished["returncode"] < 0
    assert finished["reason"] == "timeout" and finished["termination_confirmed"]
    assert not protocol._group_alive(finished["process"]["pid"])


def test_explicit_cancel_escalates_only_matching_live_identity(tmp_path):
    source = "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path('ready').touch(); time.sleep(30)"
    root, req = tmp_path / "process", request(tmp_path, source)
    protocol.start(root, req)
    wait_for(root, req, lambda _: (tmp_path / "ready").exists())
    protocol.cancel(root, req)
    finished = wait_for(root, req)
    assert finished["state"] == "CANCELLED" and finished["returncode"] == -signal.SIGKILL
    assert not protocol._group_alive(finished["process"]["pid"])


def test_missing_state_is_unknown_and_read_only(tmp_path):
    root, req = tmp_path / "missing", request(tmp_path)
    result = protocol.inspect(root, req)
    assert result["state"] == "UNKNOWN" and not result["termination_confirmed"]
    assert not root.exists()


def test_monitor_exception_preserves_private_traceback_and_unknown_state(tmp_path):
    root, req = tmp_path / "process", request(tmp_path)
    req["argv"] = [str(tmp_path / "missing-program")]
    protocol.start(root, req)
    unknown = wait_for(root, req, lambda value: value["state"] == "UNKNOWN"
                       and "Traceback" in (root / "monitor.log").read_text())
    assert not unknown["terminal"] and not unknown["termination_confirmed"]
    trace = (root / "monitor.log").read_text()
    assert "Traceback (most recent call last)" in trace and "FileNotFoundError" in trace
    assert "monitor interrupted" in json.loads((root / "state.json").read_text())["reason"]
    assert "Traceback" not in unknown["reason"]
    assert not (root / "exit_receipt.json").exists()


def test_missing_state_after_reservation_never_restarts(tmp_path):
    root, req = tmp_path / "process", request(tmp_path)
    protocol.cancel(root, req)
    (root / "state.json").unlink()
    (root / "exit_receipt.json").unlink()
    result = protocol.start(root, req)
    assert result["state"] == "UNKNOWN" and not result["termination_confirmed"]
    assert not (root / "monitor.log").exists()


def test_live_group_after_leader_exit_is_not_terminal(tmp_path):
    source = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.8); print(99)']); print('leader exited')"
    root, req = tmp_path / "process", request(tmp_path, source)
    protocol.start(root, req)
    occupied = wait_for(root, req, lambda value: value["state"] == "UNKNOWN")
    assert occupied["terminal"] is False and occupied["termination_confirmed"] is False
    assert protocol._group_alive(occupied["spawned_pid"])
    finished = wait_for(root, req)
    assert finished["state"] == "EXITED" and finished["returncode"] == 0
    assert "99" in (root / "console.log").read_text()


def test_group_permission_error_preserves_occupancy_until_process_lookup_error(monkeypatch):
    observations = [PermissionError(1, "Operation not permitted"),
                    PermissionError(1, "Operation not permitted"),
                    ProcessLookupError(3, "No such process")]

    def observe(pid, signum):
        assert pid == 123 and signum == 0
        raise observations.pop(0)

    monkeypatch.setattr(protocol.os, "killpg", observe)
    assert protocol._group_alive(123) is True
    assert protocol._group_alive(123) is True
    assert protocol._group_alive(123) is False


def test_monitor_crash_keeps_live_process_unknown_and_replay_does_not_launch(tmp_path):
    root, req = tmp_path / "process", request(tmp_path, "import time; time.sleep(1)")
    protocol.start(root, req)
    running = wait_for(root, req, lambda value: value["state"] == "RUNNING")
    identity = running["monitor"]
    assert protocol.process_identity(identity["pid"]) == identity
    os.kill(identity["pid"], signal.SIGKILL)
    unknown = wait_for(root, req, lambda value: value["state"] == "UNKNOWN")
    assert not unknown["termination_confirmed"] and unknown["process"] == running["process"]
    assert protocol.start(root, req)["process"] == running["process"]
    # Recovery does not fabricate the lost exit code. Explicit cancellation may
    # certify absence later, and does not restart the original command.
    until = time.monotonic() + 5
    while time.monotonic() < until:
        cancelled = protocol.cancel(root, req)
        if cancelled["terminal"]:
            break
        time.sleep(0.05)
    assert cancelled["state"] == "CANCELLED" and cancelled["returncode"] is None


def test_dead_monitor_cancel_escalates_term_ignoring_child_without_respawn(tmp_path):
    source = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path('once').open('x').write('only'); Path('ready').touch(); time.sleep(30)"
    )
    root, req = tmp_path / "process", request(tmp_path, source)
    protocol.start(root, req)
    running = wait_for(root, req, lambda value: value["state"] == "RUNNING" and (tmp_path / "ready").exists())
    identity = running["monitor"]
    child_identity = running["process"]
    assert protocol.process_identity(identity["pid"]) == identity
    os.kill(identity["pid"], signal.SIGKILL)
    try:
        wait_for(root, req, lambda value: value["state"] == "UNKNOWN")
        first = protocol.cancel(root, req)
        assert first["state"] == "UNKNOWN" and not first["termination_confirmed"]
        assert protocol.process_identity(child_identity["pid"]) == child_identity
        marker = json.loads((root / "cancel.json").read_text())
        assert protocol.start(root, req)["process"] == child_identity
        until = time.monotonic() + 6
        while time.monotonic() < until:
            cancelled = protocol.cancel(root, req)
            if cancelled["terminal"]:
                break
            time.sleep(0.05)
        assert cancelled["state"] == "CANCELLED" and cancelled["termination_confirmed"]
        assert cancelled["returncode"] is None  # The dead monitor's exit code is unknown.
        assert cancelled["process"] == child_identity and cancelled["monitor"] == identity
        assert not protocol._group_alive(child_identity["pid"])
        assert json.loads((root / "cancel.json").read_text()) == marker
        assert protocol.start(root, req) == cancelled
        assert (tmp_path / "once").read_text() == "only"
    finally:
        protocol._signal_owned(child_identity, signal.SIGKILL)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
def test_reused_process_identity_is_never_signalled(monkeypatch, signum):
    original = {"pid": 123, "pgid": 123, "sid": 123, "birth": "original"}
    monkeypatch.setattr(protocol, "process_identity", lambda _: {**original, "birth": "replacement"})
    monkeypatch.setattr(protocol.os, "killpg", lambda *args: pytest.fail("must not signal reused identity"))
    assert protocol._signal_owned(original, signum) is False


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan"), 10 ** 400, 86401])
def test_invalid_timeout_rejected_before_root_creation(tmp_path, timeout):
    root, req = tmp_path / "process", request(tmp_path, timeout=timeout)
    with pytest.raises(ValueError, match="timeout_seconds"):
        protocol.start(root, req)
    assert not root.exists()


def test_rejects_source_cwd_source_root_symlinks_and_used_root(tmp_path):
    req = request(tmp_path)
    with pytest.raises(ValueError, match="source repository"):
        protocol.start(REPO_ROOT / "forbidden-supervisor-output", req)
    with pytest.raises(ValueError, match="source repository"):
        protocol.start(tmp_path / "process", {**req, "cwd": str(REPO_ROOT)})
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        protocol.start(alias, req)
    (actual / "existing-user-file").write_text("preserve")
    with pytest.raises(ValueError, match="fresh and empty"):
        protocol.start(actual, req)
    assert (actual / "existing-user-file").read_text() == "preserve"


def test_unknown_request_fields_and_duplicate_json_rejected(tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        protocol.start(tmp_path / "process", {**request(tmp_path), "env": {"LEASE_TOKEN": "secret"}})
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        protocol.load_request(duplicate)


def test_child_does_not_inherit_controller_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("INFER_FORGE_TEST_CONTROLLER_TOKEN", "must-not-leak")
    root = tmp_path / "process"
    req = request(tmp_path, "import os; print(os.environ.get('INFER_FORGE_TEST_CONTROLLER_TOKEN', 'absent'))")
    protocol.start(root, req)
    assert wait_for(root, req)["returncode"] == 0
    assert (root / "console.log").read_text().strip() == "absent"
