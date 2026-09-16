"""Boundary bootstrap checks; no graph or validator monkeypatches."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from core.paths import REPO_ROOT
from tests.e2e.external import SETTINGS_ENV, prepare_environment
from validators.deployment_validator import validate_environment_status

pytestmark = pytest.mark.local_e2e


def _python(fixture, code):
    return subprocess.run(
        [sys.executable, "-c", code], cwd=fixture["root"],
        env={**os.environ, **fixture["env"]}, text=True, capture_output=True,
        timeout=30, check=False,
    )


def test_bootstrap_is_opt_in_and_invalid_settings_fail_closed(tmp_path):
    fixture = prepare_environment(tmp_path / "external")
    env = {**os.environ, **fixture["env"]}
    env.pop(SETTINGS_ENV)
    original = subprocess.run(
        [sys.executable, "-c", "from adapters import get_hardware; print(get_hardware().__name__)"],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    assert original.returncode == 0, original.stderr
    assert original.stdout.strip() == "KunlunP800Adapter"
    env[SETTINGS_ENV] = str(tmp_path / "does-not-exist.json")
    invalid = subprocess.run(
        [sys.executable, "-c", "print('MUST NOT RUN')"],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    assert invalid.returncode == 78
    assert "MUST NOT RUN" not in invalid.stdout
    assert "FileNotFoundError" in invalid.stderr


@pytest.mark.parametrize("command", [
    "import socket; socket.socket().connect(('127.0.0.1', 9))",
    "import subprocess; subprocess.run(['kubectl', 'version'])",
    "import os; os.system('true')",
    "from adapters import get_hardware; get_hardware()().run(['unexpected'])",
])
def test_unexpected_external_access_fails_closed(tmp_path, command):
    fixture = prepare_environment(tmp_path / "external")
    result = _python(fixture, command)
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr


def test_real_environment_cli_validates_raw_simulated_observations(tmp_path):
    fixture = prepare_environment(tmp_path / "external")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli/deployment/proof.py"),
         fixture["contract_instance"], "--execute", "--phase", "environment",
         "--artifact-dir", str(tmp_path / "proof"), "--run-id", "boundary-smoke"],
        cwd=tmp_path, env={**os.environ, **fixture["env"]},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads(result.stdout)
    assert validate_environment_status(status) == []
    assert status["evidence_mode"] == "simulation"
    assert status["pod"] == fixture["pod"]
    assert (Path(status["artifact_root"]) / "environment_fingerprint.txt").is_file()
    reused = _python(fixture, (
        "from adapters import get_hardware; "
        f"a=get_hardware()(); print(a.pod_ready({fixture['pod']!r})); "
        f"print(a.http_probe({fixture['pod']!r}, '/health', 8356)[0])"
    ))
    assert reused.returncode == 0, reused.stderr
    assert reused.stdout.splitlines() == ["True", "200"]
    unexpected = _python(fixture, (
        "from adapters import get_hardware; "
        f"get_hardware()().exec({fixture['pod']!r}, 'unexpected remote command')"
    ))
    assert unexpected.returncode != 0
    assert "unexpected external exec command" in unexpected.stderr
    events = [json.loads(line) for line in fixture["events"].read_text().splitlines()]
    assert {event["pod"] for event in events if "pod" in event} == {fixture["pod"]}
    assert len({event["pid"] for event in events}) >= 3
    assert all(event["evidence_mode"] == "simulation" for event in events)
