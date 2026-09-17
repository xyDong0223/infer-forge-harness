"""Boundary bootstrap checks; no graph or validator monkeypatches."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

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


@pytest.mark.parametrize("phase_args", [[], ["--phase", "environment"]])
def test_real_environment_cli_validates_raw_simulated_observations(tmp_path, phase_args):
    fixture = prepare_environment(tmp_path / "external")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli/deployment/proof.py"),
         fixture["contract_instance"], "--execute", *phase_args,
         "--artifact-dir", str(tmp_path / "proof"), "--run-id", "boundary-smoke"],
        cwd=tmp_path, env={**os.environ, **fixture["env"]},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads(result.stdout)
    assert validate_environment_status(status) == []
    assert status["evidence_mode"] == "simulation"
    assert status["pod"] == fixture["pod"]
    assert status["phase"] == "environment"
    assert "toy_bringup" not in status["checks"]
    assert not (Path(status["artifact_root"]) / "toy_bringup.json").exists()
    contract = yaml.safe_load((Path(status["artifact_root"]) / "task_contract.yaml").read_text())
    assert "toy_bringup_before_load" not in contract["actions"]
    assert contract["acceptance"]["base_prefill"] is True
    assert contract["exit_states"]["pass"] == "ENVIRONMENT_READY"
    assert (Path(status["artifact_root"]) / "environment_fingerprint.txt").is_file()
    reused = _python(fixture, (
        "from adapters import get_hardware; "
        f"a=get_hardware()(); print(a.pod_ready({fixture['pod']!r})); "
        f"print(a.http_probe({fixture['pod']!r}, '/health', 8356)[0])"
    ))
    assert reused.returncode == 0, reused.stderr
    assert reused.stdout.splitlines() == ["True", "503"]
    unexpected = _python(fixture, (
        "from adapters import get_hardware; "
        f"get_hardware()().exec({fixture['pod']!r}, 'unexpected remote command')"
    ))
    assert unexpected.returncode != 0
    assert "unexpected external exec command" in unexpected.stderr
    events = [json.loads(line) for line in fixture["events"].read_text().splitlines()]
    scripts = [event["script"] for event in events if event["operation"] == "exec"]
    launches = [script for script in scripts if "echo started $!" in script]
    assert len(launches) == 1
    assert "minimax-base-smoke" in launches[0]
    assert not any("mat028_probe.py" in script for script in scripts)
    assert {event["pod"] for event in events if "pod" in event} == {fixture["pod"]}
    assert len({event["pid"] for event in events}) >= 3
    assert all(event["evidence_mode"] == "simulation" for event in events)


def test_persistent_environment_and_toy_failure_block_target_weights(tmp_path):
    fixture = prepare_environment(tmp_path / 'external')
    env = {**os.environ, **fixture['env']}

    def proof(phase):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / 'cli/deployment/proof.py'),
             fixture['contract_instance'], '--execute', '--phase', phase,
             '--artifact-dir', str(tmp_path / 'proof'), '--run-id', 'persistent',
             *(['--attach-pod', fixture['pod']] if phase == 'service' else [])],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )
        return result, json.loads(result.stdout)

    first, baseline = proof('environment')
    assert first.returncode == 0, first.stdout + first.stderr
    identity = json.loads((Path(baseline['artifact_root']) / 'base_model_identity.json').read_text())
    assert identity['name'] == 'MiniMax-M2.5-Int8-W8A8'
    second, resumed = proof('environment')
    assert second.returncode == 0, second.stdout + second.stderr
    assert baseline['pod'] == resumed['pod']
    assert baseline['artifact_root'] != resumed['artifact_root']

    settings = json.loads(fixture['settings'].read_text())
    settings['toy_failure'] = True
    fixture['settings'].write_text(json.dumps(settings))
    failed, blocked = proof('service')
    assert failed.returncode != 0
    assert blocked['state'] == 'BRINGUP_BLOCKED'
    assert blocked['pod'] == baseline['pod']
    events = [json.loads(line) for line in fixture['events'].read_text().splitlines()]
    scripts = [event['script'] for event in events if event['operation'] == 'exec']
    launches = [script for script in scripts if 'echo started $!' in script]
    assert len(launches) == 2  # two baseline proofs, no target launch
    assert all('minimax-base-smoke' in script for script in launches)
    assert not any('patch_vllm_kunlun_drift.py' in script for script in scripts)
    applies = [event for event in events if event['operation'] == 'cluster'
               and event['args'][:2] == ['apply', '-f']]
    assert len(applies) == 1

    settings.pop('toy_failure')
    fixture['settings'].write_text(json.dumps(settings))
    passed, ready = proof('service')
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert ready['pod'] == baseline['pod']
    assert ready['checks']['toy_bringup'] is True
    assert (Path(blocked['artifact_root']) / 'toy_bringup.json').is_file()
