"""Phase descriptions match execution without discarding target launch settings."""
import copy
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from runners import task_runner

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    return task_runner.load_yaml(ROOT / relative)


@pytest.mark.parametrize('task_type,requested,effective', [
    ('deployment_proof', 'environment', 'environment'),
    ('deployment_proof', 'service', 'service'),
    ('environment_proof', 'all', 'environment'),
    ('service_proof', 'environment', 'service'),
    ('deployment_proof', 'all', 'all'),
])
def test_plan_reports_effective_type_and_actions(task_type, requested, effective):
    contract = load('tests/fixtures/deployment-proof.yaml')
    contract['metadata']['task_type'] = task_type
    original = copy.deepcopy(contract)
    plan = task_runner.build_plan(contract, phase=requested)
    assert plan['phase'] == effective
    assert plan['task_type'] == (f'{effective}_proof' if effective != 'all' else task_type)
    if effective != 'all':
        name = 'kdp-001a-environment-proof' if effective == 'environment' else 'kdp-001b-service-proof'
        assert plan['actions'] == load(f'tasks/{name}/task.yaml')['actions']
    assert contract == original


def test_service_execution_persists_phase_contract_and_preserves_target(tmp_path, capsys):
    path = ROOT / 'tests/fixtures/deployment-proof.yaml'
    contract = task_runner.load_yaml(path)
    original = copy.deepcopy(contract)
    service = load('tasks/kdp-001b-service-proof/task.yaml')
    with patch('adapters.ClusterConfig.load', return_value=SimpleNamespace(namespace=contract['execution']['namespace'])), \
            patch('runners.deployment_proof.DeploymentProofRunner') as runner:
        # Independent validation rejects this incomplete result, but the real
        # executor must still persist its effective contract before dispatch.
        runner.return_value.run.return_value = {'state': 'SERVER_START_FAILED', 'checks': {}, 'artifacts': []}
        assert task_runner.execute(contract, path, tmp_path, attach_pod='fixture-pod',
                                   phase='service', user_id='fixture') == 6
    effective = runner.call_args.kwargs['contract']
    assert effective['metadata']['task_type'] == 'service_proof'
    for field in ('actions', 'acceptance', 'exit_states'):
        assert effective[field] == service[field]
    assert effective['context']['model'] == original['context']['model']
    assert effective['context']['server'] == original['context']['server']
    assert effective['execution'] == {**original['execution'], 'user_id': 'fixture'}
    assert effective['checks'] == original['checks']
    assert set(service['artifacts']) <= set(effective['artifacts']['collect'])
    output = Path(json.loads(capsys.readouterr().out)['artifact_root'])
    persisted = json.loads((output.parent / 'input/task_contract.json').read_text())
    assert persisted == effective
    requested = json.loads((output.parent / 'input/requested_task_contract.json').read_text())
    assert requested['contract']['actions'] == original['actions']


@pytest.mark.parametrize('task,expected', [
    ('kdp-001a-environment-proof', {
        'ENVIRONMENT_READY', 'INPUT_REQUIRED', 'CONTRACT_INVALID', 'BLOCKED', 'RUNTIME_DRIFT',
        'NEEDS_HUMAN', 'INSTALL_FAILED', 'MODEL_NOT_FOUND', 'CODE_NOT_READY',
        'DEVICE_NOT_READY', 'SERVER_START_FAILED', 'READINESS_TIMEOUT',
        'API_SMOKE_FAILED', 'UNEXPECTED_FALLBACK',
    }),
    ('kdp-001b-service-proof', {
        'DEPLOYMENT_READY', 'INPUT_REQUIRED', 'CONTRACT_INVALID', 'BLOCKED', 'RUNTIME_DRIFT',
        'NEEDS_HUMAN', 'BRINGUP_BLOCKED', 'SERVER_START_FAILED',
        'READINESS_TIMEOUT', 'API_SMOKE_FAILED', 'UNEXPECTED_FALLBACK',
    }),
])
def test_reachable_phase_states_are_declared_across_contract_catalog_schema(task, expected):
    declared = set(load(f'tasks/{task}/task.yaml')['exit_states'].values())
    assert declared == expected
    catalog = load('catalog/skill_catalog.yaml')
    skill = next(entry for entry in catalog['entries'] if entry['id'] == 'environment-proof')
    assert declared <= set(skill['exit_conditions'])
    assert declared <= set(load('contracts/status.schema.yaml')['properties']['state']['enum'])


@pytest.mark.parametrize('phase', ['environment', 'service', 'all'])
def test_missing_owner_publishes_schema_valid_rejection_before_cluster_access(tmp_path, monkeypatch, capsys, phase):
    monkeypatch.delenv('USER_ID', raising=False)
    contract = load('tests/fixtures/deployment-proof.yaml')
    contract['execution'].pop('user_id', None)
    with patch('adapters.ClusterConfig.load') as cluster, \
            patch('runners.deployment_proof.DeploymentProofRunner') as runner:
        assert task_runner.execute(contract, None, tmp_path, phase=phase) == 3
    cluster.assert_not_called()
    runner.assert_not_called()
    emitted = json.loads(capsys.readouterr().out)
    persisted = json.loads((Path(emitted['artifact_root']) / 'status.json').read_text())
    assert emitted == persisted
    Draft202012Validator(load('contracts/status.schema.yaml')).validate(persisted)
    assert persisted['task_id'] == contract['metadata']['name']
    assert persisted['state'] == persisted['status'] == 'INPUT_REQUIRED'
    assert datetime.fromisoformat(persisted['updated_at']).utcoffset() is not None
    manifest = json.loads(Path(persisted['manifest_path']).read_text())
    assert manifest['outcome'] == 'INPUT_REQUIRED'


def test_service_owner_change_is_rejected_before_publishing_inconsistent_contract(tmp_path, capsys):
    contract = load('tests/fixtures/deployment-proof.yaml')
    contract['metadata']['task_type'] = 'service_proof'
    contract['execution'].update(user_id='original-owner', resource_name='original-owner-service')
    original = copy.deepcopy(contract)
    with patch('adapters.ClusterConfig.load') as cluster:
        assert task_runner.execute(contract, None, tmp_path, user_id='new-owner') == 2
    cluster.assert_not_called()
    assert contract == original
    status = json.loads(capsys.readouterr().out)
    assert 'does not match' in status['message']
    Draft202012Validator(load('contracts/status.schema.yaml')).validate(status)


def test_environment_ready_status_schema_requires_user_id():
    schema = Draft202012Validator(load('contracts/status.schema.yaml'))
    status = {
        'task_id': 'kdp-001a-environment-proof',
        'state': 'ENVIRONMENT_READY',
        'updated_at': '2026-09-17T00:00:00+00:00',
        'user_id': 'fixture-owner',
    }
    schema.validate(status)
    with pytest.raises(ValidationError):
        schema.validate({key: value for key, value in status.items() if key != 'user_id'})


def test_environment_owner_change_is_rejected_before_publishing_inconsistent_contract(tmp_path, capsys):
    contract = load('tests/fixtures/deployment-proof.yaml')
    contract['metadata']['task_type'] = 'environment_proof'
    contract['execution'].update(user_id='original-owner', resource_name='original-owner-environment')
    original = copy.deepcopy(contract)
    with patch('adapters.ClusterConfig.load') as cluster:
        assert task_runner.execute(contract, None, tmp_path, phase='environment', user_id='new-owner') == 2
    cluster.assert_not_called()
    assert contract == original
    status = json.loads(capsys.readouterr().out)
    assert 'prepared environment owner' in status['message']
    Draft202012Validator(load('contracts/status.schema.yaml')).validate(status)
