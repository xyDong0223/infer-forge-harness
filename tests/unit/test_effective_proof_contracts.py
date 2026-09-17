"""Phase descriptions match execution without discarding target launch settings."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
        'ENVIRONMENT_READY', 'CONTRACT_INVALID', 'BLOCKED', 'RUNTIME_DRIFT',
        'NEEDS_HUMAN', 'INSTALL_FAILED', 'MODEL_NOT_FOUND', 'CODE_NOT_READY',
        'DEVICE_NOT_READY', 'SERVER_START_FAILED', 'READINESS_TIMEOUT',
        'API_SMOKE_FAILED', 'UNEXPECTED_FALLBACK',
    }),
    ('kdp-001b-service-proof', {
        'DEPLOYMENT_READY', 'CONTRACT_INVALID', 'BLOCKED', 'RUNTIME_DRIFT',
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
