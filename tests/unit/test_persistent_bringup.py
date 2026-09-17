"""Regressions for environment identity, retained Pods and toy bring-up gates."""
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from cli.adaptation import _parser, _run
from engine import TaskScheduler
from runners import task_runner
from runners.deployment_proof import ActionFailed, DeploymentProofRunner
from tests.unit.test_preflight_gates import BRINGUP
from operations.deployment.toy_bringup import BringupFailed

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'tasks/kdp-001-deployment-proof/instances/qwen3-8b-p800.yaml'


def runner(tmp_path, phase='service'):
    adapter = Mock()
    adapter.assert_owned = Mock()
    adapter.get.return_value = subprocess.CompletedProcess([], 0, "pod: fixture", "")
    adapter.config.kubeconfig = 'fixture'
    adapter.http_probe.return_value = (503, '')
    adapter.exec.side_effect = lambda pod, script, **kwargs: subprocess.CompletedProcess(
        [], 0, json.dumps(BRINGUP['config']['kept_dimensions']) if script.endswith('/config.json') else '', '')
    result = DeploymentProofRunner(
        yaml.safe_load(CONTRACT.read_text()), adapter, ROOT, tmp_path,
        attach_pod='dongxinyu03-existing', phase=phase,
    )
    result.pod = result.attach_pod
    return result


@pytest.mark.parametrize('complete', [True, False])
def test_toy_gates_direct_target_launch(tmp_path, complete):
    instance = runner(tmp_path)
    report = copy.deepcopy(BRINGUP)
    if not complete:
        report.update(complete=False, error={'type': 'RuntimeError', 'message': 'decode failed'})
    with patch('operations.deployment.toy_bringup.run_probe', return_value=report) as probe:
        if complete:
            instance.start_server()
        else:
            with pytest.raises(ActionFailed, match='BRINGUP_BLOCKED'):
                instance.start_server()
    assert probe.call_args.args[1] == 'dongxinyu03-existing'
    assert probe.call_args.args[2] == instance.contract['context']['model']['path']
    assert probe.call_args.args[5] == instance.contract['context']['server']['tensor_parallel_size']
    assert (tmp_path / 'toy_bringup.json').is_file()
    launches = [call for call in instance.adapter.exec.call_args_list
                if 'echo started $!' in call.args[1]]
    assert len(launches) == int(complete)
    instance.adapter.apply.assert_not_called()


def test_existing_unready_pod_is_retained(tmp_path):
    instance = runner(tmp_path, 'environment')
    instance.adapter.pod_ready.return_value = False
    status = instance.run()
    assert status['state'] == 'NEEDS_HUMAN'
    assert status['pod'] == 'dongxinyu03-existing'
    instance.adapter.apply.assert_not_called()
    instance.adapter.exec.assert_not_called()


def test_existing_deployment_is_attached_without_apply_or_install(tmp_path):
    instance = runner(tmp_path, 'environment')
    instance.attach_pod = None
    instance.adapter.run.side_effect = [
        subprocess.CompletedProcess([], 0, json.dumps({'spec': {'selector': {
            'matchLabels': {'infer.kunlun/attempt-id': 'original'},
        }}}), ''),
        subprocess.CompletedProcess([], 0, 'pod/dongxinyu03-existing\n', ''),
    ]
    # Stop at the next read-only gate; resource reuse must happen first.
    instance._runtime_already_installed = Mock(return_value=True)
    instance.verify_runtime_importable = Mock(side_effect=ActionFailed('INSTALL_FAILED', 'import failure'))
    status = instance.run()
    assert status['pod'] == 'dongxinyu03-existing'
    assert status['state'] == 'INSTALL_FAILED'
    instance.adapter.apply.assert_not_called()
    instance.adapter.exec.assert_not_called()  # includes the removed drift patch
    assert 'infer.kunlun/attempt-id=original' in instance.adapter.run.call_args.args[0]
    manifest = yaml.safe_load((tmp_path / 'deployment_manifest.yaml').read_text())
    assert manifest['spec']['selector']['matchLabels']['infer.kunlun/attempt-id'] == 'original'


@pytest.mark.parametrize('source', ['environment', 'contract'])
def test_environment_phase_replaces_target_contract_with_minimax(tmp_path, monkeypatch, source):
    monkeypatch.setenv('USER_ID', 'fixture' if source == 'environment' else 'wrong-owner')
    contract = yaml.safe_load(CONTRACT.read_text())
    if source == 'contract':
        contract['execution']['user_id'] = 'fixture-team'
    contract['execution']['server_log'] = '/workspace/base-reproof.log'
    with patch('adapters.ClusterConfig.load', return_value=SimpleNamespace(namespace='pd-test')), \
            patch('runners.deployment_proof.DeploymentProofRunner') as constructor:
        constructor.return_value.run.return_value = {}
        task_runner._execute(contract, tmp_path, 'fixture-pod', 'environment', None,
                             lambda status, code: code)
    actual = constructor.call_args.kwargs['contract']
    owner = 'fixture' if source == 'environment' else 'fixture-team'
    assert actual['execution']['user_id'] == owner
    assert actual['execution']['resource_name'] == f'{owner}-environment-base'
    assert actual['execution']['server_log'] == '/workspace/base-reproof.log'
    assert actual['context']['model']['name'] == 'MiniMax-M2.5-Int8-W8A8'
    assert actual['checks']['chat']['payload']['model'] == 'minimax-base-smoke'
    serve = ' '.join(actual['execution']['commands']['serve'])
    assert 'MiniMax-M2.5-W8A8-INT8-Dynamic' in serve
    assert 'Qwen3' not in serve


def test_failed_environment_retry_restores_pod_from_scheduler(tmp_path):
    scheduler = TaskScheduler(tmp_path / 'state.db')
    parser = _parser()
    _run(parser.parse_args(['--state', str(tmp_path / 'state.db'), 'create-run',
                           '--run-id', 'retry', '--model', 'Qwen3', '--backend', 'kunlun']), scheduler)
    scheduler.record_environment_failure(
        'retry', {'pod': 'fixture-existing', 'user_id': 'fixture-team'}, 'failed',
    )
    args = parser.parse_args(['environment', '--run-id', 'retry', '--contract', str(CONTRACT)])
    failed = subprocess.CompletedProcess([], 6, json.dumps({'pod': 'fixture-existing', 'state': 'INSTALL_FAILED'}), '')
    with patch('cli.adaptation.subprocess.run', return_value=failed) as command:
        _run(args, scheduler)
    argv = command.call_args.args[0]
    assert argv[argv.index('--attach-pod') + 1] == 'fixture-existing'
    assert argv[argv.index('--user-id') + 1] == 'fixture-team'


@pytest.mark.parametrize('corruption', ['changed', 'missing'])
def test_toy_rejects_wrong_or_missing_kernel_dimension(tmp_path, corruption):
    instance = runner(tmp_path)
    report = copy.deepcopy(BRINGUP)
    if corruption == 'changed':
        report['config']['kept_dimensions']['hidden_size'] = 1
    else:
        report['config']['kept_dimensions'].pop('hidden_size')
    with patch('operations.deployment.toy_bringup.run_probe', return_value=report):
        with pytest.raises(ActionFailed, match='BRINGUP_BLOCKED'):
            instance.start_server()
    assert instance.checks['toy_bringup'] is False
    assert not any('echo started $!' in c.args[1] for c in instance.adapter.exec.call_args_list)


@pytest.mark.parametrize('failure', [
    subprocess.TimeoutExpired('toy', 900),
    BringupFailed(
        'NEEDS_HUMAN', 'worker died without JSON'),
])
def test_toy_transport_failure_preserves_state_and_evidence(tmp_path, failure):
    instance = runner(tmp_path)
    with patch('operations.deployment.toy_bringup.run_probe', side_effect=failure):
        with pytest.raises(ActionFailed) as error:
            instance.start_server()
    assert error.value.state == 'NEEDS_HUMAN'
    assert instance.checks['toy_bringup'] is False
    report = json.loads((tmp_path / 'toy_bringup.json').read_text())
    assert report['stage'] == 'NOTHING_RAN'
    assert report['complete'] is False
    assert report['error']['message']
    assert not any('echo started $!' in c.args[1] for c in instance.adapter.exec.call_args_list)


def test_probe_uses_launch_runtime_setup_and_workdir():
    from operations.deployment.toy_bringup import run_probe
    adapter = Mock()
    adapter.exec.return_value = subprocess.CompletedProcess([], 0, '{}', '')
    runtime = Mock()
    runtime.env_prefix.return_value = 'source /runtime/activate'
    run_probe(adapter, 'pod', '/model', 2, 8, 8, 512, 900,
              runtime=runtime, setup=['source setup_env.sh', 'export XPU_FLAG=1'], workdir='/workspace')
    script = adapter.exec.call_args.args[1]
    assert script.startswith('cd /workspace && source /runtime/activate && source setup_env.sh && export XPU_FLAG=1 && ')
    assert '2>/dev/null' not in script


def test_all_phase_attach_repairs_interrupted_install_without_replacing_pod(tmp_path):
    instance = runner(tmp_path, 'all')
    instance._runtime_already_installed = Mock(return_value=False)
    instance.install_runtime = Mock()
    instance.start_server = Mock(side_effect=ActionFailed('BRINGUP_BLOCKED', 'stop after install'))
    status = instance.run()
    instance.install_runtime.assert_called_once_with(instance.contract['execution'])
    instance.start_server.assert_called_once_with()
    instance.adapter.apply.assert_not_called()
    assert status['pod'] == 'dongxinyu03-existing'


@pytest.mark.parametrize('field', ['name', 'path', 'served_model_name'])
def test_imported_wrong_baseline_is_rejected(tmp_path, field):
    from tests.unit.test_scheduler_reliability import environment_proof
    scheduler = TaskScheduler(tmp_path / 'state.db')
    parser = _parser()
    _run(parser.parse_args(['--state', str(tmp_path / 'state.db'), 'create-run',
                           '--run-id', 'identity', '--model', 'Qwen3', '--backend', 'kunlun']), scheduler)
    proof = environment_proof(tmp_path / 'bundle')
    identity_file = tmp_path / 'bundle/base_model_identity.json'
    identity = json.loads(identity_file.read_text())
    identity[field] = 'Qwen-target'
    identity_file.write_text(json.dumps(identity))
    status_file = tmp_path / 'bundle/status.json'
    status_file.write_text(json.dumps(proof))
    result = _run(parser.parse_args(['environment', '--run-id', 'identity', '--status', str(status_file)]), scheduler)
    assert result['run']['status'] == 'ENVIRONMENT_FAILED'
    assert 'base_model_identity.' + field in result['error']


def test_bringup_failure_is_declared_by_shared_and_deployment_contracts():
    status_schema = yaml.safe_load((ROOT / 'contracts/status.schema.yaml').read_text())
    assert 'BRINGUP_BLOCKED' in status_schema['properties']['state']['enum']
    for path in (ROOT / 'tasks/kdp-001-deployment-proof').rglob('*.yaml'):
        contract = yaml.safe_load(path.read_text())
        if 'exit_states' not in contract:
            continue
        if contract['metadata']['task_type'] == 'environment_proof':
            assert contract['exit_states']['pass'] == 'ENVIRONMENT_READY'
            assert 'toy_bringup_before_load' not in contract['actions']
            continue
        assert 'BRINGUP_BLOCKED' in contract['exit_states'].values(), path
        assert 'toy_bringup_before_load' in contract['actions'], path


@pytest.mark.parametrize('state', ['ENVIRONMENT_READY', 'RUNTIME_DRIFT'])
def test_environment_states_are_declared_by_shared_status_schema(state):
    status_schema = yaml.safe_load((ROOT / 'contracts/status.schema.yaml').read_text())
    assert state in status_schema['properties']['state']['enum']


def test_environment_contracts_declare_drift_rejection():
    # Both the environment task and standalone environment instances execute
    # the same read-only runtime drift gate before baseline model loading.
    paths = list((ROOT / 'tasks/kdp-001a-environment-proof').rglob('*.yaml'))
    paths += list((ROOT / 'tasks/kdp-001-deployment-proof/instances').glob('*.yaml'))
    environment_contracts = []
    for path in paths:
        contract = yaml.safe_load(path.read_text())
        if contract['metadata']['task_type'] != 'environment_proof':
            continue
        environment_contracts.append(path)
        assert contract['exit_states']['pass'] == 'ENVIRONMENT_READY', path
        assert 'RUNTIME_DRIFT' in contract['exit_states'].values(), path
    assert environment_contracts
