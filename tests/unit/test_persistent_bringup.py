"""Regressions for environment identity, retained Pods and pre-load gates."""
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

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'tasks/kdp-001-deployment-proof/instances/qwen3-8b-p800.yaml'


def runner(tmp_path, phase='service'):
    adapter = Mock()
    adapter.assert_owned = Mock()
    adapter.get.return_value = subprocess.CompletedProcess([], 0, "pod: fixture", "")
    adapter.config.kubeconfig = 'fixture'
    adapter.http_probe.return_value = (503, '')
    adapter.exec.return_value = subprocess.CompletedProcess([], 0, '', '')
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


def test_environment_phase_replaces_target_contract_with_minimax(tmp_path, monkeypatch):
    monkeypatch.setenv('USER_ID', 'fixture')
    contract = yaml.safe_load(CONTRACT.read_text())
    contract['execution']['server_log'] = '/workspace/base-reproof.log'
    with patch('adapters.ClusterConfig.load', return_value=SimpleNamespace(namespace='pd-test')), \
            patch('runners.deployment_proof.DeploymentProofRunner') as constructor:
        constructor.return_value.run.return_value = {}
        task_runner._execute(contract, tmp_path, 'fixture-pod', 'environment', None,
                             lambda status, code: code)
    actual = constructor.call_args.kwargs['contract']
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
    scheduler.record_environment_failure('retry', {'pod': 'fixture-existing'}, 'failed')
    args = parser.parse_args(['environment', '--run-id', 'retry', '--contract', str(CONTRACT)])
    failed = subprocess.CompletedProcess([], 6, json.dumps({'pod': 'fixture-existing', 'state': 'INSTALL_FAILED'}), '')
    with patch('cli.adaptation.subprocess.run', return_value=failed) as command:
        _run(args, scheduler)
    argv = command.call_args.args[0]
    assert argv[argv.index('--attach-pod') + 1] == 'fixture-existing'
