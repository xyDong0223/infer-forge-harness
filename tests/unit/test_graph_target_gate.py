import subprocess
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import yaml

from runners import graph_runner, task_runner
from core.target import load_target, target_environment
from core.target import bind_subject
from tools.journal import record


ROOT = Path(__file__).resolve().parents[2]


class GraphTargetGateTests(unittest.TestCase):
    def run_graph(self, target):
        return subprocess.run(
            [
                sys.executable,
                "runners/graph_runner.py",
                "--subject", "demo",
                "--target", str(ROOT / target),
                "--artifact-root", str(ROOT / "artifacts/test-target-gate"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def test_planned_target_is_blocked_before_graph_walk(self):
        result = self.run_graph("config/examples/p800-sglang-kunlun.yaml")
        self.assertEqual(result.returncode, 2)
        self.assertIn("blocked:", result.stdout + result.stderr)

    def test_supported_target_reaches_plan(self):
        result = self.run_graph("config/examples/p800-vllm-kunlun.yaml")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("blocked:", result.stdout)


def write_target(path, **changes):
    target = {
        "model": "demo", "hardware": "kunlun/p800",
        "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun"},
    }
    target.update(changes)
    path.write_text(yaml.safe_dump({"target": target}))
    return path


def write_contract(path, *, model="demo", task_type="deployment_proof", **runtime):
    path.write_text(yaml.safe_dump({
        "api_version": "infer.kunlun/v1alpha1", "kind": "Task",
        "metadata": {"name": "deployment-demo", "task_type": task_type},
        "context": {"model": {"name": model}, "target": {"hardware": "p800"},
                    "runtime": {"engine": "vllm", "backend": "kunlun",
                                "plugin": "vllm-kunlun", **runtime}},
        "actions": ["preflight"], "acceptance": {"pod_ready": True},
    }))
    return path


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("mismatch", ["model", "runtime", "hardware", "revision"])
def test_graph_contract_mismatch_blocks_before_any_node(tmp_path, monkeypatch, execute, mismatch):
    target = write_target(tmp_path / "target.yaml")
    contract = write_contract(tmp_path / "contract.yaml")
    data = yaml.safe_load(contract.read_text())
    if mismatch == "model":
        data["context"]["model"]["name"] = "other"
    elif mismatch == "runtime":
        data["context"]["runtime"]["backend"] = "cuda"
    elif mismatch == "hardware":
        data["context"]["target"]["hardware"] = "unknown/hardware"
    else:
        write_target(target, runtime={"engine": "vllm", "backend": "kunlun",
                                     "plugin": "vllm-kunlun", "revisions": {"plugin": "new"}})
        data["context"]["runtime"]["revisions"] = {"plugin": "old"}
    contract.write_text(yaml.safe_dump(data))
    argv = ["graph", "--subject", "demo", "--target", str(target),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--set", f"contract_instance={contract}"]
    if execute:
        argv += ["--execute"]
    monkeypatch.setattr(sys, "argv", argv)
    with patch.object(graph_runner.evidence, "run_logged") as run, \
            patch.object(graph_runner, "load_workflow") as workflow:
        assert graph_runner.main() == 2
    run.assert_not_called()
    workflow.assert_not_called()
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("hardware,runtime", [
    ("unknown/hardware", {"engine": "unknown", "backend": "unknown"}),
    ("kunlun/p800", {"engine": "sglang", "backend": "kunlun", "plugin": "sglang-kunlun"}),
])
@pytest.mark.parametrize("execute", [False, True])
def test_unknown_or_planned_graph_target_never_resolves_resources(
    tmp_path, monkeypatch, hardware, runtime, execute
):
    target = write_target(tmp_path / "target.yaml", hardware=hardware, runtime=runtime)
    argv = ["graph", "--subject", "demo", "--target", str(target),
            "--artifact-root", str(tmp_path / "artifacts")]
    monkeypatch.setattr(sys, "argv", argv + (["--execute"] if execute else []))
    with patch("core.facade.get_hardware") as hardware_loader, \
            patch.object(graph_runner.evidence, "run_logged") as run:
        assert graph_runner.main() == 2
    hardware_loader.assert_not_called()
    run.assert_not_called()


def test_target_is_forwarded_to_deployment_command_and_revalidated(tmp_path):
    target = write_target(tmp_path / "target.yaml")
    contract = write_contract(tmp_path / "contract.yaml")
    context = {"subject": "demo", "target_file": str(target),
               "contract_instance": str(contract), "artifacts": str(tmp_path)}
    command = graph_runner.resolve(graph_runner.NODES["environment_proof"], context,
                                   tmp_path / "journal.jsonl", {"hardware": "p800"})
    assert command[-4:] == ["--target", str(target), "--subject", "demo"]
    write_contract(contract, backend="cuda")
    with pytest.raises(ValueError, match="backend"):
        graph_runner.resolve(graph_runner.NODES["environment_proof"], context,
                             tmp_path / "journal.jsonl", {"hardware": "p800"})


@pytest.mark.parametrize("execute", [False, True])
def test_task_runner_checks_target_before_cluster_or_placeholder_work(tmp_path, monkeypatch, execute):
    target = write_target(tmp_path / "target.yaml")
    contract = write_contract(tmp_path / "contract.yaml", model="other")
    monkeypatch.setattr(sys, "argv", ["task", str(contract), "--target", str(target)]
                        + (["--execute"] if execute else []))
    with patch("adapters.ClusterConfig.load") as cluster:
        assert task_runner.main() == 2
    cluster.assert_not_called()


def test_direct_execute_does_not_rewrite_unsupported_environment_into_p800(tmp_path):
    contract = write_contract(tmp_path / "contract.yaml", task_type="environment_proof",
                              engine="sglang", plugin="sglang-kunlun")
    with patch("adapters.ClusterConfig.load") as cluster, \
            patch.object(task_runner, "load_yaml") as profile:
        assert task_runner.execute(yaml.safe_load(contract.read_text()), contract, tmp_path) == 2
    profile.assert_not_called()
    cluster.assert_not_called()


@pytest.mark.parametrize("execute", [False, True])
def test_legacy_task_runner_still_blocks_planned_runtime(tmp_path, monkeypatch, execute):
    contract = write_contract(tmp_path / "contract.yaml", engine="sglang", plugin="sglang-kunlun")
    monkeypatch.setattr(sys, "argv", ["task", str(contract)] + (["--execute"] if execute else []))
    with patch("adapters.ClusterConfig.load") as cluster:
        assert task_runner.main() == 2
    cluster.assert_not_called()


def test_base_model_execution_preserves_requested_subject_and_persists_validator(tmp_path, monkeypatch):
    target = load_target(write_target(tmp_path / "target.yaml"))
    contract_path = write_contract(tmp_path / "contract.yaml", task_type="environment_proof",
                                   model="base-smoke")
    contract = yaml.safe_load(contract_path.read_text())
    profile = {
        "validation": {"base_model": {"name": "base-smoke", "path": "fixture-weights",
                                     "served_model_name": "base-smoke", "required": True,
                                     "max_model_len": 128, "max_num_seqs": 1,
                                     "tensor_parallel_size": 8, "dtype": "float16"}},
        "deployment": {"model_pvc": "fixture-pvc", "xpu_count": 8, "queue": "fixture-queue",
                       "node_pool": "fixture-pool", "image": "fixture-image",
                       "base_manifest": "fixture-manifest"},
        "cluster": {"namespace": "fixture-namespace"},
    }
    # Deliberately invalid proof: the runner must persist the independent
    # validator rejection even when the implementation reports a ready state.
    status = {"state": "ENVIRONMENT_READY", "checks": {}, "artifacts": [], "pod": "fixture-pod"}
    monkeypatch.setenv("USER_ID", "fixture")
    with patch.object(task_runner, "load_yaml", return_value=profile), \
            patch("adapters.ClusterConfig.load", return_value=SimpleNamespace(namespace="fixture-namespace")), \
            patch("core.facade.get_hardware") as hardware, \
            patch("runners.deployment_proof.DeploymentProofRunner") as runner:
        runner.return_value.run.return_value = status
        assert task_runner.execute(contract, contract_path, tmp_path, target=target) == 6
    assert contract["context"]["model"]["name"] == "base-smoke"
    persisted = json.loads((tmp_path / "status.json").read_text())
    assert persisted["target"]["model"] == "demo"
    assert persisted["validator"]["passed"] is False
    assert persisted["validator"]["errors"]


def test_graph_different_target_revision_cannot_resume_old_fact(tmp_path, monkeypatch, capsys):
    target = write_target(tmp_path / "target.yaml",
                          runtime={"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun",
                                   "revisions": {"model": "new"}})
    env = target_environment(bind_subject(load_target(target), "demo"))
    env.update(compatibility_status="supported", model_revision="old")
    (tmp_path / "intake_status.json").write_text('{"state":"INTAKE_READY"}')
    journal = tmp_path / "journal.jsonl"
    record(journal, "ModelRequest", "demo", "INTAKE_READY", tmp_path, env)
    monkeypatch.setattr(sys, "argv", [
        "graph", "--subject", "demo", "--target", str(target), "--journal", str(journal),
        "--artifact-root", str(tmp_path / "artifacts"), "--resume",
        "--until-node", "mat-001-model-intake", "--set", "model_path=fixture-model",
    ])
    assert graph_runner.main() == 0
    output = capsys.readouterr().out
    assert "[plan]" in output
    assert "REUSED" not in output


if __name__ == "__main__":
    unittest.main()
