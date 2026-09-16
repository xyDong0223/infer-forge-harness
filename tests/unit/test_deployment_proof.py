"""Manifest rendering tests for the deployment-proof executor."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from core.storage import ArtifactStore, RunPaths, WritePolicyError
from runners import task_runner
from runners.deployment_proof import DeploymentProofRunner, ActionFailed, manifest_values, render_manifest

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "tasks/kdp-001-deployment-proof/instances/qwen3-8b-p800.yaml"
TEMPLATE = REPO / "tasks/kdp-001-deployment-proof/manifests/feddeployment.template.yaml"


class TestManifestRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
        self.values = manifest_values(self.contract, "20260907T000000Z", "/workspace", "")

    def test_rendered_manifest_has_no_placeholders(self) -> None:
        rendered = render_manifest(TEMPLATE, self.values)
        self.assertNotIn("${", rendered)

    def test_rendered_manifest_is_owned_and_labelled(self) -> None:
        doc = yaml.safe_load(render_manifest(TEMPLATE, self.values))
        self.assertTrue(doc["metadata"]["name"].startswith("dongxinyu03-"))
        self.assertEqual(doc["metadata"]["namespace"], "pd-test")
        self.assertEqual(
            doc["metadata"]["labels"]["infer.kunlun/attempt-id"], "20260907T000000Z"
        )
        container = doc["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["resources"]["limits"]["kunlunxin.com/xpu"], "8")

    def test_missing_value_is_reported_as_contract_invalid(self) -> None:
        incomplete = dict(self.values)
        incomplete.pop("MODEL_PVC")
        with self.assertRaises(ActionFailed) as ctx:
            render_manifest(TEMPLATE, incomplete)
        self.assertEqual(ctx.exception.state, "CONTRACT_INVALID")


def make_runner(path):
    return DeploymentProofRunner(
        yaml.safe_load(CONTRACT.read_text()), SimpleNamespace(config=SimpleNamespace(kubeconfig="fixture")),
        REPO, path,
    )


def test_direct_runner_rejects_source_and_stale_output(tmp_path):
    with pytest.raises(WritePolicyError):
        make_runner(REPO / "artifacts" / "rejected-proof")
    (tmp_path / "status.json").write_text('{"state":"OLD"}')
    with pytest.raises(WritePolicyError, match="fresh"):
        make_runner(tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "OLD"


def test_direct_runner_rejects_interrupted_output_before_resource_side_effects(tmp_path):
    previous = tmp_path / "server.log"
    previous.write_text("original partial evidence")
    with patch.object(DeploymentProofRunner, "preflight") as preflight, \
            patch("runners.deployment_proof.default_runtime") as runtime:
        with pytest.raises(WritePolicyError, match="fresh and empty"):
            make_runner(tmp_path).run()
    preflight.assert_not_called()
    runtime.assert_not_called()
    assert previous.read_text() == "original partial evidence"
    assert list(tmp_path.iterdir()) == [previous]


def test_preallocated_output_allows_current_attempt_inputs_and_logs(tmp_path):
    attempt = RunPaths(tmp_path, "graph-run").allocate_attempt("graph-node")
    ArtifactStore(attempt.input).write_json("commands.json", ["synthetic command"])
    ArtifactStore(attempt.logs).write_text("console.log", "current invocation")
    runner = make_runner(attempt.output)
    assert runner.artifact_dir == attempt.output
    assert (attempt.logs / "console.log").read_text() == "current invocation"


@pytest.mark.parametrize("method", ["write", "write_unique", "append"])
@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt"])
def test_direct_runner_checks_evidence_names(tmp_path, method, name):
    runner = make_runner(tmp_path)
    with pytest.raises(WritePolicyError):
        getattr(runner, method)(name, "rejected")


def test_unique_evidence_never_follows_suffix_symlinks(tmp_path):
    runner = make_runner(tmp_path)
    runner.write_unique("crash.log", "first")
    (tmp_path / "crash.log.1").symlink_to(tmp_path / "unowned")
    with pytest.raises(WritePolicyError):
        runner.write_unique("crash.log", "second")
    assert not (tmp_path / "unowned").exists()


def test_direct_runner_registers_failure_and_refuses_reexecution(tmp_path):
    runner = make_runner(tmp_path)
    runner.phase = "service"
    status = runner.run()
    assert status["state"] == "CONTRACT_INVALID"
    assert json.loads((tmp_path / "status.json").read_text()) == status
    manifest = json.loads(Path(status["manifest_path"]).read_text())
    assert manifest["outcome"] == "CONTRACT_INVALID"
    assert "status.json" in {entry["path"] for entry in manifest["artifacts"]}
    with pytest.raises(WritePolicyError):
        runner.run()


@pytest.mark.parametrize("managed", [False, True])
def test_recovered_proof_retains_crash_directory_without_failing_inventory(tmp_path, capsys, managed):
    def recovered_output(output):
        runner = make_runner(output)
        runner.write("crash/first-attempt.log", "synthetic recovered crash evidence")
        return runner.collect_artifacts("DEPLOYMENT_READY")

    if managed:
        def recovered(contract, target_dir, attach_pod, phase, target, finish):
            status = recovered_output(target_dir)
            status["validator"] = {"passed": True, "errors": []}
            return finish(status, 0)
        with patch.object(task_runner, "_execute", side_effect=recovered):
            assert task_runner.execute(yaml.safe_load(CONTRACT.read_text()), CONTRACT, tmp_path) == 0
        status = json.loads(capsys.readouterr().out)
    else:
        status = recovered_output(tmp_path)
    assert status["state"] == "DEPLOYMENT_READY"
    assert "crash" in status["artifacts"]
    output = Path(status["artifact_root"])
    assert json.loads((output / "status.json").read_text()) == status
    assert (output / "crash/first-attempt.log").read_text() == "synthetic recovered crash evidence"
    manifest = json.loads(Path(status["manifest_path"]).read_text())
    relative = "output/crash/first-attempt.log" if managed else "crash/first-attempt.log"
    assert relative in {entry["path"] for entry in manifest["artifacts"]}
    assert manifest["outcome"] == "DEPLOYMENT_READY"


def test_direct_runner_preserves_inventory_error_when_failure_status_write_fails(tmp_path):
    runner = make_runner(tmp_path)
    inventory_error = OSError("inventory unavailable")
    status_error = OSError("status disk unavailable")
    write_json = runner.store.write_json
    def fail_rewrite(name, payload, **kwargs):
        if kwargs.get("overwrite"):
            raise status_error
        return write_json(name, payload, **kwargs)
    with patch.object(ArtifactStore, "register", side_effect=inventory_error), \
            patch.object(runner.store, "write_json", side_effect=fail_rewrite):
        with pytest.raises(OSError, match="inventory unavailable") as caught:
            runner.collect_artifacts("DEPLOYMENT_READY")
    assert caught.value is inventory_error
    assert caught.value.__cause__ is status_error


def test_task_declared_directory_symlink_is_rejected_before_inventory(tmp_path, capsys):
    contract = yaml.safe_load(CONTRACT.read_text())
    def redirected(contract, target_dir, attach_pod, phase, target, finish):
        (target_dir / "crash").symlink_to(target_dir.parent / "scratch", target_is_directory=True)
        return finish({"state": "DEPLOYMENT_READY", "artifacts": ["crash"]}, 0)
    with patch.object(task_runner, "_execute", side_effect=redirected):
        assert task_runner.execute(contract, CONTRACT, tmp_path) == 2
    status = json.loads(capsys.readouterr().out)
    assert status["state"] == "BLOCKED"
    assert status["manifest_path"] is None


def test_task_output_policy_precedes_cluster_loading(capsys):
    contract = yaml.safe_load(CONTRACT.read_text())
    with patch("adapters.ClusterConfig.load", side_effect=RuntimeError("missing credentials")) as cluster:
        assert task_runner.execute(contract, CONTRACT, REPO / "artifacts" / "rejected") == 2
    cluster.assert_not_called()
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def simulated_execution(contract, target_dir, attach_pod, phase, target, finish):
    (target_dir / "probe.txt").write_text("synthetic evidence")
    return finish({"state": "BLOCKED", "artifacts": ["probe.txt"]}, 2)


def test_task_retries_allocate_attempts_and_default_outside_source(tmp_path, monkeypatch, capsys):
    contract = yaml.safe_load(CONTRACT.read_text())
    contract.pop("artifacts", None)
    monkeypatch.setenv("INFER_FORGE_STATE_ROOT", str(tmp_path))
    with patch.object(task_runner, "_execute", side_effect=simulated_execution):
        assert task_runner.execute(contract, CONTRACT, None, run_id="fixture-run") == 2
        first = json.loads(capsys.readouterr().out)
        before = (Path(first["artifact_root"]) / "status.json").read_bytes()
        assert task_runner.execute(contract, CONTRACT, None, run_id="fixture-run") == 2
        second = json.loads(capsys.readouterr().out)
    assert first["artifact_root"] != second["artifact_root"]
    assert (Path(first["artifact_root"]) / "status.json").read_bytes() == before
    assert first["workspace_identity"]["run_id"] == "fixture-run"
    assert Path(first["artifact_root"]).is_relative_to(tmp_path / "runs" / "fixture-run")
    manifest = json.loads(Path(first["manifest_path"]).read_text())
    assert "output/probe.txt" in {entry["path"] for entry in manifest["artifacts"]}


def test_task_reuses_allocated_output_but_not_other_attempt_directories(tmp_path, capsys):
    contract = yaml.safe_load(CONTRACT.read_text())
    attempt = RunPaths(tmp_path, "fixture-run").allocate_attempt("proof-node")
    with patch.object(task_runner, "_execute", side_effect=simulated_execution):
        assert task_runner.execute(contract, CONTRACT, attempt.output, run_id="fixture-run") == 2
        assert json.loads(capsys.readouterr().out)["artifact_root"] == str(attempt.output)
    with patch("adapters.ClusterConfig.load") as cluster:
        assert task_runner.execute(contract, CONTRACT, attempt.scratch) == 2
        assert "output/" in json.loads(capsys.readouterr().out)["message"]
    cluster.assert_not_called()


def test_task_missing_artifact_cannot_publish_success(tmp_path, capsys):
    contract = yaml.safe_load(CONTRACT.read_text())
    def missing(contract, target_dir, attach_pod, phase, target, finish):
        return finish({"state": "DEPLOYMENT_READY", "artifacts": ["missing.txt"]}, 0)
    with patch.object(task_runner, "_execute", side_effect=missing):
        assert task_runner.execute(contract, CONTRACT, tmp_path) == 2
    status = json.loads(capsys.readouterr().out)
    assert status["state"] == "BLOCKED"
    assert status["validator"]["passed"] is False
    assert json.loads(Path(status["manifest_path"]).read_text())["outcome"] == "BLOCKED"


def test_task_executor_failure_is_registered(tmp_path, capsys):
    contract = yaml.safe_load(CONTRACT.read_text())
    with patch.object(task_runner, "_execute", side_effect=RuntimeError("configuration unavailable")):
        assert task_runner.execute(contract, CONTRACT, tmp_path) == 2
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "BLOCKED"
    assert json.loads(Path(status["manifest_path"]).read_text())["outcome"] == "BLOCKED"


def test_plan_cannot_write_source(tmp_path, monkeypatch, capsys):
    output = REPO / "rejected-plan.json"
    monkeypatch.setattr("sys.argv", ["task", str(CONTRACT), "--output", str(output)])
    assert task_runner.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"
    assert not output.exists()


def test_plan_external_output_is_atomic_and_not_overwritten(tmp_path, monkeypatch, capsys):
    output = tmp_path / "plans" / "plan.json"
    monkeypatch.setattr("sys.argv", ["task", str(CONTRACT), "--output", str(output)])
    assert task_runner.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == plan
    before = output.read_bytes()
    assert task_runner.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"
    assert output.read_bytes() == before
    assert not list(output.parent.glob("*.pending"))


if __name__ == "__main__":
    unittest.main()
