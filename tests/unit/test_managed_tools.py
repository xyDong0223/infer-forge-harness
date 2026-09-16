"""CLI output ownership without running probes or touching a cluster."""

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

from core import storage
from cli.common import run_managed_tool
from operations.deployment.plan_deployment import render_instance


REPO_ROOT = Path(__file__).resolve().parents[2]
DIRECTORY_TOOLS = {
    "accuracy_differential": "cli/validation/accuracy_differential.py",
    "check_api_conformance": "cli/validation/check_api_conformance.py",
    "classify_gaps": "cli/discovery/classify_gaps.py",
    "evaluate_capability": "cli/discovery/evaluate_capability.py",
    "match_capabilities": "cli/discovery/match_capabilities.py",
    "memory_budget": "cli/deployment/memory_budget.py",
    "model_intake": "cli/intake/model_intake.py",
    "operator_lifecycle": "cli/operators/operator_lifecycle.py",
    "plan_deployment": "cli/deployment/plan_deployment.py",
    "scan_model_support": "cli/discovery/scan_model_support.py",
    "scan_runtime_drift": "cli/discovery/scan_runtime_drift.py",
    "scan_torch_shims": "cli/discovery/scan_torch_shims.py",
    "toy_bringup": "cli/deployment/toy_bringup.py",
    "update_support_matrix": "cli/validation/update_support_matrix.py",
    "vendor_handoff": "cli/operators/vendor_handoff.py",
}
EXECUTOR_COMMANDS = {
    "triage_executor": "cli/operators/triage.py",
    "patch_executor": "cli/operators/place_patch.py",
    "correctness_executor": "cli/validation/correctness.py",
}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(storage, "REPO_ROOT", source)
    return tmp_path


def invoke(monkeypatch, main, *args, task_id="test-tool"):
    monkeypatch.setattr(sys, "argv", ["tool", *map(str, args)])
    return run_managed_tool(main, task_id=task_id)


@pytest.mark.parametrize("spelling", ["--out", "--out=", "--ou", "--o="])
def test_source_output_is_rejected_before_task(runtime, monkeypatch, spelling):
    main = Mock()
    out = runtime / "source" / "artifacts"
    args = [spelling + str(out)] if spelling.endswith("=") else [spelling, out]
    with pytest.raises(storage.WritePolicyError):
        invoke(monkeypatch, main, *args)
    main.assert_not_called()
    assert not out.exists()


@pytest.mark.parametrize("equals", [False, True])
def test_direct_output_is_exact_and_cannot_be_reused(runtime, monkeypatch, equals):
    out = runtime / "legacy-output"
    args = [f"--out={out}"] if equals else ["--out", out]

    def main():
        assert out.is_dir()
        (out / "result.json").write_text('{"observed": true}')
        return 0

    assert invoke(monkeypatch, main, "dispatch", "--subject", "model", *args) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["outcome"] == "COMPLETED"
    assert manifest["identity"]["task_id"] == "test-tool"
    assert manifest["identity"]["run_id"].startswith("local-")
    assert "result.json" in {item["path"] for item in manifest["artifacts"]}
    original = (out / "manifest.json").read_bytes()
    retry = Mock()
    with pytest.raises(storage.WritePolicyError, match="fresh"):
        invoke(monkeypatch, retry, *args)
    retry.assert_not_called()
    assert (out / "manifest.json").read_bytes() == original


def test_empty_existing_directory_can_be_claimed(runtime, monkeypatch):
    out = runtime / "empty"
    out.mkdir()
    invoke(monkeypatch, lambda: 0, "--out", out)
    assert (out / ".tool-invocation.json").is_file()


@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_quoted_home_output_is_normalized_and_argv_restored(runtime, monkeypatch, equals, fails):
    home = runtime / "home"
    source = runtime / "source"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(source)
    out = home / "output"
    original_argv = ["tool", "--out=~/output"] if equals else ["tool", "--out", "~/output"]
    monkeypatch.setattr(sys, "argv", original_argv)
    original_values = original_argv.copy()

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args()
        assert args.out == out.resolve()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "result.json").write_text("{}")
        if fails:
            raise RuntimeError("task failed")
        return 0

    if fails:
        with pytest.raises(RuntimeError, match="task failed"):
            run_managed_tool(main, task_id="test-tool")
    else:
        assert run_managed_tool(main, task_id="test-tool") == 0
    assert sys.argv is original_argv
    assert sys.argv == original_values
    assert not (source / "~").exists()
    assert (out / "result.json").is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["outcome"] == ("ERROR" if fails else "COMPLETED")


def test_failed_return_registers_partial_output_without_scratch(runtime, monkeypatch):
    out = runtime / "failed"

    def main():
        (out / "failure.log").write_text("original failure")
        (out / "scratch").mkdir()
        (out / "scratch" / "notes.txt").write_text("not formal evidence")
        return 7

    assert invoke(monkeypatch, main, "--out", out) == 7
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["outcome"] == "FAILED"
    paths = {item["path"] for item in manifest["artifacts"]}
    assert "failure.log" in paths
    assert "manifest.json" not in paths
    assert not any("scratch" in path for path in paths)


def test_exception_registers_error_and_preserves_original(runtime, monkeypatch):
    out = runtime / "exception"
    error = RuntimeError("original task error")

    def main():
        (out / "failure.log").write_text(str(error))
        raise error

    with pytest.raises(RuntimeError) as caught:
        invoke(monkeypatch, main, "--out", out)
    assert caught.value is error
    assert json.loads((out / "manifest.json").read_text())["outcome"] == "ERROR"


def test_manifest_failure_does_not_mask_task_error(runtime, monkeypatch):
    error = RuntimeError("original task error")
    monkeypatch.setattr(storage.ArtifactStore, "register", Mock(side_effect=OSError("disk full")))
    with pytest.raises(RuntimeError) as caught:
        invoke(monkeypatch, Mock(side_effect=error), "--out", runtime / "failed")
    assert caught.value is error
    if hasattr(error, "__notes__"):
        assert "disk full" in error.__notes__[0]


def test_secondary_manifest_failure_is_reported_without_exception_notes(runtime, monkeypatch, capsys):
    class LegacyError(RuntimeError):
        add_note = None

    error = LegacyError("original task error")
    monkeypatch.setattr(storage.ArtifactStore, "register", Mock(side_effect=OSError("disk full")))
    with pytest.raises(LegacyError) as caught:
        invoke(monkeypatch, Mock(side_effect=error), "--out", runtime / "failed")
    assert caught.value is error
    assert "artifact registration also failed: disk full" in capsys.readouterr().err


def test_manifest_failure_is_visible_after_normal_return(runtime, monkeypatch):
    monkeypatch.setattr(storage.ArtifactStore, "register", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        invoke(monkeypatch, lambda: 0, "--out", runtime / "failed")


def test_graph_output_uses_attempt_identity(runtime, monkeypatch):
    attempt = storage.RunPaths(runtime / "run", "run-1").allocate_attempt("graph-node")
    (attempt.scratch / "investigation.txt").write_text("not evidence")
    out = attempt.output / "dimension"

    def main():
        out.mkdir()
        (out / "result.json").write_text("{}")
        return 1

    assert invoke(monkeypatch, main, "--out", out) == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["identity"] == {**attempt.identity, "tool_id": "test-tool"}
    assert [item["path"] for item in manifest["artifacts"]] == ["result.json"]
    assert not (out / ".tool-invocation.json").exists()


@pytest.mark.parametrize("directory", ["root", "input", "scratch", "logs"])
def test_attempt_non_output_directories_are_rejected(runtime, monkeypatch, directory):
    attempt = storage.RunPaths(runtime / "run", "run-1").allocate_attempt("graph-node")
    main = Mock()
    with pytest.raises(storage.WritePolicyError, match="owning attempt output"):
        invoke(monkeypatch, main, "--out", getattr(attempt, directory))
    main.assert_not_called()


def test_attempt_symlink_escape_is_not_a_standalone_output(runtime, monkeypatch):
    attempt = storage.RunPaths(runtime / "run", "run-1").allocate_attempt("graph-node")
    outside = runtime / "outside"
    outside.mkdir()
    (attempt.output / "escape").symlink_to(outside, target_is_directory=True)
    main = Mock()
    with pytest.raises(storage.WritePolicyError, match="owning attempt output"):
        invoke(monkeypatch, main, "--out", attempt.output / "escape" / "new")
    main.assert_not_called()
    assert not list(outside.iterdir())


def test_attempt_output_symlink_is_rejected(runtime, monkeypatch):
    attempt = storage.RunPaths(runtime / "run", "run-1").allocate_attempt("graph-node")
    outside = runtime / "outside"
    outside.mkdir()
    attempt.output.rmdir()
    attempt.output.symlink_to(outside, target_is_directory=True)
    main = Mock()
    with pytest.raises(storage.WritePolicyError, match="owning attempt output"):
        invoke(monkeypatch, main, "--out", attempt.output)
    main.assert_not_called()
    assert not list(outside.iterdir())


@pytest.mark.parametrize("args", [[], ["--help"], ["--list-dimensions"]])
def test_read_only_modes_pass_through(runtime, monkeypatch, args):
    main = Mock(return_value=0)
    assert invoke(monkeypatch, main, *args, task_id="mat-008-capability-evaluation") == 0
    main.assert_called_once_with()


def test_list_dimensions_with_out_does_not_claim_output(runtime, monkeypatch):
    match = runtime / "match.json"
    match.write_text('{"axes": []}')
    out = runtime / "unused"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_capability", "--list-dimensions", "--capability-match", str(match),
        "--out", str(out),
    ])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(REPO_ROOT / "cli" / "discovery" / "evaluate_capability.py"), run_name="__main__")
    assert caught.value.code == 0
    assert not out.exists()


@pytest.mark.parametrize("tool", DIRECTORY_TOOLS)
def test_entrypoint_rejects_source_output_before_parsing_task_arguments(tool):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / DIRECTORY_TOOLS[tool]),
         f"--out={REPO_ROOT / 'forbidden-runtime-output'}"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "runtime path overlaps the source repository" in result.stderr
    assert not (REPO_ROOT / "forbidden-runtime-output").exists()


@pytest.mark.parametrize("runner", EXECUTOR_COMMANDS)
def test_executor_entrypoints_reject_source_output_before_cluster_access(runner):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / EXECUTOR_COMMANDS[runner]),
         f"--out={REPO_ROOT / 'forbidden-runtime-output'}"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "runtime path overlaps the source repository" in result.stderr
    assert not (REPO_ROOT / "forbidden-runtime-output").exists()


def deployment_report():
    return {
        "subject": "org/model", "hardware": "P800",
        "parameters": [
            {"parameter": name, "value": value, "source": "test fixture"}
            for name, value in {
                "max_model_len": 1024, "tensor_parallel_size": 1, "dtype": "bfloat16",
                "block_size": 16, "gpu_memory_utilization": 0.9,
            }.items()
        ],
    }


def test_generated_contract_uses_external_run_root(runtime, monkeypatch):
    monkeypatch.setenv("INFER_FORGE_STATE_ROOT", str(runtime / "state"))
    rendered = yaml.safe_load(render_instance(deployment_report(), {"model": {}}))
    root = Path(rendered["artifacts"]["directory"])
    assert root == runtime / "state" / "runs" / storage.safe_component("kdp-001-org/model")
    assert not root.exists()
    assert not root.is_relative_to(REPO_ROOT / "artifacts")


def test_generated_contract_accepts_explicit_external_run_root(runtime):
    out = runtime / "service-run"
    rendered = yaml.safe_load(render_instance(
        deployment_report(), {"model": {}}, runtime_artifact_root=out,
    ))
    assert rendered["artifacts"]["directory"] == str(out)
    assert not out.exists()
    with pytest.raises(storage.WritePolicyError):
        render_instance(
            deployment_report(), {"model": {}},
            runtime_artifact_root=runtime / "source" / "artifacts",
        )
