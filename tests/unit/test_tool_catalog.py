"""Tool identities delegate Graph execution metadata to source Task definitions."""

from copy import deepcopy

import pytest
import yaml

from cli.maintenance.check_repo_references import check_tool_catalog, scan
from core.paths import REPO_ROOT
from core.task_execution import load_execution_catalog, load_task

PROBE_SCRIPT = "cli/" + "probe.py"


def test_builtin_catalog_resolves_tasks_without_duplicate_commands():
    entries = yaml.safe_load((REPO_ROOT / "catalog/tool_catalog.yaml").read_text())["entries"]
    assert check_tool_catalog(REPO_ROOT) == []
    assert len(entries) == len({entry["id"] for entry in entries}) == 21
    independent = {"instrument_kernel_trace", "apply_torch_decode_patch", "tensor_diff"}
    assert {entry["id"] for entry in entries if "command" in entry} == independent
    executions = load_execution_catalog()
    for entry in entries:
        assert entry["purpose"] and entry["outputs"]
        assert type(entry["retryable"]) is bool
        assert entry["side_effect"] in {"read_only", "pod_exec", "pod_write", "cluster_write", "write_artifact"}
        if entry["id"] in independent:
            continue
        if entry["id"] == "deployment_proof":
            descriptors = [load_task(REPO_ROOT / value) for value in entry["task_definitions"]]
            assert {item.task_type for item in descriptors} == {"environment_proof", "service_proof"}
            assert entry["side_effect"] == "cluster_write" and entry["retryable"] is False
        else:
            descriptor = load_task(REPO_ROOT / entry["task_definition"])
            assert descriptor == executions[entry["id"]]
    dispatch = next(entry for entry in entries if entry["id"] == "operator_task_dispatch")
    assert dispatch["outputs"] == ["dispatch_status.json", "requests/"]
    assert dispatch["scheduler_mode"]["side_effect"] == "write_scheduler_state"
    integration = next(entry for entry in entries if entry["id"] == "operator_candidate_integration")
    assert integration["scheduler_mode"]["waiting_is_success"] is False


@pytest.fixture
def catalog_source(tmp_path):
    task = {
        "api_version": "infer.kunlun/v1alpha1", "kind": "Task",
        "metadata": {"name": "catalog-probe", "task_type": "catalog_probe"},
        "spec": {"produces": "Probe", "consumes": [], "execution_descriptor": {
            "schema_version": 1, "argv": ["python3", PROBE_SCRIPT, "--out", "{artifacts}"],
            "inputs": {}, "status_file": "status.json", "success_exit_keys": ["ready"],
        }},
        "validator": "validators/" + "probe.py:validate", "exit_states": {"ready": "PROBE_READY"},
        "acceptance": {"output": "present"}, "artifacts": ["status.json"],
    }
    task_path = tmp_path / "tasks/probe/task.yaml"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(yaml.safe_dump(task))
    for name in (PROBE_SCRIPT, "validators/" + "probe.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_text("raise RuntimeError('metadata lookup must not execute code')\n")
    catalog = tmp_path / "catalog/tool_catalog.yaml"
    catalog.parent.mkdir()
    entry = {"id": "probe", "task_definition": "tasks/probe/task.yaml",
             "purpose": "metadata-only fixture", "side_effect": "read_only",
             "retryable": True, "outputs": ["status.json"]}

    def save(entry_value):
        catalog.write_text(yaml.safe_dump({"kind": "ToolCatalog", "entries": [entry_value]}))
    save(entry)
    return tmp_path, task_path, task, catalog, entry, save


def test_changing_only_task_argv_is_visible_through_unchanged_catalog(catalog_source):
    root, task_path, task, catalog, entry, _ = catalog_source
    original_catalog = catalog.read_bytes()
    before = load_task(root / entry["task_definition"])
    task["spec"]["execution_descriptor"]["argv"] += ["--subject", "{subject}"]
    task_path.write_text(yaml.safe_dump(task))
    after = load_task(root / entry["task_definition"])
    assert after.argv == (*before.argv, "--subject", "{subject}")
    assert after.task_sha256 != before.task_sha256
    assert catalog.read_bytes() == original_catalog
    assert scan(root) == []


@pytest.mark.parametrize("mutation", [
    lambda entry: entry.update(command=f"python3 {PROBE_SCRIPT}"),
    lambda entry: entry.update(task_definition="tasks/missing/task.yaml"),
    lambda entry: entry.update(task_definition="../outside/task.yaml"),
    lambda entry: entry.update(task_definition="/absolute/task.yaml"),
    lambda entry: entry.update(task_definition=None),
])
def test_catalog_rejects_ambiguous_or_invalid_task_reference(catalog_source, mutation):
    root, _, _, _, entry, save = catalog_source
    mutation(entry)
    save(entry)
    assert check_tool_catalog(root)


def test_catalog_rejects_nonexecutable_task_and_missing_executable(catalog_source):
    root, task_path, task, _, _, _ = catalog_source
    original = deepcopy(task)
    task["spec"].pop("execution_descriptor")
    task_path.write_text(yaml.safe_dump(task))
    assert "no execution descriptor" in " ".join(check_tool_catalog(root))
    task_path.write_text(yaml.safe_dump(original))
    (root / PROBE_SCRIPT).unlink()
    assert "entrypoint does not exist" in " ".join(check_tool_catalog(root))


@pytest.mark.parametrize("references", [[], "tasks/probe/task.yaml",
                                       ["tasks/probe/task.yaml", "tasks/probe/task.yaml"]])
def test_catalog_rejects_empty_or_duplicate_multitask_references(catalog_source, references):
    root, _, _, _, entry, save = catalog_source
    entry.pop("task_definition")
    entry["task_definitions"] = references
    save(entry)
    assert check_tool_catalog(root)


def test_catalog_standalone_command_is_checked_without_execution(catalog_source):
    root, _, _, _, entry, save = catalog_source
    entry.pop("task_definition")
    entry["command"] = f"python3 {PROBE_SCRIPT} --inspect"
    save(entry)
    assert check_tool_catalog(root) == []
    entry["command"] = "python3 cli/" + "missing.py"
    save(entry)
    assert "missing or invalid standalone command" in " ".join(check_tool_catalog(root))
