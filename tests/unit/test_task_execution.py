"""Task descriptor loading is strict, read-only and independent of Graph imports."""

from dataclasses import FrozenInstanceError
import hashlib

import jsonschema
import pytest
import yaml

from core.paths import REPO_ROOT
from core.task_execution import TaskExecutionError, default_execution_catalog, load_execution_catalog, load_task


def definition():
    return {
        "api_version": "infer.kunlun/v1alpha1", "kind": "Task",
        "metadata": {"name": "simple", "task_type": "simple", "version": "1"},
        "spec": {"produces": "OutputFact", "consumes": ["InputFact"],
                 "execution_descriptor": {
                     "schema_version": 1,
                     "argv": ["python3", "cli/discovery/scan_model_support.py", "--out", "{artifacts}"],
                     "inputs": {"--input": {"kind": "InputFact", "artifact": "report.json"}},
                     "status_file": "status.json", "success_exit_keys": ["pass"],
                 }},
        "actions": ["read_input"], "acceptance": {"evidence_required": True},
        "artifacts": ["status.json"], "validator": "validators/scan_validator.py:validate_support_card",
        "exit_states": {"pass": "READY", "blocked": "BLOCKED"},
    }


def write_definition(tmp_path, body=None, name="simple"):
    path = tmp_path / "tasks" / name / "task.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body or definition(), sort_keys=False))
    return path


def test_one_simple_descriptor_is_frozen_typed_and_readonly(tmp_path):
    path = write_definition(tmp_path)
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    descriptor = load_task(path)
    assert descriptor.task_type == "simple"
    assert descriptor.success_states == ("READY",)
    assert descriptor.inputs[0].artifact == "report.json"
    assert descriptor.task_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert descriptor.to_node_spec()["needs"] == {"--input": "fact:InputFact:report.json"}
    view = descriptor.to_node_spec()
    view["command"].append("--arbitrary")
    assert "--arbitrary" not in descriptor.argv
    with pytest.raises(FrozenInstanceError):
        descriptor.task_type = "changed"
    with pytest.raises(TypeError):
        descriptor.exit_states["pass"] = "FALSE_READY"
    assert {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()} == before


def test_default_consumers_share_one_immutable_process_snapshot():
    first = default_execution_catalog()
    assert default_execution_catalog() is first
    with pytest.raises(TypeError):
        first["invented"] = next(iter(first.values()))


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("schema_version", 2), ("argv", "python3 command.py"),
    ("argv", ["sh", "-c", "echo unsafe"]), ("argv", ["python3", "../outside.py"]),
    ("argv", ["python3", "cli/../unsafe.py"]), ("status_file", "../status.json"),
    ("status_file", "/tmp/status.json"), ("status_file", "{artifacts}/status.json"),
    ("success_exit_keys", []), ("success_exit_keys", ["invented"]),
    ("success_exit_keys", ["pass", "pass"]), ("extra_dsl", {"if": True}),
])
def test_descriptor_rejects_unsafe_or_unsupported_shape(tmp_path, key, value):
    body = definition()
    body["spec"]["execution_descriptor"][key] = value
    with pytest.raises(TaskExecutionError):
        load_task(write_definition(tmp_path, body))


@pytest.mark.parametrize("token", ["{unknown}", "{subject.attr}", "{subject[0]}",
                                    "{subject!r}", "{subject:>10}", "{", "{}"])
def test_argv_templates_do_not_evaluate_attributes_expressions_or_formats(tmp_path, token):
    body = definition()
    body["spec"]["execution_descriptor"]["argv"][-1] = token
    with pytest.raises(TaskExecutionError, match="placeholder"):
        load_task(write_definition(tmp_path, body))


@pytest.mark.parametrize("binding", [
    {"kind": "InputFact", "artifact": "../report.json"},
    {"kind": "InputFact", "artifact": "report.json", "optional": "false"},
    {"kind": "UnknownFact", "artifact": "report.json"},
    {"kind": "InputFact", "artifact": "report.json", "shell": "eval"},
])
def test_input_bindings_are_explicit_and_path_safe(tmp_path, binding):
    body = definition()
    body["spec"]["execution_descriptor"]["inputs"]["--input"] = binding
    with pytest.raises(TaskExecutionError):
        load_task(write_definition(tmp_path, body))


def test_optional_fact_does_not_become_a_required_semantic_dependency(tmp_path):
    body = definition()
    body["spec"]["execution_descriptor"]["inputs"]["--optional"] = {
        "kind": "OptionalFact", "artifact": "extra.json", "optional": True,
    }
    descriptor = load_task(write_definition(tmp_path, body))
    assert descriptor.consumes == ("InputFact",)
    assert descriptor.to_node_spec()["optional"] == {"--optional": "fact:OptionalFact:extra.json"}


def test_standalone_task_is_readable_without_becoming_graph_executable(tmp_path):
    body = definition()
    body.pop("spec")
    body.pop("validator")
    descriptor = load_task(write_definition(tmp_path, body))
    assert descriptor.executable is False
    assert load_execution_catalog(tmp_path) == {}
    with pytest.raises(TaskExecutionError, match="no execution descriptor"):
        descriptor.to_node_spec()


@pytest.mark.parametrize("duplicate", ["task_type", "task_id", "fact"])
def test_catalog_rejects_ambiguous_identity(tmp_path, duplicate):
    write_definition(tmp_path)
    body = definition()
    body["metadata"].update(name="other", task_type="other")
    body["spec"]["produces"] = "OtherFact"
    if duplicate == "task_type":
        body["metadata"]["task_type"] = "simple"
    elif duplicate == "task_id":
        body["metadata"]["name"] = "simple"
    else:
        body["spec"]["produces"] = "OutputFact"
    write_definition(tmp_path, body, "other")
    with pytest.raises(TaskExecutionError, match="duplicate|ambiguous"):
        load_execution_catalog(tmp_path)


def test_source_definition_schema_matches_every_versioned_task():
    schema = yaml.safe_load((REPO_ROOT / "contracts/task_definition.schema.yaml").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    paths = sorted((REPO_ROOT / "tasks").glob("*/task.yaml"))
    assert len(paths) == 26
    for path in paths:
        errors = sorted(validator.iter_errors(yaml.safe_load(path.read_text())), key=lambda item: str(item.path))
        assert not errors, f"{path}: {[error.message for error in errors]}"
        assert load_task(path).task_path == path


def test_catalog_contains_24_typed_nodes_and_retains_special_success_semantics():
    catalog = load_execution_catalog()
    assert len(catalog) == 24
    assert "deployment_proof" not in catalog and "accuracy_smoke" not in catalog
    assert catalog["runtime_drift_scan"].success_states == ("DRIFT_CLEAR", "DRIFT_FOUND")
    assert catalog["operator_task_dispatch"].success_states == ("DISPATCHED", "DISPATCH_SKIPPED")
    assert catalog["torch_shim_handoff"].success_states == ("HANDOFF_CLEAR",)
    assert catalog["operator_candidate_integration"].success_states == (
        "WAITING_FOR_CANDIDATE", "READY_FOR_INTEGRATION", "OPERATORS_READY",
    )
    assert "fan_out" not in catalog["capability_evaluation"].to_node_spec()
    assert catalog["memory_budget"].consumes == ("DeploymentProof",)
    assert "needs" not in catalog["memory_budget"].to_node_spec()


def test_changing_one_task_changes_its_command_and_gate_without_another_mapping(tmp_path):
    body = definition()
    path = write_definition(tmp_path, body)
    before = load_execution_catalog(tmp_path)["simple"]
    body["spec"]["execution_descriptor"]["argv"] += ["--subject", "{subject}"]
    body["spec"]["execution_descriptor"]["status_file"] = "new_status.json"
    body["spec"]["produces"] = "RevisedFact"
    body["exit_states"]["pass"] = "REVISED_READY"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    after = load_execution_catalog(tmp_path)["simple"]
    assert after.task_sha256 != before.task_sha256
    assert after.to_node_spec()["command"][-2:] == ["--subject", "{subject}"]
    assert after.status_file == "new_status.json" and after.produces == "RevisedFact"
    assert after.success_states == ("REVISED_READY",)


def test_documentation_shell_text_is_never_an_execution_source(tmp_path, monkeypatch):
    body = definition()
    body["spec"]["runs_with"] = "bash -c 'this must never execute'"
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: pytest.fail("loader executed a process"))
    descriptor = load_task(write_definition(tmp_path, body))
    assert descriptor.argv[0] == "python3"
    assert "bash" not in str(descriptor.to_node_spec())


def test_source_paths_cannot_alias_symlinks_or_parent_traversal(tmp_path):
    path = write_definition(tmp_path)
    link = tmp_path / "linked.yaml"
    link.symlink_to(path)
    with pytest.raises(TaskExecutionError, match="regular file"):
        load_task(link)
    directory = tmp_path / "linked-tasks"
    directory.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(TaskExecutionError, match="symbolic links"):
        load_task(directory / "task.yaml")
    with pytest.raises(TaskExecutionError, match="parent directories"):
        load_task(path.parent / ".." / "simple" / "task.yaml")


@pytest.mark.parametrize("key,value", [("api_version", "unversioned"), ("acceptance", {}),
                                      ("artifacts", None), ("kind", "Workflow")])
def test_runtime_rejects_missing_minimal_task_contract(tmp_path, key, value):
    body = definition()
    body[key] = value
    with pytest.raises(TaskExecutionError):
        load_task(write_definition(tmp_path, body))
