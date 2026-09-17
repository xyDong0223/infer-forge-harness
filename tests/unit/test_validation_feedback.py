"""Offline feedback integrity, privacy and explicit non-promotion boundaries."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import xml.etree.ElementTree as ET

import pytest
import yaml
from jsonschema import Draft202012Validator

from core.paths import REPO_ROOT
from operations.validation import feedback


SECRET = "PRIVATE_TOKEN_AND_USER_PATH_DO_NOT_SHARE"


@pytest.fixture
def plan(tmp_path, monkeypatch):
    source = tmp_path / "source"
    test = source / "tests/e2e/test_model_adaptation_hardware.py"
    test.parent.mkdir(parents=True)
    test.write_text("# fixture source; no executable is invoked by the planner\n")
    matrix = source / "compatibility/matrix.yaml"
    matrix.parent.mkdir()
    matrix.write_text("schema_version: 1\n")
    (source / "pyproject.toml").write_text('[project]\nversion = "fixture"\n')
    monkeypatch.setattr(feedback, "REPO_ROOT", source)
    scenario = tmp_path / "scenario.json"
    value = {"run_id": SECRET, "state": str(tmp_path / "missing-private-db.sqlite"),
             "artifact_root": str(tmp_path / "inner-run"), "subject": SECRET, "pod": SECRET,
             "namespace": SECRET, "image_digest": "sha256:" + "1" * 64,
             "hardware": "P800", "model_revision": SECRET, "plugin_revision": SECRET,
             "cleanup_policy": "retain_prepared_pod", "environment": {"endpoint": SECRET},
             "context": {"user_id": SECRET, "model_path": SECRET}, "private_credential": SECRET}
    scenario.write_text(json.dumps(value))
    result = feedback.create_validation_plan(scenario, "device_smoke", tmp_path / "plan")
    return {"root": tmp_path, "scenario": scenario, "source": source, "value": value, **result}


def junit(plan, *, outcome="passed", binding="full"):
    flags = {"failures": int(outcome == "failed"), "errors": int(outcome == "error"),
             "skipped": int(outcome == "skipped")}
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite", tests="1", **{key: str(value) for key, value in flags.items()})
    case = ET.SubElement(suite, "testcase", name="test_prepared_device_smoke",
                         classname="tests.e2e.test_model_adaptation_hardware", file=SECRET)
    properties = ET.SubElement(case, "properties")
    if binding:
        for phase in ("before", "after") if binding == "full" else ("before",):
            for key, value in feedback.execution_properties(plan["request_path"], plan["scenario"], phase=phase):
                ET.SubElement(properties, "property", name=key, value=value)
    ET.SubElement(properties, "property", name="artifact_root", value=SECRET)
    if outcome != "passed":
        tag = {"failed": "failure", "error": "error", "skipped": "skipped"}[outcome]
        ET.SubElement(case, tag, message=SECRET).text = SECRET
    ET.SubElement(case, "system-out").text = SECRET
    path = plan["root"] / "junit.xml"
    ET.ElementTree(suites).write(path, encoding="utf-8", xml_declaration=True)
    return path, suites


def export(plan, *, outcome="passed", binding="full", exit_code=None):
    path, _ = junit(plan, outcome=outcome, binding=binding)
    if exit_code is None:
        exit_code = 1 if outcome in {"failed", "error"} else 0
    return feedback.export_feedback(plan["request_path"], path, exit_code, plan["root"] / "share")


def rewrite_feedback(result, mutate):
    root = Path(result["feedback_dir"])
    path = root / "feedback.json"
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 size_bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))


def test_plan_does_not_read_or_create_authoritative_state_and_separates_private_inputs(plan):
    assert not Path(plan["value"]["state"]).exists()
    assert not Path(plan["value"]["artifact_root"]).exists()
    assert plan["execution_authorized"] is False
    private = json.loads(Path(plan["private_plan_path"]).read_text())
    assert "INFER_FORGE_RUN_DEVICE_SMOKE" not in private["environment"]
    assert private["authorization_required"]["must_be_explicitly_authorized"] is True
    assert private["argv"][6] == plan["request"]["case_id"]
    assert private["environment"]["INFER_FORGE_VALIDATION_REQUEST"] == plan["request_path"]
    assert SECRET not in Path(plan["request_path"]).read_text()
    assert str(plan["root"]) not in Path(plan["request_path"]).read_text()
    assert stat.S_IMODE(Path(plan["private_plan_path"]).stat().st_mode) == 0o600
    private_scenario = Path(private["environment"]["INFER_FORGE_HARDWARE_SCENARIO"])
    assert json.loads(private_scenario.read_text()) == plan["value"]


def test_schema_privacy_and_consistency_are_not_promotion(plan):
    result = export(plan)
    root = Path(result["feedback_dir"])
    assert sorted(path.name for path in root.iterdir()) == ["feedback.json", "manifest.json"]
    for path in root.iterdir():
        assert SECRET not in path.read_text()
        assert str(plan["root"]) not in path.read_text()
    schema = yaml.safe_load((REPO_ROOT / "contracts/validation_feedback.schema.yaml").read_text())
    for definition, value in (("request", plan["request"]), ("feedback", result["summary"]),
                              ("manifest", json.loads((root / "manifest.json").read_text()))):
        Draft202012Validator({**schema, "$ref": f"#/$defs/{definition}"}).validate(value)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    checked = feedback.check_feedback(plan["request_path"], root)
    assert checked["status"] == "FEEDBACK_VALID"
    assert checked["observation"]["state"] == "TEST_PASSED"
    assert checked["observations_verified"] is False and checked["promoted"] is False
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert not Path(plan["value"]["state"]).exists()


@pytest.mark.parametrize("outcome,binding,code,state,reason", [
    ("skipped", None, 0, "NOT_RUN", "SKIPPED"),
    ("passed", None, 0, "BLOCKED", "EXECUTION_BINDING_MISSING"),
    ("passed", "before", 0, "BLOCKED", "EXECUTION_BINDING_MISSING"),
    ("failed", "full", 1, "TEST_FAILED", "TEST_FAILURE"),
    ("error", "full", 1, "TEST_FAILED", "TEST_ERROR"),
    ("passed", "full", 1, "BLOCKED", "EXIT_STATUS_MISMATCH"),
    ("failed", "full", 0, "BLOCKED", "EXIT_STATUS_MISMATCH"),
])
def test_executed_skipped_and_unbound_outcomes_are_distinct(plan, outcome, binding, code, state, reason):
    result = export(plan, outcome=outcome, binding=binding, exit_code=code)
    observed = result["summary"]["observation"]
    assert observed["state"] == state and observed["reason_code"] == reason
    checked = feedback.check_feedback(plan["request_path"], result["feedback_dir"])
    assert checked["status"] == "FEEDBACK_VALID" and checked["promoted"] is False


@pytest.mark.parametrize("mutation", [
    lambda tree: tree.find(".//testcase").set("name", "test_another_case"),
    lambda tree: tree.find(".//testsuite").append(deepcopy(tree.find(".//testcase"))),
    lambda tree: tree.find(".//testsuite").set("tests", "0"),
    lambda tree: tree.find(".//property").set("value", "0" * 64),
    lambda tree: tree.find(".//properties").append(deepcopy(tree.find(".//property"))),
])
def test_wrong_case_counts_duplicate_or_wrong_binding_is_rejected(plan, mutation):
    path, tree = junit(plan)
    mutation(tree)
    ET.ElementTree(tree).write(path, encoding="utf-8")
    with pytest.raises(feedback.FeedbackError):
        feedback.export_feedback(plan["request_path"], path, 0, plan["root"] / "share")
    assert not (plan["root"] / "share").exists()


@pytest.mark.parametrize("content", [b"not XML", b"<!DOCTYPE root><testsuites/>",
                                      '<!DOCTYPE root><testsuites/>'.encode("utf-16")])
def test_unsafe_or_invalid_xml_is_rejected(plan, content):
    path = plan["root"] / "invalid.xml"
    path.write_bytes(content)
    with pytest.raises(feedback.FeedbackError):
        feedback.export_feedback(plan["request_path"], path, 0, plan["root"] / "share")


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra=SECRET),
    lambda value: value.update(promoted=True),
    lambda value: value["source"].update(host=SECRET),
    lambda value: value["observation"].update(raw_traceback=SECRET),
    lambda value: value["observation"].update(state="FUNCTIONAL_READY"),
    lambda value: value["observation"]["test_counts"].update(passed=True),
    lambda value: value["evidence"]["junit"].update(content_included=True),
])
def test_even_rehashed_unknown_fields_and_promotions_are_rejected(plan, mutation):
    result = export(plan)
    rewrite_feedback(result, mutation)
    checked = feedback.check_feedback(plan["request_path"], result["feedback_dir"])
    assert checked["status"] == "REJECTED" and checked["promoted"] is False
    assert SECRET not in json.dumps(checked)


def test_different_request_and_tampered_content_are_rejected(plan):
    result = export(plan)
    another = feedback.create_validation_plan(plan["scenario"], "device_smoke", plan["root"] / "another-plan")
    assert feedback.check_feedback(another["request_path"], result["feedback_dir"])["status"] == "REJECTED"
    path = Path(result["feedback_dir"]) / "feedback.json"
    path.write_bytes(path.read_bytes() + b" ")
    assert feedback.check_feedback(plan["request_path"], result["feedback_dir"])["errors"] == ["FEEDBACK_HASH_MISMATCH"]


def test_compatibility_source_changes_are_bound_before_and_after(plan):
    result = export(plan)
    (plan["source"] / "compatibility/matrix.yaml").write_text("schema_version: 2\n")
    with pytest.raises(feedback.FeedbackError, match="SOURCE_TREE_CHANGED"):
        feedback.execution_properties(plan["request_path"], plan["scenario"], phase="after")
    assert feedback.check_feedback(plan["request_path"], result["feedback_dir"])["errors"] == ["SOURCE_TREE_CHANGED"]


def test_changed_scenario_and_unknown_request_fields_are_rejected(plan):
    value = plan["value"]
    value["pod"] = "a-different-prepared-pod"
    plan["scenario"].write_text(json.dumps(value))
    with pytest.raises(feedback.FeedbackError, match="SCENARIO_CHANGED"):
        feedback.execution_properties(plan["request_path"], plan["scenario"])
    request_path = Path(plan["request_path"])
    request = json.loads(request_path.read_text())
    request["free_form"] = SECRET
    request_path.write_text(json.dumps(request))
    checked = feedback.check_feedback(request_path, plan["root"] / "absent")
    assert checked["errors"] == ["REQUEST_FIELDS_INVALID"]


def test_no_extra_private_files_or_symlinks_can_enter_the_share(plan):
    result = export(plan)
    root = Path(result["feedback_dir"])
    (root / "private.txt").write_text(SECRET)
    assert feedback.check_feedback(plan["request_path"], root)["status"] == "REJECTED"
    (root / "private.txt").unlink()
    original = (root / "feedback.json").read_bytes()
    (root / "feedback.json").unlink()
    other = plan["root"] / "external.json"
    other.write_bytes(original)
    (root / "feedback.json").symlink_to(other)
    assert feedback.check_feedback(plan["request_path"], root)["errors"] == ["SYMLINK_INPUT_REJECTED"]


def test_plan_and_export_outputs_are_fresh_and_unsupported_tiers_are_rejected(plan):
    with pytest.raises(feedback.FeedbackError, match="OUTPUT_ALREADY_EXISTS"):
        feedback.create_validation_plan(plan["scenario"], "device_smoke", plan["root"] / "plan")
    with pytest.raises(feedback.FeedbackError, match="TIER_UNSUPPORTED"):
        feedback.create_validation_plan(plan["scenario"], "arbitrary-command", plan["root"] / "bad-plan")
    export(plan)
    with pytest.raises(feedback.FeedbackError, match="OUTPUT_ALREADY_EXISTS"):
        export(plan)
