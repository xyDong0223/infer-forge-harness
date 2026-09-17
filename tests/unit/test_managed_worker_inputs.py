"""Bounded real CPU geometry; no accelerator or torch dependency for contract tests."""

from copy import deepcopy
import hashlib
import json

import pytest

from operations.validation.managed_worker import (
    ValidationContractError, check_supported_mode, digest, grade_observations,
    python_dtype, validation_plan,
)
from operations.validation.worker_probe import _inputs, measure


def make_spec(tmp_path, *, cases=None, input_layout="contiguous", shape=None):
    geometry = {"shape": shape if shape is not None else ["N", 3],
                "dtype": "float64", "layout": input_layout}
    output = {**geometry, "layout": "contiguous"}
    if cases is None:
        cases = [
            {"id": "empty", "bindings": {"N": 0}, "inputs": {"x": {
                **geometry, "shape": [0, 3], "values": []}},
             "outputs": {"y": {**output, "shape": [0, 3]}}},
            {"id": "nonempty", "bindings": {"N": 2}, "inputs": {"x": {
                **geometry, "shape": [2, 3], "values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}},
             "outputs": {"y": {**output, "shape": [2, 3]}}},
        ]
    reference = str(tmp_path / "reference.py")
    return {"inputs": [{"name": "x", **geometry}], "outputs": [{"name": "y", **output}],
            "semantics": {"validation": {
                "schema_version": 1, "candidate_entry": "candidate.py",
                "reference": {"entry": reference, "files": {reference: "a" * 64},
                              "provenance": "independent addition oracle"},
                "cases": cases, "thresholds": {"y": {"max_relative_l2": 1e-8}},
                "negative_control": {"kind": "offset", "value": 1.0},
                "expected_dispatch": {"symbol": "run_case", "device": "cpu", "ranks": [0]},
                "fallback": {"allowed_devices": ["cpu"]},
                "service": {"require_http_status": 200},
            }}}


def noncontiguous_spec(tmp_path, strides=(1, 2)):
    spec = make_spec(tmp_path, input_layout="noncontiguous")
    contract = spec["semantics"]["validation"]
    contract["cases"] = contract["cases"][1:]
    contract["cases"][0]["inputs"]["x"]["strides"] = list(strides)
    return spec


def map_values(value, function):
    return [map_values(item, function) for item in value] if isinstance(value, list) else function(value)


def raw_observations(plan):
    """Synthetic protocol fixtures test the pure grader, not execution evidence."""
    result = {}
    for role in ("candidate", "reference", "control"):
        entries = []
        for case in plan["contract"]["cases"]:
            source = case["inputs"]["x"]
            values = map_values(source["values"], lambda value: 2 * value + (role == "control"))
            entries.append({
                "id": case["id"], "inputs_sha256": digest(case["inputs"]),
                "input_metadata": {"x": {key: deepcopy(value) for key, value in source.items()
                                         if key != "values"} | {"device": "cpu"}},
                "outputs": {"y": {**case["outputs"]["y"], "values": values, "device": "cpu"}},
            })
        result[role] = {"schema_version": 1, "validation_id": "unit", "role": role,
                        "stage": "torch", "evidence_mode": "real",
                        "contract_sha256": plan["contract_sha256"], "cases": entries}
    return result


def grade_raw(plan, raw):
    return grade_observations(plan, **raw, evidence_mode="real", validation_id="unit")


@pytest.mark.parametrize("shape,values", [([0], []), ([0, 3], []), ([2, 0, 4], [[], []])])
def test_explicit_empty_geometry_is_accepted_without_inferring_trailing_dimensions(tmp_path, shape, values):
    spec = make_spec(tmp_path, shape=shape)
    case = spec["semantics"]["validation"]["cases"][0]
    case["inputs"]["x"].update(shape=shape, values=values)
    case["outputs"]["y"]["shape"] = shape
    spec["semantics"]["validation"]["cases"] = [case]
    plan = validation_plan(spec, "torch")
    check_supported_mode(plan, "real")
    assert plan["contract"]["cases"][0]["inputs"]["x"]["shape"] == shape
    with pytest.raises(ValidationContractError, match="real CPU probe"):
        check_supported_mode(plan, "simulation")


@pytest.mark.parametrize("shape,values", [([2, 0, 4], []), ([0, 3], [[]]), ([2, 3], [[], []])])
def test_empty_values_do_not_hide_incompatible_geometry(tmp_path, shape, values):
    spec = make_spec(tmp_path, shape=shape)
    case = spec["semantics"]["validation"]["cases"][0]
    case["inputs"]["x"].update(shape=shape, values=values)
    case["outputs"]["y"]["shape"] = shape
    spec["semantics"]["validation"]["cases"] = [case]
    with pytest.raises(ValidationContractError, match="declared shape"):
        validation_plan(spec, "torch")


def test_simulation_never_labels_empty_values_as_int64():
    with pytest.raises(ValidationContractError, match="empty tensor's dtype"):
        python_dtype([[], []])


@pytest.mark.parametrize("strides", [[1, 2], [6, 2], [1, 4]])
def test_explicit_nonoverlapping_strides_are_frozen(tmp_path, strides):
    spec = noncontiguous_spec(tmp_path, strides)
    plan = validation_plan(spec, "torch")
    assert plan["contract"]["cases"][0]["inputs"]["x"]["strides"] == strides
    spec["semantics"]["validation"]["cases"][0]["inputs"]["x"]["strides"][0] += 10
    assert plan["contract_sha256"] == digest(plan["contract"])
    assert plan["contract_sha256"] != digest(spec["semantics"]["validation"])
    with pytest.raises(ValidationContractError, match="real CPU probe"):
        check_supported_mode(plan, "simulation")


@pytest.mark.parametrize("strides", [None, [], [1], [True, 2], [-1, 2], [0, 1], [1, 1], [3, 1]])
def test_ambiguous_overlapping_or_mislabeled_strides_are_rejected(tmp_path, strides):
    spec = noncontiguous_spec(tmp_path)
    spec["semantics"]["validation"]["cases"][0]["inputs"]["x"]["strides"] = strides
    with pytest.raises(ValidationContractError, match="strides"):
        validation_plan(spec, "torch")


def test_missing_strides_and_unimplemented_offset_are_not_guessed(tmp_path):
    spec = noncontiguous_spec(tmp_path)
    tensor = spec["semantics"]["validation"]["cases"][0]["inputs"]["x"]
    strides = tensor.pop("strides")
    with pytest.raises(ValidationContractError, match="explicit"):
        validation_plan(spec, "torch")
    tensor.update(strides=strides, storage_offset=1)
    with pytest.raises(ValidationContractError, match="storage_offset"):
        validation_plan(spec, "torch")


@pytest.mark.parametrize("dtype,itemsize", [("float16", 2), ("bfloat16", 2), ("float32", 4),
                                           ("float64", 8), ("int32", 4), ("int64", 8)])
def test_strided_storage_has_a_fixed_byte_limit_before_materialization(tmp_path, dtype, itemsize):
    spec = noncontiguous_spec(tmp_path)
    spec["inputs"][0]["dtype"] = dtype
    tensor = spec["semantics"]["validation"]["cases"][0]["inputs"]["x"]
    tensor.update(dtype=dtype, strides=[64 * 1024 * 1024 // itemsize, 1])
    with pytest.raises(ValidationContractError, match="backing storage exceeds 64 MiB.*specialized probe"):
        validation_plan(spec, "torch")


def test_empty_edge_case_requires_nonempty_numerical_evidence(tmp_path):
    plan = validation_plan(make_spec(tmp_path), "torch")
    report = grade_raw(plan, raw_observations(plan))
    assert report["verdict"] == "PASS"
    assert report["cases"][0]["metrics"]["y"] == {
        "numel": 0, "numerical": "not_applicable_empty", "shape": [0, 3]}
    spec = make_spec(tmp_path)
    spec["semantics"]["validation"]["cases"].pop()
    empty_plan = validation_plan(spec, "torch")
    rejected = grade_raw(empty_plan, raw_observations(empty_plan))
    assert rejected["verdict"] == "FAIL"
    assert {"name": "y:nonempty_numerical_coverage", "passed": False} in rejected["checks"]


@pytest.mark.parametrize("change", ["shape", "metadata", "strides", "device"])
def test_new_input_geometry_requires_pre_call_metadata(tmp_path, change):
    plan = validation_plan(noncontiguous_spec(tmp_path), "torch")
    raw = raw_observations(plan)
    entry = raw["candidate"]["cases"][0]
    if change == "metadata":
        entry.pop("input_metadata")
    else:
        entry["input_metadata"]["x"][change] = {"shape": [3, 2], "strides": [3, 1], "device": "xpu"}[change]
    with pytest.raises(ValidationContractError, match="input"):
        grade_raw(plan, raw)


def test_empty_metadata_and_nonempty_negative_controls_still_fail_closed(tmp_path):
    plan = validation_plan(make_spec(tmp_path), "torch")
    raw = raw_observations(plan)
    raw["candidate"]["cases"][0]["outputs"]["y"]["shape"] = [0, 4]
    with pytest.raises(ValidationContractError, match="shape mismatch"):
        grade_raw(plan, raw)
    raw = raw_observations(plan)
    raw["control"]["cases"] = deepcopy(raw["reference"]["cases"])
    assert grade_raw(plan, raw)["verdict"] == "FAIL"
    raw = raw_observations(plan)
    raw["candidate"]["cases"][1]["outputs"]["y"]["values"][0][0] = 1000.0
    assert grade_raw(plan, raw)["verdict"] == "FAIL"


@pytest.mark.parametrize("stage", ["xpu", "integration"])
def test_new_cpu_geometries_do_not_open_real_device_or_service_gate(tmp_path, stage):
    plan = validation_plan(noncontiguous_spec(tmp_path), stage)
    with pytest.raises(ValidationContractError, match="real XPU/integration validation is blocked"):
        check_supported_mode(plan, "real")


def actual_measurements(tmp_path, spec, candidate_source):
    pytest.importorskip("torch")
    candidate = tmp_path / "candidate.py"
    candidate.write_text(candidate_source)
    reference = tmp_path / "reference.py"
    reference.write_text("def run_case(inputs):\n    return {'y': (inputs['x'] + inputs['x']).contiguous()}\n")
    spec["semantics"]["validation"]["reference"]["files"][str(reference)] = hashlib.sha256(reference.read_bytes()).hexdigest()
    plan = validation_plan(spec, "torch")
    base = {"validation_id": "unit", "evidence_mode": "real", "plan": plan}
    raw = {}
    for role, entry in (("candidate", candidate), ("reference", reference)):
        raw[role] = measure({**base, "role": role, "entry": {
            "path": str(entry), "sha256": hashlib.sha256(entry.read_bytes()).hexdigest()}})
    observations = tmp_path / "reference-observations.json"
    observations.write_text(json.dumps(raw["reference"]))
    raw["control"] = measure({**base, "role": "control", "reference_observations": str(observations)})
    return plan, raw


@pytest.mark.parametrize("middle_empty", [False, True])
def test_real_cpu_empty_tails_and_numeric_case_are_measured(tmp_path, middle_empty):
    spec = make_spec(tmp_path)
    empty_shape = [0, 3]
    if middle_empty:
        spec["inputs"][0]["shape"] = spec["outputs"][0]["shape"] = [2, "N", 4]
        empty, nonempty = spec["semantics"]["validation"]["cases"]
        empty_shape = [2, 0, 4]
        empty["inputs"]["x"].update(shape=empty_shape, values=[[], []])
        empty["outputs"]["y"]["shape"] = empty_shape
        nonempty["bindings"]["N"] = 1
        nonempty["inputs"]["x"].update(shape=[2, 1, 4], values=[[[1.0, 2.0, 3.0, 4.0]],
                                                             [[5.0, 6.0, 7.0, 8.0]]])
        nonempty["outputs"]["y"]["shape"] = [2, 1, 4]
    plan, raw = actual_measurements(tmp_path, spec,
        "def run_case(inputs):\n    return {'y': (inputs['x'] * 2).contiguous()}\n")
    assert grade_raw(plan, raw)["verdict"] == "PASS"
    for role in raw:
        assert raw[role]["cases"][0]["outputs"]["y"]["shape"] == empty_shape
        assert raw[role]["cases"][0]["outputs"]["y"]["values"] == ([[], []] if middle_empty else [])


@pytest.mark.parametrize("strides", [[1, 2], [6, 2]])
def test_real_cpu_strides_are_observed_before_mutation_and_roles_have_fresh_inputs(tmp_path, strides):
    spec = noncontiguous_spec(tmp_path, strides)
    before = deepcopy(spec["semantics"]["validation"]["cases"])
    plan, raw = actual_measurements(tmp_path, spec,
        "def run_case(inputs):\n"
        "    x = inputs['x']\n"
        "    assert not x.is_contiguous()\n"
        "    result = (x * 2).contiguous()\n"
        "    x.fill_(1000)\n"
        "    x.transpose_(0, 1)\n"
        "    return {'y': result}\n")
    assert grade_raw(plan, raw)["verdict"] == "PASS"
    for role in ("candidate", "reference"):
        observed = raw[role]["cases"][0]["input_metadata"]["x"]
        assert observed["strides"] == strides
        assert observed["shape"] == [2, 3]
        assert observed["layout"] == "noncontiguous"
    assert spec["semantics"]["validation"]["cases"] == before


def test_python_simulation_inputs_are_also_fresh_per_invocation():
    case = {"inputs": {"x": {"values": [1.0, 2.0]}}}
    first = _inputs(case, "simulation")
    first["x"][0] = 99.0
    assert _inputs(case, "simulation")["x"] == [1.0, 2.0]
    assert case["inputs"]["x"]["values"] == [1.0, 2.0]
