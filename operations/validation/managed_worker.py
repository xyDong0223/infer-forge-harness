"""Built-in worker measurement contract and deterministic grading.

This module grades raw measurements, never a producer's PASS report. Filesystem,
process, lease and receipt ownership belong to the validation runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from operations.validation.tensor_diff import grade


_CPU_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
                    "int32": 4, "int64": 8}
# A small logical tensor must not request unbounded storage through huge strides.
# This fixed resource ceiling is not a correctness tolerance or user override.
_MAX_STRIDED_INPUT_BYTES = 64 * 1024 * 1024


class ValidationContractError(ValueError):
    """A measurement cannot be justified by the frozen operator contract."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValidationContractError(message)


def _number(value, label, *, nonnegative=False):
    _require(type(value) in (int, float) and math.isfinite(value),
             f"{label} must be a finite number, not bool")
    _require(not nonnegative or value >= 0, f"{label} must be nonnegative")
    return value


def tensor_shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        _number(value, "tensor element")
        return []
    _require(bool(value), "simulation cannot infer empty tensor geometry; use the real CPU probe")
    shapes = [tensor_shape(item) for item in value]
    _require(all(shape == shapes[0] for shape in shapes), "tensor must be rectangular")
    return [len(value), *shapes[0]]


def python_dtype(value: Any) -> str:
    values = []

    def visit(item):
        if isinstance(item, list):
            for child in item:
                visit(child)
        else:
            _number(item, "tensor element")
            values.append(item)
    visit(value)
    _require(bool(values), "simulation cannot observe an empty tensor's dtype; use the real CPU probe")
    return "float64" if any(type(item) is float for item in values) else "int64"


def validate_tensor_values(value: Any, shape: list[int], label: str) -> None:
    """Check JSON values against explicit geometry, including unrepresented empty tails."""
    if not shape:
        _require(not isinstance(value, list), f"{label} must have the declared shape")
        _number(value, label)
        return
    _require(isinstance(value, list) and len(value) == shape[0],
             f"{label} must have the declared shape")
    for item in value:
        validate_tensor_values(item, shape[1:], label)


def validate_input_strides(value: dict, label: str) -> None:
    """Accept only explicit, provably non-overlapping CPU input strides.

    Sorted dimensions must occupy disjoint storage spans. This sufficient check
    covers transposes and ordinary gapped slices, not every possible strided view.
    Overlap, storage offsets and exotic layouts require a specialized probe.
    """
    _require("storage_offset" not in value, f"{label}.storage_offset needs a specialized probe")
    strides = value.get("strides")
    if strides is None:
        _require("strides" not in value and value["layout"] == "contiguous",
                 f"{label}.strides must be explicit for noncontiguous inputs")
        return
    shape = value["shape"]
    _require(isinstance(strides, list) and len(strides) == len(shape)
             and all(type(stride) is int and 0 <= stride <= 2**63 - 1 for stride in strides),
             f"{label}.strides must contain one nonnegative integer per dimension")
    if 0 in shape:
        _require(value["layout"] == "contiguous",
                 f"{label}: empty CPU tensors cannot prove a noncontiguous layout")
        return
    span = 1
    for stride, dim in sorted((stride, dim) for dim, stride in zip(shape, strides) if dim > 1):
        _require(stride >= span, f"{label}.strides are overlapping or need a specialized probe")
        span += (dim - 1) * stride
    _require(span * _CPU_DTYPE_BYTES[value["dtype"]] <= _MAX_STRIDED_INPUT_BYTES,
             f"{label}: strided input backing storage exceeds 64 MiB; use a specialized probe")
    contiguous, expected_stride = True, 1
    for dim, stride in reversed(list(zip(shape, strides))):
        if dim > 1 and stride != expected_stride:
            contiguous = False
        expected_stride *= dim
    _require(value["layout"] == ("contiguous" if contiguous else "noncontiguous"),
             f"{label}.strides disagree with the declared layout")


def _tensor_contract(value, label, *, input_tensor=False):
    _require(isinstance(value, dict), f"{label} must be an object")
    shape = value.get("shape")
    _require(isinstance(shape, list) and all(type(dim) is int and dim >= 0 for dim in shape),
             f"{label}.shape must contain explicit nonnegative dimensions")
    _require(value.get("dtype") in _CPU_DTYPE_BYTES,
             f"{label}.dtype is not supported by the builtin probe")
    _require(value.get("layout") in {"contiguous", "noncontiguous"},
             f"{label}.layout must explicitly be contiguous or noncontiguous")
    if input_tensor:
        validate_input_strides(value, label)
    else:
        _require("strides" not in value and "storage_offset" not in value,
                 f"{label}: explicit output strides/offsets need a specialized probe")


def validation_plan(spec, stage: str) -> dict:
    """Validate explicit cases/thresholds, without selecting convenient defaults."""
    spec = spec.to_dict() if hasattr(spec, "to_dict") else spec
    _require(stage in {"torch", "xpu", "integration"}, "unsupported validation stage")
    _require(isinstance(spec, dict), "OperatorSpec must be an object")
    contract = spec.get("semantics", {}).get("validation")
    _require(isinstance(contract, dict) and type(contract.get("schema_version")) is int
             and contract["schema_version"] == 1, "semantics.validation schema_version 1 is required")
    entry = contract.get("candidate_entry")
    _require(isinstance(entry, str) and entry and not Path(entry).is_absolute()
             and ".." not in Path(entry).parts and Path(entry).suffix == ".py",
             "candidate_entry must be a relative Python source path without traversal")
    reference = contract.get("reference")
    _require(isinstance(reference, dict), "independent reference is required")
    reference_entry = reference.get("entry")
    files = reference.get("files")
    _require(isinstance(reference_entry, str) and Path(reference_entry).is_absolute()
             and Path(reference_entry).suffix == ".py", "reference.entry must be absolute Python source")
    _require(isinstance(files, dict) and reference_entry in files and bool(files),
             "reference.files must bind its entry source")
    for name, sha in files.items():
        _require(isinstance(name, str) and Path(name).is_absolute()
                 and isinstance(sha, str) and len(sha) == 64
                 and all(char in "0123456789abcdef" for char in sha),
                 "reference.files requires absolute paths and SHA-256 hashes")
    _require(isinstance(reference.get("provenance"), str) and reference["provenance"].strip(),
             "reference.provenance must explain the independently written reference")
    controls = contract.get("negative_control")
    _require(isinstance(controls, dict) and controls.get("kind") in {"zeros", "offset"},
             "negative_control must explicitly select zeros or offset")
    if controls["kind"] == "offset":
        _require(_number(controls.get("value"), "negative_control.value") != 0,
                 "negative_control offset cannot be zero")
    input_specs = {item["name"]: item for item in spec.get("inputs", [])}
    output_specs = {item["name"]: item for item in spec.get("outputs", [])}
    _require(bool(input_specs) and bool(output_specs), "OperatorSpec inputs and outputs are required")
    cases = contract.get("cases")
    _require(isinstance(cases, list) and bool(cases), "explicit nonempty validation cases are required")
    ids = set()
    for case in cases:
        _require(isinstance(case, dict), "each case must be an object")
        identifier = case.get("id")
        _require(isinstance(identifier, str) and identifier.strip() and identifier not in ids,
                 "case IDs must be nonempty and unique")
        ids.add(identifier)
        bindings = case.get("bindings", {})
        _require(isinstance(bindings, dict) and all(type(value) is int and value >= 0
                                                  for value in bindings.values()),
                 "symbolic shape bindings must be explicit nonnegative integers")
        for group, specs in (("inputs", input_specs), ("outputs", output_specs)):
            values = case.get(group)
            _require(isinstance(values, dict) and set(values) == set(specs),
                     f"{identifier}.{group} must exactly cover OperatorSpec names")
            for name, value in values.items():
                label = f"{identifier}.{group}.{name}"
                _tensor_contract(value, label, input_tensor=group == "inputs")
                expected = specs[name]
                declared_shape = expected.get("shape")
                _require(isinstance(declared_shape, list), "OperatorSpec shape must be a dimension list")
                resolved_shape = [bindings.get(dim) if isinstance(dim, str) else dim
                                  for dim in declared_shape]
                _require(value["shape"] == resolved_shape
                         and all(type(dim) is int and dim >= 0 for dim in resolved_shape),
                         f"{label}.shape must match the observed OperatorSpec and explicit bindings")
                for field in ("dtype", "layout"):
                    _require(value[field] == expected[field], f"{label}.{field} differs from OperatorSpec")
                if group == "inputs":
                    _require("values" in value, f"{label}.values are required")
                    validate_tensor_values(value["values"], value["shape"], f"{label}.values")
    thresholds = contract.get("thresholds")
    _require(isinstance(thresholds, dict) and set(thresholds) == set(output_specs),
             "thresholds must explicitly cover every output")
    for name, threshold in thresholds.items():
        _require(isinstance(threshold, dict) and set(threshold) == {"max_relative_l2"},
                 f"thresholds.{name} must declare max_relative_l2 only")
        _number(threshold["max_relative_l2"], f"thresholds.{name}.max_relative_l2", nonnegative=True)
    if stage in {"xpu", "integration"}:
        dispatch = contract.get("expected_dispatch")
        _require(isinstance(dispatch, dict) and dispatch.get("symbol") == "run_case",
                 "builtin dispatch contract must name run_case")
        _require(isinstance(dispatch.get("device"), str) and bool(dispatch["device"]),
                 "expected_dispatch.device must be explicit")
        _require(dispatch.get("ranks") == [0], "builtin probe supports one explicitly declared rank [0]")
        fallback = contract.get("fallback")
        _require(isinstance(fallback, dict) and isinstance(fallback.get("allowed_devices"), list)
                 and bool(fallback["allowed_devices"])
                 and all(isinstance(device, str) and device for device in fallback["allowed_devices"]),
                 "fallback.allowed_devices must be explicitly declared")
    if stage == "integration":
        _require(isinstance(contract.get("service"), dict)
                 and type(contract["service"].get("require_http_status")) is int
                 and contract["service"]["require_http_status"] == 200,
                 "builtin service contract requires HTTP 200")
    # A JSON round-trip freezes caller-owned nested mappings and rejects NaN/Inf.
    return json.loads(json.dumps({"schema_version": 1, "stage": stage,
                                 "contract": contract, "contract_sha256": digest(contract)}, allow_nan=False))


def check_supported_mode(plan: dict, evidence_mode: str) -> None:
    _require(evidence_mode in {"real", "simulation"}, "unknown evidence mode")
    if evidence_mode == "real" and plan["stage"] in {"xpu", "integration"}:
        raise ValidationContractError(
            "builtin worker probe has no trusted real model-dispatch/service driver; "
            "real XPU/integration validation is blocked")
    if evidence_mode == "simulation":
        for case in plan["contract"]["cases"]:
            for group in ("inputs", "outputs"):
                for tensor in case[group].values():
                    _require(0 not in tensor["shape"] and "strides" not in tensor,
                             "empty/explicit-stride tensors require the real CPU probe")
                    _require(tensor["dtype"] in {"float64", "int64"}
                             and tensor["layout"] == "contiguous",
                             "simulation Python tensors support observed float64/int64 contiguous data only")
                    if group == "inputs":
                        _require(python_dtype(tensor["values"]) == tensor["dtype"],
                                 "simulation input dtype differs from actual Python values")
        if plan["stage"] in {"xpu", "integration"}:
            _require(plan["contract"]["expected_dispatch"]["device"] == "simulation-cpu",
                     "simulation dispatch must be labelled simulation-cpu")


def _measurements(plan, raw, role, evidence_mode, validation_id):
    _require(isinstance(raw, dict) and type(raw.get("schema_version")) is int and raw["schema_version"] == 1,
             f"{role} must contain raw measurement protocol 1")
    _require(raw.get("role") == role and raw.get("validation_id") == validation_id
             and raw.get("evidence_mode") == evidence_mode and raw.get("stage") == plan["stage"],
             f"{role} measurement identity mismatch")
    _require(raw.get("contract_sha256") == plan["contract_sha256"], f"{role} used a different contract")
    entries = raw.get("cases")
    _require(isinstance(entries, list) and len(entries) == len(plan["contract"]["cases"]),
             f"{role} must measure every required case exactly once")
    indexed = {}
    for entry in entries:
        _require(isinstance(entry, dict) and isinstance(entry.get("id"), str)
                 and entry["id"] not in indexed, f"{role} has an invalid or duplicate case")
        indexed[entry["id"]] = entry
    _require(set(indexed) == {case["id"] for case in plan["contract"]["cases"]},
             f"{role} measured the wrong case IDs")
    for case in plan["contract"]["cases"]:
        entry = indexed[case["id"]]
        _require(entry.get("inputs_sha256") == digest(case["inputs"]), f"{role} inputs differ from the frozen case")
        if evidence_mode == "real" and role != "control":
            # Protocol-1 ordinary cases remain replayable. New geometry requires
            # actual pre-call metadata, not merely a hash of the intended input.
            if any(0 in tensor["shape"] or "strides" in tensor for tensor in case["inputs"].values()):
                inputs = entry.get("input_metadata")
                _require(isinstance(inputs, dict) and set(inputs) == set(case["inputs"]),
                         f"{role} requires measured input metadata for empty/strided cases")
                for name, expected in case["inputs"].items():
                    observed = inputs[name]
                    _require(isinstance(observed, dict), f"{role}.{name} input metadata is missing")
                    for field in ("shape", "dtype", "layout"):
                        _require(observed.get(field) == expected[field], f"{role}.{name} input {field} mismatch")
                    _require(observed.get("device") == "cpu", f"{role}.{name} input must be CPU")
                    if "strides" in expected:
                        _require(observed.get("strides") == expected["strides"],
                                 f"{role}.{name} input strides mismatch")
        outputs = entry.get("outputs")
        _require(isinstance(outputs, dict) and set(outputs) == set(case["outputs"]),
                 f"{role} output names differ from the frozen case")
        for name, expected in case["outputs"].items():
            observed = outputs[name]
            _require(isinstance(observed, dict), f"{role}.{name} must be a raw tensor")
            validate_tensor_values(observed.get("values"), expected["shape"], f"{role}.{name} values")
            for field in ("shape", "dtype", "layout"):
                _require(observed.get(field) == expected[field], f"{role}.{name} {field} mismatch")
            device = observed.get("device")
            if role == "reference" or plan["stage"] == "torch":
                expected_device = "simulation-cpu" if evidence_mode == "simulation" else "cpu"
                _require(device == expected_device, f"{role}.{name} is not on the independent CPU path")
            _require(isinstance(device, str) and bool(device), f"{role}.{name} has no measured device")
    return indexed


def grade_observations(plan: dict, candidate: dict, reference: dict, control: dict,
                       *, evidence_mode: str, validation_id: str) -> dict:
    """Recompute gates from fixed probe observations; fail closed on omissions."""
    check_supported_mode(plan, evidence_mode)
    measured = {role: _measurements(plan, raw, role, evidence_mode, validation_id)
                for role, raw in (("candidate", candidate), ("reference", reference), ("control", control))}
    checks, cases = [], []
    nonempty_outputs = set()
    for case in plan["contract"]["cases"]:
        identifier = case["id"]
        metrics = {}
        for name in case["outputs"]:
            if 0 in case["outputs"][name]["shape"]:
                # There are no numbers with which to distinguish a wrong kernel.
                # Geometry was measured above; numerical evidence must come from
                # a separate nonempty case for this same output name.
                metrics[name] = {"numel": 0, "numerical": "not_applicable_empty",
                                 "shape": case["outputs"][name]["shape"]}
                checks.append({"name": f"{identifier}:{name}:empty_structure", "passed": True})
                continue
            nonempty_outputs.add(name)
            tensors = {role: data[identifier]["outputs"][name]["values"]
                       for role, data in measured.items()}
            metric = grade(tensors["candidate"], tensors["reference"], tensors["control"],
                           plan["contract"]["thresholds"][name]["max_relative_l2"])
            metrics[name] = metric
            checks.extend([
                {"name": f"{identifier}:{name}:numerical", "passed": metric.get("pass") is True},
                {"name": f"{identifier}:{name}:negative_control", "passed": metric.get("control_discriminates") is True},
            ])
        cases.append({"id": identifier, "metrics": metrics})
    for name in plan["contract"]["thresholds"]:
        checks.append({"name": f"{name}:nonempty_numerical_coverage", "passed": name in nonempty_outputs})
    if plan["stage"] in {"xpu", "integration"}:
        contract = plan["contract"]
        for case in contract["cases"]:
            observed = measured["candidate"][case["id"]]
            dispatch = observed.get("dispatch")
            _require(isinstance(dispatch, dict), "raw candidate dispatch observation is required")
            _require(dispatch.get("symbol") == contract["expected_dispatch"]["symbol"]
                     and type(dispatch.get("rank")) is int
                     and dispatch["rank"] in contract["expected_dispatch"]["ranks"]
                     and type(dispatch.get("calls")) is int and dispatch["calls"] > 0
                     and dispatch.get("entry_sha256") == candidate.get("entry", {}).get("sha256"),
                     "dispatch does not identify an executed candidate entry")
            _require(dispatch.get("device") == contract["expected_dispatch"]["device"],
                     "dispatch used the wrong device")
            observed_devices = {tensor["device"] for tensor in observed["outputs"].values()}
            _require(observed_devices == {dispatch["device"]}, "dispatch and tensor devices disagree")
            fallback = observed.get("fallback")
            _require(isinstance(fallback, dict) and fallback.get("observed_devices") == sorted(observed_devices),
                     "raw fallback observation is missing or incomplete")
            checks.extend([
                {"name": f"{case['id']}:dispatch", "passed": True},
                {"name": f"{case['id']}:fallback", "passed": observed_devices <= set(contract["fallback"]["allowed_devices"])},
            ])
            if plan["stage"] == "integration":
                service = observed.get("service")
                _require(isinstance(service, dict) and service.get("transport") == "http-loopback-simulation",
                         "raw service request/response observation is required")
                _require(service.get("request", {}).get("case_id") == case["id"]
                         and service["request"].get("inputs") == {
                             name: tensor["values"] for name, tensor in case["inputs"].items()},
                         "service request did not exercise the frozen case")
                _require(service.get("response", {}).get("outputs") == observed["outputs"],
                         "service response differs from graded output tensors")
                checks.append({"name": f"{case['id']}:service", "passed":
                               service.get("http_status") == contract["service"]["require_http_status"]})
    return {"verdict": "PASS" if all(check["passed"] for check in checks) else "FAIL",
            "checks": checks, "cases": cases,
            "reference_provenance": plan["contract"]["reference"]["provenance"],
            "evidence_mode": evidence_mode, "contract_sha256": plan["contract_sha256"]}
