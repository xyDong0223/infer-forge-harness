"""Minimal offline handoff for opt-in hardware tests, never a promotion channel.

Only an explicitly supplied scenario and JUnit file are read. The authoritative
scheduler stays on the execution host. Matching hashes/properties establish
consistency, not authenticity of an externally supplied observation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4
import xml.etree.ElementTree as ET

from core.paths import REPO_ROOT
from core.storage import ArtifactStore, ensure_external


_HARDWARE_TEST = "tests/e2e/test_model_adaptation_hardware.py"
_TIERS = {
    "device_smoke": ("test_prepared_device_smoke", "INFER_FORGE_RUN_DEVICE_SMOKE"),
    "real_model": ("test_real_model_service_regression", "INFER_FORGE_RUN_REAL_MODEL"),
}
_SOURCE_ROOTS = (
    "adapters", "catalog", "cli", "compatibility", "config", "contracts", "core", "engine", "operations",
    "runners", "runtimes", "tasks", "tests", "tools", "validators", "workflows",
)
_SOURCE_IGNORED = {"__pycache__", ".pytest_cache", ".venv", ".git", ".DS_Store"}
_SOURCE_SCOPE = "repository-execution-files-v1"
_LIMITATIONS = ("external_observation_is_not_authenticated", "hashes_do_not_prove_hardware_correctness",
                "only_the_original_scheduler_can_accept_evidence")
_MAX_JSON = 2 * 1024 * 1024
_MAX_JUNIT = 8 * 1024 * 1024
_HEX = re.compile(r"[a-f0-9]{64}\Z")


class FeedbackError(ValueError):
    """A fixed, share-safe reason code; never an input path or raw error text."""


def _require(condition, reason):
    if not condition:
        raise FeedbackError(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _keys(value, keys, reason):
    _require(isinstance(value, dict) and set(value) == set(keys), reason)


def _read(path, limit):
    path = Path(path).expanduser()
    _require(not path.is_symlink(), "SYMLINK_INPUT_REJECTED")
    path = ensure_external(path)
    _require(path.is_file() and path.stat().st_size <= limit, "INPUT_NOT_REGULAR_OR_TOO_LARGE")
    content = path.read_bytes()
    _require(len(content) <= limit, "INPUT_TOO_LARGE")
    return content


def _json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    try:
        return json.loads(content, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(FeedbackError("NONFINITE_JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeedbackError("INVALID_JSON") from error


def _scenario(path):
    value = _json(_read(path, _MAX_JSON))
    fields = {"run_id", "state", "artifact_root", "subject", "pod", "namespace", "image_digest",
              "hardware", "model_revision", "plugin_revision", "cleanup_policy", "environment", "context"}
    _require(isinstance(value, dict) and fields <= value.keys(), "SCENARIO_FIELDS_MISSING")
    for key in fields - {"environment", "context"}:
        _require(isinstance(value[key], str) and value[key].strip(), "SCENARIO_FIELD_INVALID")
    _require(value["cleanup_policy"] == "retain_prepared_pod", "PREPARED_POD_MUST_BE_RETAINED")
    _require(value["model_revision"] != "unknown" and value["plugin_revision"] != "unknown",
             "PINNED_REVISIONS_REQUIRED")
    _require(value["image_digest"].startswith("sha256:") and _sha(value["image_digest"][7:]),
             "PINNED_IMAGE_DIGEST_REQUIRED")
    _require(isinstance(value["environment"], dict) and isinstance(value["context"], dict),
             "SCENARIO_CONTEXT_INVALID")
    owner = value["context"].get("user_id")
    _require(isinstance(owner, str) and owner.strip(), "USER_SUPPLIED_OWNER_REQUIRED")
    for key in ("state", "artifact_root"):
        _require(Path(value[key]).is_absolute(), "SCENARIO_PATH_MUST_BE_ABSOLUTE")
        ensure_external(value[key])  # Validate the path; do not open the DB or inspect its contents.
    return value


def _source_tree():
    """Hash actual execution source bytes, including uncommitted source changes."""
    root = REPO_ROOT.resolve()
    files = []
    for name in _SOURCE_ROOTS:
        directory = root / name
        if not directory.exists():
            continue
        _require(directory.is_dir() and not directory.is_symlink(), "SOURCE_TREE_INVALID")
        for base, directories, names in os.walk(directory, followlinks=False):
            base = Path(base)
            directories[:] = sorted(item for item in directories if item not in _SOURCE_IGNORED)
            _require(all(not (base / item).is_symlink() for item in directories), "SOURCE_SYMLINK_REJECTED")
            files.extend(base / item for item in names if item not in _SOURCE_IGNORED)
    files.extend(root / name for name in ("pyproject.toml", "AGENTS.md") if (root / name).exists())
    records = []
    for path in sorted(files):
        _require(path.is_file() and not path.is_symlink(), "SOURCE_NOT_REGULAR")
        sha = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
                size += len(block)
        records.append({"path": path.relative_to(root).as_posix(), "size_bytes": size,
                        "sha256": sha.hexdigest()})
    _require(records and (root / _HARDWARE_TEST).is_file(), "HARDWARE_TEST_SOURCE_MISSING")
    return {"scope": _SOURCE_SCOPE, "sha256": _digest(records), "file_count": len(records)}, records


def _request(path, *, current=True):
    value = _json(_read(path, _MAX_JSON))
    _keys(value, {"schema_version", "kind", "request_id", "tier", "case_id", "source",
                  "scenario_sha256", "authority", "cleanup_policy"}, "REQUEST_FIELDS_INVALID")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
             and value["kind"] == "ValidationRequest", "REQUEST_PROTOCOL_INVALID")
    _require(isinstance(value["request_id"], str) and re.fullmatch(r"[a-f0-9]{32}", value["request_id"]),
             "REQUEST_ID_INVALID")
    _require(isinstance(value["tier"], str) and value["tier"] in _TIERS, "TIER_UNSUPPORTED")
    _require(value["case_id"] == f"{_HARDWARE_TEST}::{_TIERS[value['tier']][0]}", "CASE_NOT_ALLOWED")
    _keys(value["source"], {"scope", "sha256", "file_count"}, "SOURCE_BINDING_INVALID")
    _require(value["source"]["scope"] == _SOURCE_SCOPE and _sha(value["source"]["sha256"])
             and type(value["source"]["file_count"]) is int and value["source"]["file_count"] > 0,
             "SOURCE_BINDING_INVALID")
    _require(_sha(value["scenario_sha256"]), "SCENARIO_BINDING_INVALID")
    _require(value["authority"] == "original_inner_scheduler_only"
             and value["cleanup_policy"] == "retain_prepared_pod", "AUTHORITY_POLICY_INVALID")
    if current:
        _require(value["source"] == _source_tree()[0], "SOURCE_TREE_CHANGED")
    return value


def _fresh(out):
    lexical = Path(out).expanduser()
    _require(not lexical.is_symlink(), "SYMLINK_OUTPUT_REJECTED")
    root = ensure_external(lexical)
    _require(not root.exists(), "OUTPUT_ALREADY_EXISTS")
    root.mkdir(parents=True, mode=0o700)
    return root


def create_validation_plan(scenario_path, tier, out):
    """Prepare fixed instructions only; no DB access, authorization or execution."""
    _require(isinstance(tier, str) and tier in _TIERS, "TIER_UNSUPPORTED")
    scenario = _scenario(scenario_path)
    source, files = _source_tree()
    request = {
        "schema_version": 1, "kind": "ValidationRequest", "request_id": uuid4().hex,
        "tier": tier, "case_id": f"{_HARDWARE_TEST}::{_TIERS[tier][0]}", "source": source,
        "scenario_sha256": _digest(scenario), "authority": "original_inner_scheduler_only",
        "cleanup_policy": "retain_prepared_pod",
    }
    root = _fresh(out)
    store = ArtifactStore(root)
    request_path = store.write_json("public/request.json", request)
    scenario_copy = store.write_json("private/scenario.json", scenario)
    source_manifest = store.write_json("private/source-manifest.json", files)
    plan = {
        "schema_version": 1, "kind": "PrivateValidationPlan", "request_sha256": _digest(request),
        "argv": ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider", request["case_id"],
                 "--junitxml", str(root / "private/junit.xml"),
                 "--basetemp", str(root / "private/pytest")],
        "cwd": str(REPO_ROOT),
        "environment": {"INFER_FORGE_HARDWARE_SCENARIO": str(scenario_copy),
                        "INFER_FORGE_VALIDATION_REQUEST": str(request_path), "PYTHONDONTWRITEBYTECODE": "1"},
        "remove_environment": ["PYTHONPATH", "INFER_FORGE_E2E_*"],
        "authorization_required": {"variable": _TIERS[tier][1], "value": "1",
                                   "must_be_explicitly_authorized": True},
        "execution_authorized": False,
        "limitations": ["planning does not validate the existing run or authorize hardware execution",
                        "managed-v2 real execution remains blocked without trusted drivers",
                        "share public/request.json only, never the private directory"],
    }
    private_plan = store.write_json("private/execution-plan.json", plan)
    (root / "private").chmod(0o700)
    for path in (scenario_copy, source_manifest, private_plan):
        path.chmod(0o600)
    return {"status": "PLANNED", "request": request, "request_path": str(request_path),
            "private_plan_path": str(private_plan), "execution_authorized": False}


def execution_properties(request_path, scenario_path, *, phase="before"):
    """Bind actual hardware-test properties before/after; never query the DB here."""
    _require(phase in {"before", "after"}, "EXECUTION_PHASE_INVALID")
    request = _request(request_path)
    _require(_digest(_scenario(scenario_path)) == request["scenario_sha256"], "SCENARIO_CHANGED")
    suffix = "_after" if phase == "after" else ""
    values = {"request_sha256": _digest(request), "scenario_sha256": request["scenario_sha256"],
              "source_tree_sha256": request["source"]["sha256"]}
    return [(f"infer_forge.{key}{suffix}", value) for key, value in values.items()]


def _classification(counts, exit_code, bound):
    if counts["skipped"]:
        state, reason = "NOT_RUN", "SKIPPED"
    elif not bound:
        state, reason = "BLOCKED", "EXECUTION_BINDING_MISSING"
    elif counts["failed"] or counts["errors"]:
        state, reason = "TEST_FAILED", "TEST_ERROR" if counts["errors"] else "TEST_FAILURE"
    else:
        state, reason = "TEST_PASSED", "REQUESTED_CASE_PASSED"
    if ((counts["passed"] and exit_code != 0)
            or ((counts["failed"] or counts["errors"]) and exit_code == 0)):
        state, reason = "BLOCKED", "EXIT_STATUS_MISMATCH"
    return state, reason


def _observe(request, content, exit_code):
    _require(type(exit_code) is int and -255 <= exit_code <= 255, "EXIT_CODE_INVALID")
    try:
        text = content.decode("utf-8-sig")
        _require("<!DOCTYPE" not in text.upper() and "<!ENTITY" not in text.upper(), "UNSAFE_JUNIT_XML")
        tree = ET.fromstring(text)
    except (ET.ParseError, UnicodeDecodeError) as error:
        raise FeedbackError("INVALID_JUNIT_XML") from error
    _require(tree.tag in {"testsuite", "testsuites"}, "JUNIT_ROOT_INVALID")
    cases = list(tree.iter("testcase"))
    _require(len(cases) == 1, "JUNIT_MUST_CONTAIN_EXACTLY_REQUESTED_CASE")
    case = cases[0]
    _require(case.get("name") == _TIERS[request["tier"]][0]
             and case.get("classname") == _HARDWARE_TEST.removesuffix(".py").replace("/", "."),
             "JUNIT_CASE_MISMATCH")
    flags = {key: len(case.findall(tag)) for key, tag in
             (("failed", "failure"), ("errors", "error"), ("skipped", "skipped"))}
    _require(sum(flags.values()) <= 1, "JUNIT_OUTCOME_AMBIGUOUS")
    _require(all(item.tag in {"properties", "system-out", "system-err", "failure", "error", "skipped"}
                 for item in case), "JUNIT_CASE_STRUCTURE_INVALID")
    suites = list(tree.iter("testsuite"))
    _require(len(suites) == 1, "JUNIT_MUST_CONTAIN_EXACTLY_REQUESTED_CASE")
    expected_counts = {"tests": 1, "failures": flags["failed"], "errors": flags["errors"], "skipped": flags["skipped"]}
    for label, expected_count in expected_counts.items():
        recorded = suites[0].get(label)
        _require(recorded is not None and re.fullmatch(r"[0-9]+", recorded)
                 and int(recorded) == expected_count, "JUNIT_COUNTS_INCONSISTENT")
    # Ignore all raw messages, stdout, traceback, paths and non-binding properties.
    expected = {"request_sha256": _digest(request), "scenario_sha256": request["scenario_sha256"],
                "source_tree_sha256": request["source"]["sha256"]}
    expected = {f"infer_forge.{key}{suffix}": value for key, value in expected.items()
                for suffix in ("", "_after")}
    observed = {}
    for prop in case.findall("./properties/property"):
        name = prop.get("name")
        if name in expected:
            _require(name not in observed, "JUNIT_BINDING_DUPLICATED")
            observed[name] = prop.get("value")
    _require(all(value == expected[key] for key, value in observed.items()), "JUNIT_BINDING_MISMATCH")
    bound = observed == expected
    counts = {"tests": 1, "executed": 0 if flags["skipped"] else 1,
              "passed": 0 if any(flags.values()) else 1, **flags}
    state, reason = _classification(counts, exit_code, bound)
    return {"state": state, "reason_code": reason, "test_counts": counts,
            "exit_code": exit_code, "request_binding_matches": bound}


def export_feedback(request_path, junit_path, exit_code, out):
    """Export a fixed safe summary, never raw JUnit or caller-selected artifacts."""
    request = _request(request_path)
    content = _read(junit_path, _MAX_JUNIT)
    observation = _observe(request, content, exit_code)
    summary = {
        "schema_version": 1, "kind": "ValidationFeedback", "request_id": request["request_id"],
        "request_sha256": _digest(request), "tier": request["tier"], "case_id": request["case_id"],
        "source": request["source"], "scenario_sha256": request["scenario_sha256"],
        "observation": observation,
        "evidence": {"junit": {"sha256": hashlib.sha256(content).hexdigest(),
                               "size_bytes": len(content), "content_included": False}},
        "observations_verified": False, "promoted": False, "limitations": list(_LIMITATIONS),
    }
    root = _fresh(out)
    feedback = ArtifactStore(root).write_json("feedback.json", summary)
    data = feedback.read_bytes()
    manifest = {"schema_version": 1, "kind": "ValidationFeedbackManifest",
                "request_sha256": _digest(request), "files": [{"name": "feedback.json",
                "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}]}
    ArtifactStore(root).write_json("manifest.json", manifest)
    return {"status": "FEEDBACK_EXPORTED", "feedback_dir": str(root), "summary": summary,
            "observations_verified": False, "promoted": False}


def _check_summary(summary, request):
    _keys(summary, {"schema_version", "kind", "request_id", "request_sha256", "tier", "case_id",
                    "source", "scenario_sha256", "observation", "evidence", "observations_verified",
                    "promoted", "limitations"}, "FEEDBACK_FIELDS_INVALID")
    _require(type(summary["schema_version"]) is int and summary["schema_version"] == 1
             and summary["kind"] == "ValidationFeedback", "FEEDBACK_PROTOCOL_INVALID")
    for key in ("request_id", "tier", "case_id", "source", "scenario_sha256"):
        _require(summary[key] == request[key], "FEEDBACK_REQUEST_MISMATCH")
    _require(summary["request_sha256"] == _digest(request), "FEEDBACK_REQUEST_MISMATCH")
    _require(summary["observations_verified"] is False and summary["promoted"] is False
             and summary["limitations"] == list(_LIMITATIONS), "FEEDBACK_CANNOT_PROMOTE")
    observation = summary["observation"]
    _keys(observation, {"state", "reason_code", "test_counts", "exit_code", "request_binding_matches"},
          "OBSERVATION_FIELDS_INVALID")
    counts = observation["test_counts"]
    _keys(counts, {"tests", "executed", "passed", "failed", "errors", "skipped"}, "COUNTS_INVALID")
    _require(all(type(value) is int and value in (0, 1) for value in counts.values())
             and counts["tests"] == 1 and sum(counts[key] for key in ("passed", "failed", "errors", "skipped")) == 1
             and counts["executed"] == 1 - counts["skipped"], "COUNTS_INVALID")
    _require(type(observation["exit_code"]) is int and -255 <= observation["exit_code"] <= 255
             and type(observation["request_binding_matches"]) is bool, "OBSERVATION_FIELDS_INVALID")
    # Recompute the same finite state mapping, without trusting a claimed PASS.
    state, reason = _classification(counts, observation["exit_code"], observation["request_binding_matches"])
    _require(observation["state"] == state and observation["reason_code"] == reason, "OBSERVATION_STATE_INVALID")
    _keys(summary["evidence"], {"junit"}, "EVIDENCE_FIELDS_INVALID")
    evidence = summary["evidence"]["junit"]
    _keys(evidence, {"sha256", "size_bytes", "content_included"}, "EVIDENCE_FIELDS_INVALID")
    _require(_sha(evidence["sha256"]) and type(evidence["size_bytes"]) is int
             and 0 < evidence["size_bytes"] <= _MAX_JUNIT and evidence["content_included"] is False,
             "RAW_EVIDENCE_CANNOT_BE_INCLUDED")


def check_feedback(request_path, feedback_dir):
    """Read-only structural/integrity check; explicitly not evidence acceptance."""
    try:
        request = _request(request_path)
        root = Path(feedback_dir).expanduser()
        _require(not root.is_symlink(), "SYMLINK_INPUT_REJECTED")
        root = ensure_external(root)
        _require(root.is_dir() and {path.name for path in root.iterdir()} == {"feedback.json", "manifest.json"},
                 "FEEDBACK_DIRECTORY_MUST_CONTAIN_ONLY_PUBLIC_FILES")
        feedback_bytes = _read(root / "feedback.json", _MAX_JSON)
        manifest = _json(_read(root / "manifest.json", _MAX_JSON))
        _keys(manifest, {"schema_version", "kind", "request_sha256", "files"}, "MANIFEST_FIELDS_INVALID")
        _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
                 and manifest["kind"] == "ValidationFeedbackManifest"
                 and manifest["request_sha256"] == _digest(request), "MANIFEST_BINDING_INVALID")
        expected_files = [{"name": "feedback.json", "sha256": hashlib.sha256(feedback_bytes).hexdigest(),
                           "size_bytes": len(feedback_bytes)}]
        _require(manifest["files"] == expected_files, "FEEDBACK_HASH_MISMATCH")
        summary = _json(feedback_bytes)
        _check_summary(summary, request)
        return {"status": "FEEDBACK_VALID", "request_id": request["request_id"],
                "observation": summary["observation"], "observations_verified": False, "promoted": False,
                "errors": [], "limitations": list(_LIMITATIONS)}
    except FeedbackError as error:
        reason = str(error)
    except (OSError, ValueError, TypeError, KeyError):
        reason = "FEEDBACK_UNREADABLE_OR_INVALID"
    return {"status": "REJECTED", "errors": [reason], "observations_verified": False, "promoted": False}
