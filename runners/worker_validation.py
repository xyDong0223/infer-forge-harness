"""Run fixed independent worker probes and bind their measured result to a receipt."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

from core.paths import REPO_ROOT
from core.storage import ArtifactStore, ensure_external, safe_component
from engine.execution import has_active_task_execution
from engine.managed_validation import (
    RECIPE_FILES, RECIPE_PATH, begin_validation, finish_validation, get_candidate,
)
from engine.result_validation import STAGE_EVIDENCE
from operations.validation.managed_worker import (
    ValidationContractError, check_supported_mode, digest, grade_observations, validation_plan,
)
from runners.managed_execution import execute_managed, host_identity, process_identity


def _file(path):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or path != path.resolve()
            or not path.is_file()):
        raise ValidationContractError(f"validation requires a canonical regular file: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _check_files(files):
    for name, expected in files.items():
        if _file(name)["sha256"] != expected:
            raise ValidationContractError(f"validation input changed: {name}")


def _identity(task, run):
    return {
        "task_id": task.task_id, "operator_key": task.operator_key, "stage": task.stage,
        "attempt": task.attempt, "environment_fingerprint":
        run.environment.get("environment_proof", {}).get("fingerprint"),
        "evidence_mode": run.metadata.get("evidence_mode", "real"),
    }


def _evidence_inputs(task, evidence):
    if not isinstance(evidence, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                              for key, value in evidence.items()):
        raise ValidationContractError("evidence must map stage evidence names to file paths")
    if "independent_validation" in evidence:
        raise ValidationContractError("producer-supplied independent_validation is not accepted")
    required = set(STAGE_EVIDENCE[task.stage]) - {"independent_validation"}
    if not required <= evidence.keys():
        raise ValidationContractError(f"missing stage evidence: {sorted(required - evidence.keys())}")
    output = Path(task.input["workspace"]["output"]).resolve()
    result = {}
    for name, value in evidence.items():
        item = _file(value)
        if not Path(item["path"]).is_relative_to(output) or not Path(value).stat().st_size:
            raise ValidationContractError("producer evidence must be nonempty and inside current attempt output")
        result[name] = item
    return result


def _replay(existing, task, worker, validator, candidate_id, evidence):
    expected = {"task_id": task.task_id, "controller": worker,
                "validator": validator, "candidate_id": candidate_id}
    if any(existing.get(key) != value for key, value in expected.items()):
        raise ValidationContractError("validation_id conflicts with its recorded assignment")
    recorded = existing.get("recipe", {}).get("input_evidence", {})
    if {key: value["path"] for key, value in recorded.items()} != evidence:
        raise ValidationContractError("validation_id conflicts with its recorded evidence paths")
    return {"status": "RECORDED", "receipt": existing, "result": existing.get("result"),
            "replayed": True, "evidence_revalidated": False,
            "blocked": existing["state"] != "succeeded",
            "execution_uncertain": existing["state"] == "executing"}


def validate_worker(scheduler, task_id: str, worker: str, lease_token: str, validator: str,
                    candidate_id: str, validation_id: str, evidence: dict[str, str],
                    *, timeout=300) -> dict:
    """Validate once and return the envelope; completing the task is a separate action."""
    task = scheduler.store.get_task(task_id)
    if task is None:
        raise ValueError(f"unknown task: {task_id}")
    run = scheduler.store.run(task.run_id)
    existing = run.metadata.get("managed_validations", {}).get(validation_id)
    if existing is not None:
        return _replay(existing, task, worker, validator, candidate_id, evidence)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValidationContractError("validation timeout must be finite and positive")
    plan = validation_plan(task.input["operator_spec"], task.stage)
    identity = _identity(task, run)
    try:
        check_supported_mode(plan, identity["evidence_mode"])
    except ValidationContractError as error:
        return {"status": "BLOCKED", "error": str(error), "result": None, "receipt": None,
                "replayed": False, "task_id": task_id}
    candidate = get_candidate(scheduler, task_id, candidate_id)
    source_evidence = _evidence_inputs(task, evidence)
    reference_files = plan["contract"]["reference"]["files"]
    _check_files(reference_files)
    recipe_files = {str(path): _file(path)["sha256"] for path in RECIPE_FILES}
    recipe = {
        "path": str(RECIPE_PATH), "sha256": recipe_files[str(RECIPE_PATH)],
        "files": recipe_files, "candidate_entry": plan["contract"]["candidate_entry"],
        "reference_entry": plan["contract"]["reference"]["entry"],
        "input_evidence": source_evidence,
        "controller_process": {"host_id": host_identity(), "pid": os.getpid(),
                               "process_identity": process_identity(os.getpid())},
    }
    receipt, is_new = begin_validation(
        scheduler, task_id, worker, lease_token, validation_id, candidate_id,
        validator, recipe, plan["contract_sha256"], reference_files,
    )
    if not is_new:
        return _replay(receipt, task, worker, validator, candidate_id, evidence)
    root = ensure_external(Path(task.input["workspace"]["output"]) / "validation" / safe_component(validation_id))
    store = ArtifactStore(root)
    records, raw, paths, execution_inputs = {}, {}, {}, {}
    execution_ids = [f"validation:{validation_id}:{role}" for role in ("candidate", "reference", "control")]
    try:
        if root.exists():
            raise ValidationContractError("validation output is already owned; do not reuse another execution")
        plan_path = store.write_json("plan.json", plan)
        execution_inputs[str(plan_path)] = _file(plan_path)["sha256"]
        for role, execution_id in zip(("candidate", "reference", "control"), execution_ids):
            _check_files(recipe_files)
            _check_files(reference_files)
            _check_files(execution_inputs)
            get_candidate(scheduler, task_id, candidate_id)
            request = {
                "schema_version": 1, "validation_id": validation_id, "role": role,
                "plan": plan, "evidence_mode": identity["evidence_mode"], "timeout": timeout,
            }
            if role == "candidate":
                request["entry"] = _file(Path(candidate["candidate_root"]) / plan["contract"]["candidate_entry"])
            elif role == "reference":
                request["entry"] = _file(plan["contract"]["reference"]["entry"])
            else:
                request["reference_observations"] = str(paths["reference"])
            request_path = store.write_json(f"{role}-request.json", request)
            execution_inputs[str(request_path)] = _file(request_path)["sha256"]
            output_path = root / f"{role}-observations.json"
            record = execute_managed(
                scheduler, task.run_id, execution_id,
                [sys.executable, str(REPO_ROOT / "cli/validation/worker_probe.py"),
                 "--request", str(request_path), "--output", str(output_path)],
                cwd=REPO_ROOT, output_dir=root / "executions" / role,
                task_id=task_id, worker=worker, lease_token=lease_token, timeout=timeout,
                env_overrides={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            records[role] = record
            if record["state"] == "UNKNOWN" or not record.get("termination_confirmed"):
                return {"status": "BLOCKED", "error": "validation execution termination is uncertain",
                        "receipt": receipt, "result": None, "execution": record, "replayed": False}
            if record["state"] != "SUCCEEDED" or record.get("returncode") != 0:
                raise ValidationContractError(f"{role} probe did not complete: {record.get('error') or record['state']}")
            if not output_path.is_file():
                raise ValidationContractError(f"{role} probe exited without raw observations")
            _check_files(execution_inputs)
            execution_inputs[str(output_path)] = _file(output_path)["sha256"]
            raw[role] = json.loads(output_path.read_text())
            if role != "control" and raw[role].get("entry") != request["entry"]:
                raise ValidationContractError(f"{role} observations identify a different executed entry")
            paths[role] = output_path
        get_candidate(scheduler, task_id, candidate_id)
        _check_files(recipe_files)
        _check_files(reference_files)
        _check_files(execution_inputs)
        _check_files({value["path"]: value["sha256"] for value in source_evidence.values()})
        verdict = grade_observations(plan, raw["candidate"], raw["reference"], raw["control"],
                                     evidence_mode=identity["evidence_mode"], validation_id=validation_id)
        completed_evidence = dict(evidence)
        completed_evidence.update({f"measurement_{role}": str(path) for role, path in paths.items()})
        completed_evidence["validation_plan"] = str(root / "plan.json")
        completed_evidence["validation_execution"] = str(store.write_json("executions.json", records))
        for role, record in records.items():
            completed_evidence[f"validation_log_{role}"] = record["log_path"]
        hashes = {key: _file(value)["sha256"] for key, value in completed_evidence.items()}
        report = {
            "schema_version": 1, **identity, **verdict, "validator": validator,
            "producer": candidate["producer"], "controller": worker,
            "candidate_id": candidate_id, "managed_validation_id": validation_id,
            "execution_ids": execution_ids, "evidence_sha256": hashes,
        }
        report_path = store.write_json("independent_validation.json", report)
        completed_evidence["independent_validation"] = str(report_path)
        hashes["independent_validation"] = _file(report_path)["sha256"]
        result = {
            "schema_version": 1, "status": verdict["verdict"], "verdict": verdict["verdict"],
            **identity, "managed_validation_id": validation_id, "candidate_id": candidate_id,
            "execution_ids": execution_ids, "evidence": completed_evidence, "evidence_sha256": hashes,
        }
        result_path = store.write_json("result.json", result)
        receipt = finish_validation(scheduler, task.run_id, validation_id, result,
                                    status=verdict["verdict"],
                                    error=None if verdict["verdict"] == "PASS" else "measured checks failed")
        return {"status": verdict["verdict"], "receipt": receipt, "result": result,
                "result_path": str(result_path), "replayed": False,
                "blocked": verdict["verdict"] != "PASS",
                "evidence_mode": identity["evidence_mode"], "artifact_root": str(root)}
    except Exception as error:
        # Unknown execution must retain its active receipt; it cannot be replayed.
        if has_active_task_execution(scheduler.store, task_id):
            return {"status": "BLOCKED", "receipt": receipt, "result": None,
                    "error": str(error), "execution_uncertain": True, "replayed": False}
        try:
            failure = store.write_json("failure.json", {"status": "BLOCKED", "error": str(error)})
            receipt = finish_validation(scheduler, task.run_id, validation_id, None,
                                        status="FAIL", error=str(error))
        except Exception as final_error:
            return {"status": "BLOCKED", "receipt": receipt, "result": None,
                    "error": f"{error}; could not finish receipt: {final_error}", "replayed": False}
        return {"status": "BLOCKED", "receipt": receipt, "result": None, "error": str(error),
                "artifacts": [str(failure)], "replayed": False}
