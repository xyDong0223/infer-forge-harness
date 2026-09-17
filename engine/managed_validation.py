"""Frozen candidates and scheduler-owned independent validation receipts.

Files remain cooperating-tool artifacts, not an OS sandbox. A worker-authored
report cannot substitute for the receipt this module records around the trusted
Runner's execution. Lease ownership and execution-resource ownership are separate.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import json
from pathlib import Path
import time

from core.paths import REPO_ROOT
from core.storage import ArtifactStore, ensure_external
from .interaction import canonical_digest


WORKER_PROTOCOL = "managed-v2"
RECIPE_PATH = REPO_ROOT / "operations/validation/managed_worker.py"
RECIPE_FILES = frozenset(str(REPO_ROOT / name) for name in (
    "operations/validation/managed_worker.py", "operations/validation/worker_probe.py",
    "operations/validation/tensor_diff.py", "cli/validation/worker_probe.py",
    "runners/worker_validation.py",
))


def requires_managed_validation(run, task=None) -> bool:
    return run.metadata.get("worker_protocol") == WORKER_PROTOCOL and (
        task is None or task.stage != "diagnosis"
    )


def _name(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _file(path: str | Path) -> dict:
    lexical = Path(path)
    if not lexical.is_absolute() or lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"validation input must be a regular absolute file: {lexical}")
    if lexical != lexical.resolve():
        raise ValueError(f"validation input cannot contain symbolic links or noncanonical paths: {lexical}")
    content = lexical.read_bytes()
    return {"path": str(lexical), "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content)}


def _inventory(root: Path) -> list[dict]:
    if not root.is_dir() or root.is_symlink() or root != root.resolve():
        raise ValueError("candidate must be a regular canonical directory")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"candidate contains a non-regular file or symlink: {path}")
        if path.is_file():
            entry = _file(path)
            entry["path"] = path.relative_to(root).as_posix()
            files.append(entry)
    if not files:
        raise ValueError("candidate tree must contain at least one regular file")
    return files


def _identity(task, run) -> dict:
    workspace = task.input.get("workspace", {})
    return {
        "run_id": run.run_id, "task_id": task.task_id, "operator_key": task.operator_key,
        "stage": task.stage, "attempt": task.attempt, "attempt_id": workspace.get("attempt_id"),
        "environment_fingerprint": run.environment.get("environment_proof", {}).get("fingerprint"),
        "environment_sha256": canonical_digest(run.environment),
        "model_revision": run.model_revision, "plugin_revision": run.plugin_revision,
        "backend": run.backend,
        "evidence_mode": run.metadata.get("evidence_mode", "real"),
    }


def _task_run(scheduler, task_id):
    task = scheduler.store.get_task(task_id)
    if task is None:
        raise KeyError(f"unknown task: {task_id}")
    run = scheduler.store.run(task.run_id)
    if not requires_managed_validation(run, task):
        raise ValueError("managed validation requires a managed-v2 non-diagnosis task")
    return task, run


def _validation_contract(task) -> dict:
    spec = task.input.get("operator_spec", {})
    contract = spec.get("semantics", {}).get("validation")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("OperatorSpec.semantics.validation is required for managed validation")
    canonical_digest(contract)
    return contract


def _candidate_body(record: dict) -> dict:
    return {key: record[key] for key in (
        "schema_version", "identity", "producer", "controller", "candidate_root",
        "base_revision", "spec_sha256", "files",
    )}


def validate_candidate(task, run, candidate: dict) -> list[str]:
    """Re-read the entire candidate tree, including unexpected additional files."""
    errors = []
    try:
        if candidate["identity"] != _identity(task, run):
            raise ValueError("candidate identity does not match the current task/run")
        output = task.input.get("workspace", {}).get("output")
        if not output or Path(candidate["candidate_root"]) != Path(output).resolve() / "candidate":
            raise ValueError("candidate must belong to the claimed attempt output/candidate")
        if candidate["spec_sha256"] != canonical_digest(task.input["operator_spec"]):
            raise ValueError("candidate OperatorSpec changed")
        if candidate["candidate_id"] != canonical_digest(_candidate_body(candidate)):
            raise ValueError("candidate identity digest is invalid")
        if _inventory(Path(candidate["candidate_root"])) != candidate["files"]:
            raise ValueError("frozen candidate tree changed")
        manifest = _file(candidate["manifest_path"])
        if manifest["sha256"] != candidate["manifest_sha256"]:
            raise ValueError("frozen candidate manifest changed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return errors


def freeze_candidate(scheduler, task_id: str, worker: str, lease_token: str,
                     producer: str, candidate_root: str | Path, base_revision: str) -> dict:
    """Bind one immutable candidate tree to the current claimed attempt."""
    _name(producer, "producer")
    _name(base_revision, "base_revision")
    with scheduler.store.transaction():
        task, run = _task_run(scheduler, task_id)
        scheduler._owned_lease(task_id, worker, lease_token)
        if scheduler.has_active_execution(task_id):
            raise ValueError("cannot freeze a candidate while task execution is active or uncertain")
        _validation_contract(task)
        root = Path(candidate_root)
        output = task.input.get("workspace", {}).get("output")
        if not output or root != Path(output).resolve() / "candidate":
            raise ValueError("candidate_root must be the current attempt output/candidate")
        ensure_external(root)
        body = {
            "schema_version": 1, "identity": _identity(task, run),
            "producer": producer, "controller": worker, "candidate_root": str(root),
            "base_revision": base_revision, "spec_sha256": canonical_digest(task.input["operator_spec"]),
            "files": _inventory(root),
        }
        candidate_id = canonical_digest(body)
        records = deepcopy(run.metadata.get("managed_candidates", {}))
        current = [record for record in records.values()
                   if record["identity"]["task_id"] == task_id
                   and record["identity"]["attempt"] == task.attempt]
        if current:
            if len(current) != 1 or current[0]["candidate_id"] != candidate_id:
                raise ValueError("this attempt already has a different frozen candidate")
            errors = validate_candidate(task, run, current[0])
            if errors:
                raise ValueError("; ".join(errors))
            return current[0]
        manifest_body = {**body, "candidate_id": candidate_id}
        manifest_path = Path(output) / "candidate-manifest.json"
        if manifest_path.exists():
            if manifest_path.is_symlink() or json.loads(manifest_path.read_text()) != manifest_body:
                raise ValueError("candidate manifest conflicts with the frozen candidate")
        else:
            ArtifactStore(output).write_json(manifest_path.name, manifest_body)
        record = {
            **manifest_body, "manifest_path": str(manifest_path),
            "manifest_sha256": _file(manifest_path)["sha256"], "frozen_at": time.time(),
        }
        scheduler._owned_lease(task_id, worker, lease_token)
        records[candidate_id] = record
        scheduler.record_managed_transition(run.run_id, "managed_candidates", records)
        return deepcopy(record)


def get_candidate(scheduler, task_id: str, candidate_id: str | None = None) -> dict:
    """Read and verify a frozen candidate without renewing or claiming a lease."""
    task, run = _task_run(scheduler, task_id)
    records = run.metadata.get("managed_candidates", {})
    if candidate_id is None:
        matches = [record for record in records.values()
                   if record.get("identity") == _identity(task, run)]
        if len(matches) != 1:
            raise ValueError("task must have exactly one current frozen candidate")
        candidate = matches[0]
    else:
        candidate = records.get(candidate_id)
        if candidate is None:
            raise ValueError("unknown frozen candidate")
    errors = validate_candidate(task, run, candidate)
    if errors:
        raise ValueError("; ".join(errors))
    return deepcopy(candidate)


def _validate_recipe(task, candidate: dict, recipe: dict, contract_sha256: str,
                     reference_files: dict) -> None:
    contract = _validation_contract(task)
    if contract_sha256 != canonical_digest(contract):
        raise ValueError("validation contract hash does not match the accepted OperatorSpec")
    if not isinstance(recipe, dict) or recipe.get("path") != str(RECIPE_PATH):
        raise ValueError("managed validation requires the builtin numerical recipe")
    controller = recipe.get("controller_process")
    if (not isinstance(controller, dict)
            or set(controller) != {"host_id", "pid", "process_identity"}
            or type(controller.get("pid")) is not int or controller["pid"] <= 0):
        raise ValueError("validation recipe requires a bound controller process identity")
    _name(controller.get("host_id"), "controller_process.host_id")
    _name(controller.get("process_identity"), "controller_process.process_identity")
    if _file(recipe["path"])["sha256"] != recipe.get("sha256"):
        raise ValueError("builtin validation recipe hash changed")
    files = recipe.get("files")
    if not isinstance(files, dict) or set(files) != RECIPE_FILES:
        raise ValueError("recipe.files must bind the complete builtin validation implementation")
    for name, expected in files.items():
        if _file(name)["sha256"] != expected:
            raise ValueError(f"builtin validation implementation changed: {name}")
    entry = Path(_name(contract.get("candidate_entry"), "validation.candidate_entry"))
    if entry.is_absolute() or ".." in entry.parts or entry.as_posix() in {".", ""}:
        raise ValueError("candidate_entry must be relative to the frozen candidate")
    if recipe.get("candidate_entry") != entry.as_posix():
        raise ValueError("recipe candidate_entry differs from the accepted contract")
    candidate_entry = _file(Path(candidate["candidate_root"]) / entry)
    reference = contract.get("reference")
    if not isinstance(reference, dict) or not isinstance(reference_files, dict) or not reference_files:
        raise ValueError("independent reference files are required")
    _name(reference.get("provenance"), "reference.provenance")
    if reference_files != reference.get("files") or recipe.get("reference_entry") != reference.get("entry"):
        raise ValueError("reference files/entry differ from the accepted OperatorSpec")
    reference_entry = reference.get("entry")
    if reference_entry not in reference_files:
        raise ValueError("the executed reference entry must be bound in reference_files")
    candidate_root = Path(candidate["candidate_root"])
    for name, expected in reference_files.items():
        item = _file(name)
        if Path(name).is_relative_to(candidate_root):
            raise ValueError("independent reference must not belong to the candidate tree")
        if item["sha256"] != expected:
            raise ValueError(f"independent reference file changed: {name}")
    if reference_files[reference_entry] == candidate_entry["sha256"]:
        raise ValueError("reference entry must not duplicate the candidate entry")


def begin_validation(scheduler, task_id: str, worker: str, lease_token: str,
                     validation_id: str, candidate_id: str, validator: str,
                     recipe: dict, contract_sha256: str, reference_files: dict) -> tuple[dict, bool]:
    """Accept one validation execution; replay never authorizes another process."""
    _name(validation_id, "validation_id")
    _name(validator, "validator")
    request = {"task_id": task_id, "controller": worker, "candidate_id": candidate_id,
               "validator": validator, "recipe": recipe, "contract_sha256": contract_sha256,
               "reference_files": reference_files}
    request_sha256 = canonical_digest(request)
    with scheduler.store.transaction():
        task, run = _task_run(scheduler, task_id)
        records = deepcopy(run.metadata.get("managed_validations", {}))
        existing = records.get(validation_id)
        if existing is not None:
            if existing["request_sha256"] != request_sha256:
                raise ValueError("validation_id conflicts with an accepted validation request")
            return existing, False
        row = scheduler._owned_lease(task_id, worker, lease_token)
        if scheduler.has_active_execution(task_id):
            raise ValueError("task execution is active or uncertain")
        if any(record["identity"]["task_id"] == task_id and record["state"] == "executing"
               for record in records.values()):
            raise ValueError("a validation execution is already active or uncertain for this task")
        candidate = run.metadata.get("managed_candidates", {}).get(candidate_id)
        if candidate is None:
            raise ValueError("unknown frozen candidate")
        errors = validate_candidate(task, run, candidate)
        if errors:
            raise ValueError("; ".join(errors))
        if worker != candidate["controller"]:
            raise ValueError("validation controller differs from the frozen candidate controller")
        if validator == candidate["producer"]:
            raise ValueError("validator must be distinct from the actual candidate producer")
        _validate_recipe(task, candidate, recipe, contract_sha256, reference_files)
        scheduler._owned_lease(task_id, worker, lease_token)
        record = {
            "schema_version": 1, "validation_id": validation_id, "identity": _identity(task, run),
            **request, "producer": candidate["producer"], "request_sha256": request_sha256,
            "lease_token_sha256": hashlib.sha256(row["lease_token"].encode()).hexdigest(),
            "state": "executing", "execution_status": "IN_PROGRESS_OR_UNKNOWN",
            "accepted_at": time.time(),
            "execution_ids": [f"validation:{validation_id}:{role}"
                              for role in ("candidate", "reference", "control")],
        }
        records[validation_id] = record
        scheduler.record_managed_transition(run.run_id, "managed_validations", records)
        return deepcopy(record), True


def result_envelope_digest(result: dict) -> str:
    return canonical_digest({key: value for key, value in result.items()
                             if key not in {"managed_validation_id", "_submission", "artifact_manifest"}})


def _receipt_digest(record: dict) -> str:
    return canonical_digest({key: value for key, value in record.items() if key != "receipt_sha256"})


def _receipt_inputs(task, run, receipt: dict) -> tuple[dict | None, list[str]]:
    errors = []
    candidate = run.metadata.get("managed_candidates", {}).get(receipt.get("candidate_id"))
    if candidate is None:
        return None, ["managed validation references an unknown frozen candidate"]
    errors.extend(validate_candidate(task, run, candidate))
    if receipt.get("identity") != _identity(task, run):
        errors.append("managed validation identity does not match the current task/run")
    if (receipt.get("producer") != candidate.get("producer")
            or receipt.get("controller") != candidate.get("controller")
            or receipt.get("validator") == candidate.get("producer")):
        errors.append("managed validation producer/controller/validator binding is invalid")
    try:
        _validate_recipe(task, candidate, receipt["recipe"], receipt["contract_sha256"], receipt["reference_files"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return candidate, errors


def _execution_errors(task, run, receipt: dict, *, terminal: bool) -> list[str]:
    errors = []
    expected = [f"validation:{receipt['validation_id']}:{role}"
                for role in ("candidate", "reference", "control")]
    if receipt.get("execution_ids") != expected:
        return ["validation execution identities differ from the fixed independent recipe"]
    records = run.metadata.get("execution_records", {})
    for identity in expected:
        record = records.get(identity)
        if not isinstance(record, dict):
            errors.append(f"managed validation is missing executor record: {identity}")
            continue
        if (record.get("run_id") != run.run_id or record.get("task_id") != task.task_id
                or record.get("task_attempt") != task.attempt
                or record.get("worker") != receipt.get("controller")
                or record.get("lease_token_sha256") != receipt.get("lease_token_sha256")):
            errors.append(f"validation executor ownership does not match the receipt: {identity}")
        if terminal:
            if (record.get("state") != "SUCCEEDED" or record.get("termination_confirmed") is not True
                    or type(record.get("returncode")) is not int or record["returncode"] != 0):
                errors.append(f"validation executor has no confirmed successful termination: {identity}")
            if receipt.get("execution_requests", {}).get(identity) != record.get("request_sha256"):
                errors.append(f"validation executor request differs from the receipt: {identity}")
    return errors


def finish_validation(scheduler, run_id: str, validation_id: str, result: dict | None,
                      *, status: str = "PASS", error=None) -> dict:
    """Record the trusted Runner's outcome after rechecking inputs and evidence."""
    if status not in {"PASS", "FAIL"}:
        raise ValueError("validation status must be PASS or FAIL")
    finish_body = {"status": status, "result": result, "error": error}
    finish_sha256 = canonical_digest(finish_body)
    with scheduler.store.transaction():
        run = scheduler.store.run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        records = deepcopy(run.metadata.get("managed_validations", {}))
        receipt = records.get(validation_id)
        if receipt is None:
            raise KeyError(f"unknown managed validation: {validation_id}")
        if receipt["state"] != "executing":
            if receipt.get("finish_sha256") != finish_sha256:
                raise ValueError("validation outcome conflicts with the recorded receipt")
            return receipt
        task, run = _task_run(scheduler, receipt["identity"]["task_id"])
        row = scheduler._owned_lease(task.task_id, receipt["controller"],
                                     scheduler.store.get_task(task.task_id).lease_token)
        if not hmac.compare_digest(receipt["lease_token_sha256"], hashlib.sha256(row["lease_token"].encode()).hexdigest()):
            raise ValueError("validation belongs to an obsolete lease")
        if receipt["identity"] != _identity(task, run):
            raise ValueError("validation belongs to an obsolete task attempt or environment")
        if status == "PASS":
            if not isinstance(result, dict):
                raise ValueError("passing validation requires a result envelope")
            if result.get("managed_validation_id") not in (None, validation_id):
                raise ValueError("result belongs to another managed validation")
            if result.get("candidate_id") != receipt["candidate_id"]:
                raise ValueError("result belongs to another frozen candidate")
            _, errors = _receipt_inputs(task, run, receipt)
            errors.extend(_execution_errors(task, run, receipt, terminal=False))
            if result.get("execution_ids") != receipt["execution_ids"]:
                errors.append("result execution_ids differ from the accepted validation")
            from .result_validation import _validate_legacy_result
            errors.extend(_validate_legacy_result(task, run, result, receipt["producer"]))
            report_path = (result.get("evidence") or {}).get("independent_validation")
            if report_path:
                path = Path(report_path)
                if not path.is_absolute():
                    path = Path(run.metadata["artifact_root"]) / path
                try:
                    report = json.loads(path.read_text())
                    if report.get("validator") != receipt["validator"]:
                        errors.append("validation report does not match the bound validator")
                except (OSError, ValueError, AttributeError) as exc:
                    errors.append(f"cannot read independent validation identity: {exc}")
            if errors:
                raise ValueError("managed validation rejected: " + "; ".join(errors))
        scheduler._owned_lease(task.task_id, receipt["controller"], row["lease_token"])
        receipt.update(
            state="succeeded" if status == "PASS" else "failed", execution_status="FINISHED",
            status=status, finished_at=time.time(), finish_sha256=finish_sha256, error=error,
            result_envelope_sha256=result_envelope_digest(result) if result is not None else None,
            evidence_sha256=deepcopy(result.get("evidence_sha256", {})) if result is not None else {},
            result=deepcopy(result),
            execution_requests={identity: run.metadata.get("execution_records", {}).get(identity, {}).get("request_sha256")
                                for identity in receipt["execution_ids"]},
        )
        receipt["receipt_sha256"] = _receipt_digest(receipt)
        records[validation_id] = receipt
        scheduler.record_managed_transition(run_id, "managed_validations", records)
        return deepcopy(receipt)


def validate_managed_result(task, run, result: dict, controller: str) -> list[str]:
    if not requires_managed_validation(run, task):
        return []
    validation_id = result.get("managed_validation_id")
    if not isinstance(validation_id, str) or not validation_id:
        return ["managed-v2 result requires managed_validation_id"]
    receipt = run.metadata.get("managed_validations", {}).get(validation_id)
    if receipt is None:
        return ["managed validation receipt was not recorded by the scheduler"]
    _, errors = _receipt_inputs(task, run, receipt)
    errors.extend(_execution_errors(task, run, receipt, terminal=True))
    if receipt.get("state") != "succeeded" or receipt.get("status") != "PASS":
        errors.append("managed validation has no completed passing execution receipt")
    if receipt.get("controller") != controller:
        errors.append("managed result controller differs from the accepted validation")
    if receipt.get("receipt_sha256") != _receipt_digest(receipt):
        errors.append("managed validation receipt digest mismatch")
    if receipt.get("result_envelope_sha256") != result_envelope_digest(result):
        errors.append("result envelope differs from the independently validated result")
    if receipt.get("evidence_sha256") != result.get("evidence_sha256"):
        errors.append("result evidence differs from the independently validated evidence")
    return errors


def reconcile_managed_validation(scheduler, run_id: str, validation_id: str, *, observer) -> dict:
    """Fail an orphan receipt only after a trusted observer proves controller exit.

    Process observation is outside the database lock. The exact observed receipt
    must still be current once the transaction starts. Lease expiry alone never
    authorizes recovery, and this function cannot record a passing validation.
    """
    from .execution import has_active_task_execution

    if not callable(observer):
        raise ValueError("validation reconciliation requires a trusted controller observer")
    run = scheduler.store.run(run_id)
    if run is None:
        raise KeyError(f"unknown run: {run_id}")
    original = deepcopy(run.metadata.get("managed_validations", {}).get(validation_id))
    if original is None:
        raise KeyError(f"unknown managed validation: {validation_id}")
    if original.get("state") != "executing":
        return original
    controller = original.get("recipe", {}).get("controller_process")
    if not isinstance(controller, dict) or not controller:
        raise ValueError("validation receipt has no bound controller process identity")
    source_sha256 = canonical_digest(original)
    observed = observer(deepcopy(original))
    canonical_digest(observed)
    with scheduler.store.transaction():
        run = scheduler.store.run(run_id)
        records = deepcopy(run.metadata.get("managed_validations", {}))
        receipt = records.get(validation_id)
        if receipt is None or canonical_digest(receipt) != source_sha256:
            raise ValueError("validation receipt changed during controller observation")
        if (not isinstance(observed, dict) or observed.get("controller_absent") is not True
                or observed.get("controller_process") != controller):
            return deepcopy(receipt)
        if has_active_task_execution(scheduler.store, receipt["identity"]["task_id"]):
            return deepcopy(receipt)
        error = "validation controller exited before recording a complete outcome"
        receipt.update(
            state="failed", status="FAIL", execution_status="FINISHED", finished_at=time.time(),
            result=None, result_envelope_sha256=None, evidence_sha256={}, error=error,
            finish_sha256=canonical_digest({"status": "FAIL", "result": None, "error": error}),
            reconciliation={"source_sha256": source_sha256, "observation": deepcopy(observed)},
            execution_requests={identity: run.metadata.get("execution_records", {}).get(identity, {}).get("request_sha256")
                                for identity in receipt.get("execution_ids", [])},
        )
        receipt["receipt_sha256"] = _receipt_digest(receipt)
        records[validation_id] = receipt
        scheduler.record_managed_transition(run_id, "managed_validations", records)
        return deepcopy(receipt)


def managed_result_binding(run, result: dict) -> dict:
    """Project immutable identities into submission and final delivery snapshots."""
    validation_id = result.get("managed_validation_id")
    if not requires_managed_validation(run) or not isinstance(validation_id, str):
        return {}
    receipt = run.metadata.get("managed_validations", {}).get(validation_id)
    if not receipt:
        return {}
    return {"managed_validation_id": receipt["validation_id"],
            "managed_validation_sha256": receipt.get("receipt_sha256"),
            "candidate_id": receipt["candidate_id"], "producer": receipt["producer"],
            "validator": receipt["validator"], "controller": receipt["controller"],
            "execution_ids": receipt.get("execution_ids", []),
            "evidence_mode": receipt["identity"]["evidence_mode"]}
