"""Durable graph handoffs into the operator scheduler, never synthetic task rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.storage import ArtifactStore, RunPaths, ensure_external, locate_attempt
from core.target import canonical_hardware
from core.task_execution import default_execution_catalog
from validators.accuracy_validator import validate_accuracy_report
from validators.deployment_validator import (
    ENVIRONMENT_ARTIFACTS, validate_deployment_status, validate_environment_status,
)
from validators.shim_validator import validate_shim_handoff
from validators.operator_lifecycle_validator import validate_dispatch, validate_integration

from .discovery import operator_specs_from_report
from .result_validation import validate_result
from .managed_validation import managed_result_binding
from .scheduler import EventStore, TaskScheduler


# Delivery policy selects mandatory facts; each Task owns its status filename.
# Optional triage/vendor branches must not become mandatory delivery gates.
DELIVERY_FACTS = frozenset({
    "ModelRequest", "EnvironmentProof", "ModelSupportCard", "CapabilityMatch",
    "GapClassification", "CapabilityEvaluation", "DeploymentPlan", "ToyBringupReport",
    "TorchShimRegistry", "DeploymentProof", "AccuracyDifferential", "ServingBaseline",
    "OperatorTaskDispatch", "OperatorIntegration", "SupportMatrixEntry",
})
FACT_STATUS = {task.produces: task.status_file for task in default_execution_catalog().values()
               if task.produces in DELIVERY_FACTS}
if set(FACT_STATUS) != DELIVERY_FACTS:
    raise ValueError("delivery facts require executable Task definitions")


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence must be a regular non-symlink file: {path}")
    content = path.read_bytes()
    if not content:
        raise ValueError(f"empty evidence: {path}")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}


def _identifier(entry: dict) -> str | None:
    return (entry.get("operator_id") or entry.get("operator") or entry.get("symbol")
            or entry.get("name") or
            (entry.get("axis") if entry.get("class") == "CAPABILITY_MISSING" else None))


def _accuracy_errors(report: dict) -> list[str]:
    if (not isinstance(report.get("cases"), list)
            or not all(isinstance(case, dict) for case in report["cases"])
            or not isinstance(report.get("reference"), dict)
            or not isinstance(report.get("candidate"), dict)):
        return ["accuracy evidence requires case objects and reference/candidate objects"]
    return validate_accuracy_report(report, {"require_top1_match_on_all_cases": True})


def _deployment_errors(report: dict) -> list[str]:
    if not isinstance(report.get("checks"), dict) or not isinstance(report.get("artifacts"), list):
        return ["deployment evidence requires checks and an artifact list"]
    return validate_deployment_status(report)


def _entries(report: Any) -> list:
    if isinstance(report, dict):
        for name in ("entries", "gaps", "findings"):
            if name in report:
                values = report[name]
                break
        else:
            if _identifier(report):
                return [report]
            raise ValueError("discovery report must explicitly contain entries, gaps, or findings")
    else:
        values = report
    if not isinstance(values, list):
        raise ValueError("discovery entries must be a list")
    return values


class GraphSchedulerBridge:
    def __init__(
        self, state: Path, run_id: str, subject: str, artifact_root: Path,
        environment: dict, *, execute: bool = True,
    ):
        self.state = ensure_external(state)
        self.artifact_root = ensure_external(artifact_root)
        self.run_id = run_id
        self.subject = subject
        self.environment = dict(environment)
        self.execute = execute
        # Open read-only first: a typo must not create a database or migrate it.
        probe = EventStore(self.state, readonly=True)
        try:
            run = probe.run(run_id)
            if run is None:
                raise ValueError(f"unknown adaptation run: {run_id}")
            if run.model_id != subject:
                raise ValueError("graph subject does not match adaptation run model")
            if not run.metadata.get("artifact_root") or (
                ensure_external(run.metadata["artifact_root"]) != self.artifact_root
            ):
                raise ValueError("graph artifact root does not match adaptation run")
            self._check_identity(environment, run)
        except BaseException:
            probe.close()
            raise
        if execute:
            probe.close()
            self.scheduler = TaskScheduler(self.state)
            self.scheduler.record_graph_transition(run_id, "graph_environment_required", {})
            if self.run.metadata.get("graph_environment"):
                self._environment_errors()
        else:
            self.scheduler = TaskScheduler(probe)

    @property
    def run(self):
        run = self.scheduler.store.run(self.run_id)
        if run is None:
            raise ValueError(f"unknown adaptation run: {self.run_id}")
        return run

    @property
    def evidence_mode(self) -> str:
        return self.run.metadata.get("evidence_mode", "real")

    def close(self) -> None:
        self.scheduler.store.close()

    def _check_identity(
        self, data: dict, run=None, *, environment_proof=False, imported_environment=False,
    ) -> None:
        if not isinstance(data, dict):
            raise ValueError("evidence identity must be an object")
        run = run or self.run
        expected = {
            "run_id": run.run_id, "subject": run.model_id, "model_id": run.model_id,
            "model": run.model_id, "backend": run.backend,
            "model_revision": run.model_revision, "plugin_revision": run.plugin_revision,
            "evidence_mode": run.metadata.get("evidence_mode", "real"),
        }
        expected.update({key: value for key, value in run.environment.items()
                         if isinstance(value, (str, int, bool)) and key not in expected})
        expected = {**self.environment, **expected}
        if imported_environment:
            expected.pop("run_id", None)
        fingerprint = run.environment.get("environment_proof", {}).get("fingerprint")
        if fingerprint:
            expected["environment_fingerprint"] = fingerprint
            expected["fingerprint"] = fingerprint
            if not environment_proof:
                expected["pod"] = run.environment["environment_proof"]["pod"]
        for key, value in expected.items():
            if key in data:
                actual = data[key]
                if key == "hardware":
                    actual, value = canonical_hardware(actual), canonical_hardware(value)
                if key == "plugin" and isinstance(actual, str) and isinstance(value, str):
                    actual, value = actual.replace("_", "-"), value.replace("_", "-")
                if actual != value:
                    raise ValueError(f"graph evidence {key} does not match adaptation run")
        target = data.get("target")
        if isinstance(target, dict):
            runtime = target.get("runtime", {})
            if not isinstance(runtime, dict):
                raise ValueError("evidence target runtime must be an object")
            flat = {**target, **runtime}
            flat.pop("target", None)
            revisions = flat.pop("revisions", {})
            if not isinstance(revisions, dict):
                raise ValueError("evidence target revisions must be an object")
            flat.update({f"{key}_revision": value for key, value in revisions.items()})
            # EnvironmentProof may smoke-test a different base model.
            if environment_proof:
                flat.pop("model", None)
                flat.pop("model_revision", None)
            self._check_identity(flat, run)
        workspace = data.get("workspace_identity")
        if workspace is not None and not imported_environment and (
            not isinstance(workspace, dict) or workspace.get("run_id") != run.run_id
        ):
            raise ValueError("environment proof belongs to a different run")
        if isinstance(data.get("candidate"), dict):
            self._check_identity(data["candidate"], run)

    def _require_execute(self) -> None:
        if not self.execute:
            raise ValueError("graph bridge is read-only in plan mode")

    def _environment_errors(self) -> list[str]:
        run = self.run
        if run.status != "ENVIRONMENT_READY":
            return [f"environment is not ready: {run.status}"]
        accepted = run.metadata.get("graph_environment")
        if not accepted:
            return ["graph has not accepted a persisted environment proof"]
        try:
            for item in accepted["files"]:
                if _file(Path(item["path"])) != item:
                    raise ValueError(f"accepted environment evidence changed: {item['path']}")
            proof = _load(Path(accepted["status_path"]))
            if not isinstance(proof.get("checks"), dict):
                raise ValueError("environment proof checks must be an object")
            errors = validate_environment_status(proof)
            if errors:
                raise ValueError("; ".join(errors))
            self._check_identity(
                proof, environment_proof=True,
                imported_environment=accepted.get("imported", False),
            )
            if proof.get("evidence_mode", "real") != self.evidence_mode:
                raise ValueError("environment evidence_mode does not match the run")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if self.execute:
                self.scheduler.record_environment_failure(self.run_id, {}, str(exc))
            return [str(exc)]
        return []

    def bind_environment(self, artifacts: Path) -> dict:
        self._require_execute()
        proof: dict = {}
        try:
            root = ensure_external(artifacts)
            path = root / "status.json"
            bound = self.run.environment.get("environment_proof", {})
            previously_bound = bool(
                bound.get("artifact_root") and Path(bound["artifact_root"]).resolve() == root
                and set(ENVIRONMENT_ARTIFACTS) <= set(bound.get("evidence_sha256", {}))
            )
            if previously_bound:
                for name, digest in bound["evidence_sha256"].items():
                    if _file(root / name)["sha256"] != digest:
                        raise ValueError(f"accepted environment evidence changed: {root / name}")
            if not root.is_relative_to(self.artifact_root) and not previously_bound:
                raise ValueError("external environment evidence requires an exact scheduler binding")
            accepted = self.run.metadata.get("graph_environment")
            owner = locate_attempt(root)
            imported = (
                accepted.get("imported", False)
                if accepted and accepted.get("status_path") == str(path)
                else previously_bound and (
                    not root.is_relative_to(self.artifact_root)
                    or not owner or owner.identity["run_id"] != self.run_id
                )
            )
            proof = _load(path)
            if not isinstance(proof.get("checks"), dict):
                raise ValueError("environment proof checks must be an object")
            errors = validate_environment_status(proof)
            if errors:
                raise ValueError("; ".join(errors))
            self._check_identity(
                proof, environment_proof=True, imported_environment=imported,
            )
            if proof.get("evidence_mode", "real") != self.evidence_mode:
                raise ValueError("environment evidence_mode does not match the run")
            if proof.get("artifact_root") and Path(proof["artifact_root"]).resolve() != root:
                raise ValueError("environment proof artifact_root does not match its location")
            files = [_file(path)]
            for name in proof["artifacts"]:
                evidence = ArtifactStore(root).path(name)
                files.append(_file(evidence))
            record = {"status_path": str(path), "files": files, "imported": imported}
            if accepted:
                # Rebinding a modified accepted file must never silently bless corruption.
                old_paths = {item["path"]: item["sha256"] for item in accepted["files"]}
                for item in files:
                    if item["path"] in old_paths and old_paths[item["path"]] != item["sha256"]:
                        raise ValueError(f"accepted environment evidence changed: {item['path']}")
            if self.run.status == "ENVIRONMENT_READY" and accepted == record:
                return record
            self.scheduler.bind_environment(self.run_id, proof, root)
            self.scheduler.record_graph_transition(self.run_id, "graph_environment", record)
            return record
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.scheduler.record_environment_failure(self.run_id, proof, str(exc))
            raise ValueError(f"environment proof rejected: {exc}") from exc

    def dispatch(
        self, gaps_path: Path, out_dir: Path, operator_report: Path | None = None, *,
        _channel: str = "graph_discovery",
    ) -> dict:
        self._require_execute()
        out_dir = ensure_external(out_dir)
        if not out_dir.is_relative_to(self.artifact_root):
            raise ValueError("dispatch output must belong to the run artifact root")
        errors = self._environment_errors()
        tasks, operator_keys, reports = [], [], []
        try:
            source = json.loads(gaps_path.read_text(encoding="utf-8"))
            gaps = _entries(source)
            reports.append(_file(gaps_path))
            report = json.loads((operator_report or gaps_path).read_text(encoding="utf-8"))
            if operator_report:
                reports.append(_file(operator_report))
            entries = _entries(report)
            if isinstance(report, dict):
                self._check_identity(report)
                if report.get("unmapped_signals"):
                    errors.append("operator report contains unmapped signals")
            if isinstance(source, dict):
                self._check_identity(source)
            resolved = set()
            if not self._environment_errors():
                for index, entry in enumerate(entries):
                    try:
                        if not isinstance(entry, dict):
                            raise ValueError("operator entry must be an object")
                        if entry.get("status") == "WAIVED" and not entry.get("reason"):
                            raise ValueError("waived operator evidence requires a reason")
                        self._check_identity(entry)
                        normalized = dict(entry)
                        if _identifier(entry):
                            normalized.setdefault("operator_id", _identifier(entry))
                        specs = operator_specs_from_report(
                            {"entries": [normalized]}, model_id=self.run.model_id,
                            backend=self.run.backend, model_revision=self.run.model_revision,
                            plugin_revision=self.run.plugin_revision,
                            environment={"fingerprint":
                                         self.run.environment["environment_proof"]["fingerprint"]},
                        )
                        for spec in specs:
                            existing = self.scheduler.store.operator(self.run_id, spec.operator_key)
                            if existing and existing.operator_id != spec.operator_id:
                                raise ValueError(
                                    "operator contract aliases another operator; measured semantics "
                                    "must distinguish its dispatch target"
                                )
                            task = self.scheduler.discover_operator(self.run_id, spec)
                            tasks.append(task.task_id)
                            operator_keys.append(task.operator_key)
                            resolved.add(spec.operator_id)
                    except (ValueError, TypeError, KeyError) as exc:
                        errors.append(f"entries[{index}]: {exc}")
            for index, gap in enumerate(gaps):
                if not isinstance(gap, dict):
                    errors.append(f"gaps[{index}] must be an object")
                elif gap.get("status") == "WAIVED" and gap.get("reason"):
                    continue
                elif not _identifier(gap) or _identifier(gap) not in resolved:
                    errors.append(f"unresolved actionable gap: {_identifier(gap) or index}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
        state = "DISPATCH_BLOCKED" if errors else "DISPATCHED" if tasks else "DISPATCH_SKIPPED"
        payload = {
            "state": state, "run_id": self.run_id, "subject": self.subject,
            "scheduler_state": str(self.state), "artifact_root": str(self.artifact_root),
            "evidence_mode": self.evidence_mode,
            "environment_fingerprint": self.run.environment.get("environment_proof", {}).get("fingerprint"),
            "task_ids": sorted(set(tasks)), "operator_keys": sorted(set(operator_keys)),
            "requests": sorted(set(tasks)), "request_count": len(set(tasks)),
            "reports": reports, "errors": errors,
        }
        validation_errors = validate_dispatch(payload)
        if validation_errors:
            payload["state"] = "DISPATCH_BLOCKED"
            payload["errors"] = [*errors, *validation_errors]
        payload["validator"] = {"passed": not validation_errors, "errors": validation_errors}
        store = ArtifactStore(out_dir)
        store.write_json("dispatch_status.json", payload, overwrite=True)
        self.scheduler.record_graph_transition(self.run_id, _channel, {
            **payload, "status_path": str((out_dir / "dispatch_status.json").resolve()),
            "status_sha256": _file(out_dir / "dispatch_status.json")["sha256"],
        })
        return payload

    def dispatch_shims(self, artifacts: Path) -> dict:
        """Discover late runtime shims without replacing the earlier gap handoff."""
        self._require_execute()
        root = ensure_external(artifacts)
        if not root.is_relative_to(self.artifact_root):
            raise ValueError("shim evidence must belong to the run artifact root")
        attempt = RunPaths(self.artifact_root, self.run_id).allocate_attempt("graph-shim-discovery")
        return self.dispatch(
            root / "torch_shim_registry.json", attempt.output,
            _channel="graph_shim_discovery",
        )

    def delivery_status(self) -> dict:
        errors = self._environment_errors()
        run = self.run
        discovery = run.metadata.get("graph_discovery")
        waiting = not discovery
        discoveries = [run.metadata[key] for key in ("graph_discovery", "graph_shim_discovery")
                       if key in run.metadata]
        for accepted in discoveries:
            if accepted.get("state") not in {"DISPATCHED", "DISPATCH_SKIPPED"}:
                errors.extend(accepted.get("errors") or ["operator discovery is blocked"])
            try:
                if _file(Path(accepted["status_path"]))["sha256"] != accepted["status_sha256"]:
                    raise ValueError("accepted dispatch status changed")
                for report in accepted["reports"]:
                    if _file(Path(report["path"])) != report:
                        raise ValueError(f"accepted discovery report changed: {report['path']}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        tasks = self.scheduler.store.tasks(self.run_id)
        keys = sorted({task.operator_key for task in tasks if task.stage != "diagnosis"})
        for task in tasks:
            if task.status == "failed":
                errors.append(f"{task.task_id}: failed")
            elif task.stage == "diagnosis" and task.output.get("next_action") == "BLOCKED":
                errors.append(f"{task.task_id}: diagnosis BLOCKED")
            elif task.status == "succeeded":
                submission = task.output.get("_submission", {})
                producer = submission.get("worker")
                if not producer or submission.get("attempt") != task.attempt:
                    errors.append(f"{task.task_id}: missing current producer submission")
                else:
                    errors.extend(f"{task.task_id}: {error}" for error in
                                  validate_result(task, run, task.output, producer))
            elif task.stage == "diagnosis":
                errors.append(f"{task.task_id}: unresolved diagnosis")
        for key in keys:
            for stage in ("torch", "xpu", "integration"):
                if not any(task.operator_key == key and task.stage == stage
                           and task.status == "succeeded" for task in tasks):
                    waiting = True
        status = {
            "state": "OPERATORS_BLOCKED" if errors else
                     "WAITING_FOR_OPERATORS" if waiting else "OPERATORS_READY",
            "run_id": self.run_id, "subject": self.subject, "evidence_mode": self.evidence_mode,
            "scheduler_state": str(self.state), "artifact_root": str(self.artifact_root),
            "environment_fingerprint": run.environment.get("environment_proof", {}).get("fingerprint"),
            "task_ids": sorted(task.task_id for task in tasks), "operator_keys": keys,
            "task_results": [
                {
                    "task_id": task.task_id, "operator_key": task.operator_key,
                    "stage": task.stage, "status": task.status, "attempt": task.attempt,
                    "evidence_sha256": task.output.get("evidence_sha256", {}),
                    "submission": task.output.get("_submission", {}),
                    **({"managed_validation": binding} if
                       (binding := managed_result_binding(run, task.output)) else {}),
                }
                for task in sorted(tasks, key=lambda task: task.task_id)
            ],
            "errors": errors,
        }
        snapshot = {key: status[key] for key in (
            "state", "run_id", "evidence_mode", "environment_fingerprint",
            "task_ids", "operator_keys", "task_results",
        )}
        status["snapshot_token"] = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ).hexdigest()
        return status

    def _regression_snapshot(self, root: Path, gate: dict, kind: str) -> dict:
        attempt = locate_attempt(root)
        if not attempt or attempt.identity["run_id"] != self.run_id:
            raise ValueError(f"final {kind} must belong to a managed run attempt")
        path = attempt.input / "scheduler_snapshot.json"
        evidence = _file(path)
        snapshot = _load(path)
        if snapshot.get("state") != "OPERATORS_READY":
            raise ValueError(f"final {kind} must start after the operator gate is OPERATORS_READY")
        for key in ("run_id", "evidence_mode", "environment_fingerprint",
                    "task_ids", "operator_keys", "task_results", "snapshot_token"):
            if key not in snapshot or snapshot[key] != gate[key]:
                raise ValueError(f"final {kind} scheduler snapshot {key} does not match current results")
        return evidence

    def _service_snapshot(self, service_root: Path, gate: dict) -> dict:
        return self._regression_snapshot(service_root, gate, "service")

    def _accuracy_snapshot(self, accuracy_root: Path, service_root: Path, gate: dict) -> dict:
        evidence = self._regression_snapshot(accuracy_root, gate, "accuracy")
        snapshot = _load(Path(evidence["path"]))
        if snapshot.get("service_proof") != _file(service_root / "status.json"):
            raise ValueError("final accuracy must reference the current service proof")
        return evidence

    def validate_baseline(self, baseline_path: Path) -> dict:
        baseline = _load(baseline_path)
        self._check_identity(baseline)
        self._check_identity(baseline.get("environment", {}))
        if baseline.get("status") != "FROZEN" or not baseline.get("baseline_id"):
            raise ValueError("serving baseline must be FROZEN")
        if baseline.get("subject") != self.subject:
            raise ValueError("serving baseline subject does not match run")
        for field, expected in (("service", "DEPLOYMENT_READY"), ("accuracy", "ACCURACY_PASS")):
            reference = baseline.get(field, {})
            if reference.get("state") != expected or not reference.get("artifact"):
                raise ValueError(f"baseline {field} must reference passed evidence")
            path = Path(reference["artifact"])
            if not path.is_absolute():
                path = baseline_path.parent / path
            if not path.resolve().is_relative_to(self.artifact_root):
                raise ValueError(f"baseline {field} evidence belongs to another run")
            owner = locate_attempt(path)
            if not owner or owner.identity["run_id"] != self.run_id:
                raise ValueError(f"baseline {field} evidence lacks managed run provenance")
            report = _load(path)
            self._check_identity(report)
            if field == "accuracy" and report.get("revision") not in (None, self.run.model_revision):
                raise ValueError("baseline accuracy model revision does not match run")
            if report.get("state", report.get("status")) != expected:
                raise ValueError(f"baseline {field} evidence is not passing")
            if reference.get("sha256") != _file(path)["sha256"]:
                raise ValueError(f"baseline {field} evidence hash changed")
            if field == "service":
                errors = _deployment_errors(report)
                if errors:
                    raise ValueError("; ".join(errors))
                for name in report.get("artifacts", []):
                    _file(ArtifactStore(path.parent).path(name))
            else:
                errors = _accuracy_errors(report)
            if errors:
                raise ValueError("; ".join(errors))
        return baseline

    def finalize(self, facts: dict[str, Path]) -> dict:
        self._require_execute()
        try:
            # Keep discovery/completion on other connections behind the final
            # comparison until its receipt and metadata have been published.
            with self.scheduler.store.transaction():
                return self._finalize(facts)
        except (OSError, ValueError, KeyError, TypeError) as error:
            gate = self.delivery_status()
            self.scheduler.record_graph_transition(
                self.run_id, "graph_delivery",
                {**gate, "state": "OPERATORS_BLOCKED", "errors": [str(error)]},
            )
            raise

    def _finalize(self, facts: dict[str, Path]) -> dict:
        gate = self.delivery_status()
        errors = list(gate["errors"])
        if gate["state"] != "OPERATORS_READY":
            errors.append(f"operator gate is {gate['state']}")
        errors.extend(f"missing graph fact: {name}" for name in FACT_STATUS if name not in facts)
        inventory, statuses = {}, {}
        try:
            if errors:
                raise ValueError("; ".join(errors))
            for name, state_file in FACT_STATUS.items():
                root = ensure_external(facts[name])
                imported = (
                    name == "EnvironmentProof"
                    and self.run.metadata["graph_environment"].get("imported", False)
                    and root / state_file == Path(self.run.metadata["graph_environment"]["status_path"])
                )
                if not root.is_relative_to(self.artifact_root) and not imported:
                    raise ValueError(f"{name} fact is outside the run artifact root")
                attempt = locate_attempt(root)
                if not imported and (not attempt or attempt.identity["run_id"] != self.run_id):
                    raise ValueError(f"{name} fact is not owned by this managed run")
                status = _load(root / state_file)
                self._check_identity(
                    status, environment_proof=name == "EnvironmentProof",
                    imported_environment=imported,
                )
                validator = status.get("validator", {})
                if not isinstance(validator, dict) or validator.get("passed") is False or validator.get("errors"):
                    raise ValueError(f"{name} fact validator rejected evidence")
                statuses[name] = status
                files = (
                    self.run.metadata["graph_environment"]["files"]
                    if name == "EnvironmentProof" else
                    [_file(path) for path in sorted(root.rglob("*")) if path.is_file()]
                )
                inventory[name] = {"root": str(root), "files": files}
            if Path(facts["EnvironmentProof"]).resolve() != Path(
                self.run.environment["environment_proof"]["artifact_root"]
            ).resolve():
                raise ValueError("graph EnvironmentProof does not match bound proof")
            required_states = {
                "DeploymentProof": "DEPLOYMENT_READY", "AccuracyDifferential": "ACCURACY_PASS",
                "ServingBaseline": "BASELINE_FROZEN", "OperatorIntegration": "OPERATORS_READY",
                "TorchShimRegistry": "HANDOFF_CLEAR",
            }
            for name, expected in required_states.items():
                if statuses[name].get("state") != expected:
                    raise ValueError(f"{name} must be {expected}")
            inventory["DeploymentProof"]["scheduler_snapshot"] = self._service_snapshot(
                Path(facts["DeploymentProof"]), gate,
            )
            inventory["AccuracyDifferential"]["scheduler_snapshot"] = self._accuracy_snapshot(
                Path(facts["AccuracyDifferential"]), Path(facts["DeploymentProof"]), gate,
            )
            dispatch = self.run.metadata["graph_discovery"]
            if statuses["OperatorTaskDispatch"] != _load(Path(dispatch["status_path"])):
                raise ValueError("graph dispatch does not match accepted scheduler discovery")
            errors = validate_integration(statuses["OperatorIntegration"])
            if errors:
                raise ValueError("; ".join(errors))
            if Path(statuses["OperatorIntegration"]["baseline_path"]).resolve() != (
                Path(facts["ServingBaseline"]) / "baseline_manifest.json"
            ).resolve():
                raise ValueError("integration does not reference the graph's frozen baseline")
            baseline = self.validate_baseline(Path(facts["ServingBaseline"]) / "baseline_manifest.json")
            # The frozen pre-integration baseline deliberately remains immutable
            # when fresh service/accuracy regressions produce newer graph facts.
            for field, fact, filename in (
                ("service", "DeploymentProof", "status.json"),
                ("accuracy", "AccuracyDifferential", "accuracy_differential.json"),
            ):
                path = Path(facts[fact]) / filename
                report = _load(path)
                self._check_identity(report)
                if field == "service":
                    errors = _deployment_errors(report)
                    if errors:
                        raise ValueError("; ".join(errors))
                    for filename in report.get("artifacts", []):
                        _file(ArtifactStore(path.parent).path(filename))
                else:
                    if report.get("status") != "ACCURACY_PASS":
                        raise ValueError("current accuracy report must pass")
                    if report.get("revision") not in (None, self.run.model_revision):
                        raise ValueError("current accuracy model revision does not match run")
                    errors = _accuracy_errors(report)
                if errors:
                    raise ValueError("; ".join(errors))
                inventory["ServingBaseline"][f"{field}_source"] = _file(
                    Path(baseline[field]["artifact"]),
                )
            shim = _load(Path(facts["TorchShimRegistry"]) / "torch_shim_registry.json")
            for field in ("signals", "entries", "unmapped_signals"):
                if not isinstance(shim.get(field), list) or not all(
                    isinstance(entry, dict) for entry in shim[field]
                ):
                    raise ValueError(f"shim registry {field} must be a list of objects")
            errors = validate_shim_handoff(shim, {})
            if errors:
                raise ValueError("; ".join(errors))
            names = {self.scheduler.store.operator(self.run_id, key).operator_id
                     for key in gate["operator_keys"]}
            for entry in shim.get("entries", []):
                if entry.get("status") != "WAIVED" and _identifier(entry) not in names:
                    raise ValueError("non-waived shim is not integrated by the scheduler")
            final_gate = self.delivery_status()
            if final_gate["state"] != "OPERATORS_READY":
                raise ValueError("operator evidence changed during delivery")
            if self._service_snapshot(Path(facts["DeploymentProof"]), final_gate) != (
                inventory["DeploymentProof"]["scheduler_snapshot"]
            ):
                raise ValueError("service scheduler snapshot changed during delivery")
            if self._accuracy_snapshot(
                Path(facts["AccuracyDifferential"]), Path(facts["DeploymentProof"]), final_gate,
            ) != inventory["AccuracyDifferential"]["scheduler_snapshot"]:
                raise ValueError("accuracy scheduler snapshot changed during delivery")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"model delivery blocked: {exc}") from exc
        verdict = "SIMULATION_PASS" if self.evidence_mode == "simulation" else "FUNCTIONAL_READY"
        attempt = RunPaths(self.artifact_root, self.run_id).allocate_attempt("model-adaptation-delivery")
        receipt = {
            "schema_version": 1, "state": verdict, "verdict": verdict,
            "run_id": self.run_id, "subject": self.subject, "evidence_mode": self.evidence_mode,
            "model_revision": self.run.model_revision, "plugin_revision": self.run.plugin_revision,
            "backend": self.run.backend,
            "environment_fingerprint": self.run.environment["environment_proof"]["fingerprint"],
            "task_ids": gate["task_ids"], "operator_keys": gate["operator_keys"],
            "task_results": gate["task_results"], "snapshot_token": gate["snapshot_token"],
            "facts": inventory,
        }
        path = ArtifactStore(attempt.output).write_json("delivery_receipt.json", receipt)
        manifest = ArtifactStore(attempt.root).register(
            identity={**attempt.identity, "evidence_mode": self.evidence_mode},
            outcome=verdict, required=["output/delivery_receipt.json"],
        )
        result = {**receipt, "receipt_path": str(path), "manifest_path": str(manifest),
                  "receipt_sha256": _file(path)["sha256"], "manifest_sha256": _file(manifest)["sha256"]}
        self.scheduler.record_graph_transition(self.run_id, "graph_delivery", result)
        return result
