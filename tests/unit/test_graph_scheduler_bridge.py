import hashlib
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from types import SimpleNamespace

import pytest

from core.storage import RunPaths
from engine.graph_bridge import FACT_STATUS, GraphSchedulerBridge
from engine.scheduler import TaskScheduler
from operations.operators.operator_lifecycle import freeze_baseline, integration_decision
from tests.scheduler_helpers import simulation_result
from validators.deployment_validator import ENVIRONMENT_ARTIFACTS
from validators.operator_lifecycle_validator import validate_dispatch, validate_integration


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def proof(root, mode="simulation", **changes):
    for name in ENVIRONMENT_ARTIFACTS:
        write(root / name, {"fixture": name})
    data = {
        "state": "ENVIRONMENT_READY", "pod": "test-pod", "evidence_mode": mode,
        "checks": {**{name: True for name in (
            "pod_ready", "runtime_importable", "code_ready", "device_ready",
            "base_model_loaded", "base_prefill", "base_decode",
        )}, "base_health_check": 200, "base_chat_completion": "non_empty",
                   "unexpected_fallback": False},
        "artifacts": list(ENVIRONMENT_ARTIFACTS), **changes,
    }
    write(root / "status.json", data)
    return data


@pytest.fixture
def bridge(tmp_path):
    root = tmp_path / "run"
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(
        run_id="r", model_id="m", model_revision="1", plugin_revision="p", backend="kunlun",
        metadata={"evidence_mode": "simulation", "environment_required": False,
                  "artifact_root": str(root)},
    )
    scheduler.store.close()
    result = GraphSchedulerBridge(tmp_path / "state.db", "r", "m", root, {})
    environment = RunPaths(root, "r").allocate_attempt("environment").output
    proof(environment)
    result.bind_environment(environment)
    yield result
    result.close()


def entry(name="scale"):
    return {
        "name": name,
        "inputs": [{"name": "x", "dtype": "float32", "shape": [4], "layout": "contiguous"}],
        "outputs": [{"name": "y", "dtype": "float32", "shape": [4], "layout": "contiguous"}],
        "semantics": {"formula": "y = 2 * x", "reference": name},
    }


def dispatch(bridge, entries, gaps=None):
    attempt = RunPaths(bridge.artifact_root, "r").allocate_attempt("dispatch")
    source = write(attempt.input / "gaps.json", {"gaps": entries if gaps is None else gaps})
    report = write(attempt.input / "operators.json", {"entries": entries})
    return bridge.dispatch(source, attempt.output, report)


def complete(bridge, stage):
    task = bridge.scheduler.claim_ready("worker", stage=stage)[0]
    result = simulation_result(task, bridge.artifact_root, run=bridge.run)
    done = bridge.scheduler.complete(
        task.task_id, worker_id="worker", lease_token=task.lease_token, result=result,
    )
    assert done.status == "succeeded"
    return done


def test_plan_is_readonly_and_unknown_run_does_not_create_database(tmp_path, bridge):
    unknown = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="existing state"):
        GraphSchedulerBridge(unknown, "r", "m", tmp_path / "root", {}, execute=False)
    assert not unknown.exists()
    before = len(bridge.scheduler.store.events("r"))
    plan = GraphSchedulerBridge(bridge.state, "r", "m", bridge.artifact_root, {}, execute=False)
    with pytest.raises(ValueError, match="read-only"):
        plan.bind_environment(Path(bridge.run.environment["environment_proof"]["artifact_root"]))
    plan.close()
    assert len(bridge.scheduler.store.events("r")) == before
    with pytest.raises(ValueError, match="unknown adaptation"):
        GraphSchedulerBridge(bridge.state, "missing", "m", bridge.artifact_root, {})


@pytest.mark.parametrize("subject,environment", [
    ("other", {}), ("m", {"backend": "cuda"}), ("m", {"model_revision": "2"}),
    ("m", {"plugin_revision": "q"}), ("m", {"evidence_mode": "real"}),
])
def test_constructor_identity(bridge, subject, environment):
    with pytest.raises(ValueError, match="match"):
        GraphSchedulerBridge(bridge.state, "r", subject, bridge.artifact_root, environment)


def test_bind_rejection_keeps_old_fingerprint_and_blocks_claims(bridge):
    dispatch(bridge, [entry()])
    old = bridge.run.environment["environment_proof"]["fingerprint"]
    root = RunPaths(bridge.artifact_root, "r").allocate_attempt("bad-environment").output
    proof(root, mode="real")
    with pytest.raises(ValueError, match="evidence_mode"):
        bridge.bind_environment(root)
    assert bridge.run.status == "ENVIRONMENT_FAILED"
    assert bridge.run.environment["environment_proof"]["fingerprint"] == old
    assert bridge.scheduler.claim_ready("worker") == []
    assert bridge.delivery_status()["state"] == "OPERATORS_BLOCKED"


def test_real_scheduler_rejects_simulated_environment(tmp_path):
    scheduler = TaskScheduler(tmp_path / "real.db")
    scheduler.create_run(run_id="r", model_id="m")
    data = proof(tmp_path / "proof")
    with pytest.raises(ValueError, match="simulation"):
        scheduler.bind_environment("r", data, tmp_path / "proof")
    scheduler.store.close()


def test_environment_idempotency_and_tamper_detection(bridge):
    root = Path(bridge.run.environment["environment_proof"]["artifact_root"])
    before = len(bridge.scheduler.store.events("r"))
    bridge.bind_environment(root)
    assert len(bridge.scheduler.store.events("r")) == before
    (root / "runtime_import.txt").write_text("corrupted")
    assert bridge.delivery_status()["state"] == "OPERATORS_BLOCKED"
    assert bridge.run.status == "ENVIRONMENT_FAILED"
    with pytest.raises(ValueError, match="changed"):
        bridge.bind_environment(root)


@pytest.mark.parametrize("changes", [
    {"run_id": "other"}, {"evidence_mode": "real"},
    {"checks": []}, {"target": {"runtime": {"backend": "cuda"}}},
])
def test_rejected_proof_identity_is_recorded(bridge, changes):
    root = RunPaths(bridge.artifact_root, "r").allocate_attempt("rejected").output
    proof(root, **changes)
    with pytest.raises(ValueError, match="rejected"):
        bridge.bind_environment(root)
    assert bridge.run.status == "ENVIRONMENT_FAILED"
    assert bridge.scheduler.store.events("r")[-1].event_type == "environment_failed"


def test_initial_graph_binding_rejects_corrupted_previously_imported_proof(tmp_path):
    root = tmp_path / "run"
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(
        run_id="r", model_id="m", metadata={
            "evidence_mode": "simulation", "artifact_root": str(root),
        },
    )
    artifacts = RunPaths(root, "r").allocate_attempt("environment").output
    scheduler.bind_environment("r", proof(artifacts), artifacts)
    scheduler.store.close()
    (artifacts / "runtime_import.txt").write_text("corrupted before bridge activation")
    bridge = GraphSchedulerBridge(tmp_path / "state.db", "r", "m", root, {})
    try:
        with pytest.raises(ValueError, match="changed"):
            bridge.bind_environment(artifacts)
        assert bridge.run.status == "ENVIRONMENT_FAILED"
    finally:
        bridge.close()


def test_external_preaccepted_environment_can_be_imported_without_mutation(tmp_path):
    run_root = tmp_path / "run"
    state = tmp_path / "state.db"
    source = RunPaths(tmp_path / "proof-workflow", "proof-workflow").allocate_attempt("proof").output
    status = proof(source, workspace_identity={
        "run_id": "proof-workflow", "task_id": "proof", "attempt_id": "000001",
    })
    scheduler = TaskScheduler(state)
    scheduler.create_run(
        run_id="r", model_id="m", model_revision="1", plugin_revision="p", backend="kunlun",
        metadata={"evidence_mode": "simulation", "artifact_root": str(run_root)},
    )
    scheduler.bind_environment("r", status, source)
    scheduler.store.close()
    before = {path: path.read_bytes() for path in source.iterdir()}
    bridge = GraphSchedulerBridge(state, "r", "m", run_root, {})
    try:
        bridge.bind_environment(source)
        assert bridge.run.metadata["graph_environment"]["imported"] is True
        assert before == {path: path.read_bytes() for path in source.iterdir()}
        (source / "journal.jsonl").write_text("mutable workflow journal")
        facts = facts_for_delivery(bridge)
        result = bridge.finalize(facts)
        assert result["state"] == "SIMULATION_PASS"
        assert all(not item["path"].endswith("journal.jsonl")
                   for item in result["facts"]["EnvironmentProof"]["files"])
        (source / "journal.jsonl").write_text("workflow journal advanced")
        assert bridge.delivery_status()["state"] == "OPERATORS_READY"
        status["pod"] = "corrupted-pod"
        write(source / "status.json", status)
        with pytest.raises(ValueError, match="changed"):
            bridge.bind_environment(source)
        assert bridge.run.status == "ENVIRONMENT_FAILED"
    finally:
        bridge.close()


def test_external_environment_requires_preexisting_exact_binding(bridge, tmp_path):
    external = tmp_path / "unaccepted-proof"
    proof(external)
    with pytest.raises(ValueError, match="exact scheduler binding"):
        bridge.bind_environment(external)
    assert bridge.run.status == "ENVIRONMENT_FAILED"


def test_no_discovery_is_waiting_but_explicit_zero_gap_is_ready(bridge):
    assert bridge.delivery_status()["state"] == "WAITING_FOR_OPERATORS"
    assert dispatch(bridge, [])["state"] == "DISPATCH_SKIPPED"
    assert bridge.delivery_status()["state"] == "OPERATORS_READY"


def test_partial_discovery_persists_valid_operator_and_blocks_until_repaired(bridge):
    incomplete = {"name": "other"}
    status = dispatch(bridge, [entry(), incomplete])
    assert status["state"] == "DISPATCH_BLOCKED"
    assert len(status["task_ids"]) == 1
    complete(bridge, "torch")
    assert bridge.delivery_status()["state"] == "OPERATORS_BLOCKED"
    repaired = dispatch(bridge, [entry(), entry("other")])
    assert repaired["state"] == "DISPATCHED"
    assert len(bridge.scheduler.store.tasks("r")) == 3
    assert bridge.delivery_status()["state"] == "WAITING_FOR_OPERATORS"
    events = [event for event in bridge.scheduler.store.events("r")
              if event.event_type == "graph_discovery"]
    assert [event.payload["state"] for event in events] == ["DISPATCH_BLOCKED", "DISPATCHED"]


@pytest.mark.parametrize("entries", [[], [entry("unrelated")]])
def test_supplemental_report_cannot_hide_actionable_gaps(bridge, entries):
    status = dispatch(bridge, entries, [{"class": "CAPABILITY_MISSING", "axis": "scale"}])
    assert status["state"] == "DISPATCH_BLOCKED"
    assert "scale" in " ".join(status["errors"])


def test_late_shims_are_additive_and_cannot_clear_primary_blockers(bridge):
    dispatch(bridge, [], [{"operator": "scale"}])
    root = RunPaths(bridge.artifact_root, "r").allocate_attempt("shims").output
    write(root / "torch_shim_registry.json", {"entries": []})
    assert bridge.dispatch_shims(root)["state"] == "DISPATCH_SKIPPED"
    assert bridge.delivery_status()["state"] == "OPERATORS_BLOCKED"
    dispatch(bridge, [])
    write(root / "torch_shim_registry.json", {"entries": [{"name": "late-shim"}]})
    assert bridge.dispatch_shims(root)["state"] == "DISPATCH_BLOCKED"
    assert bridge.delivery_status()["state"] == "OPERATORS_BLOCKED"
    write(root / "torch_shim_registry.json", {"entries": [entry("late-shim")]})
    assert bridge.dispatch_shims(root)["state"] == "DISPATCHED"
    assert bridge.delivery_status()["state"] == "WAITING_FOR_OPERATORS"
    assert bridge.run.metadata["graph_discovery"]["state"] == "DISPATCH_SKIPPED"


def test_shim_dispatch_preserves_finalized_graph_artifacts(bridge):
    root = RunPaths(bridge.artifact_root, "r").allocate_attempt("shims").output
    write(root / "torch_shim_registry.json", {"entries": [entry(), {"name": "incomplete"}]})
    write(root / "dispatch_status.json", {"state": "DISPATCHED", "requests": ["legacy-file-only"]})
    write(root.parent / "manifest.json", {"fixture": "already finalized"})
    before = {path: path.read_bytes() for path in root.parent.rglob("*") if path.is_file()}
    status = bridge.dispatch_shims(root)
    assert status["state"] == "DISPATCH_BLOCKED"
    assert len(status["task_ids"]) == 1
    assert before == {path: path.read_bytes() for path in root.parent.rglob("*") if path.is_file()}
    receipt = Path(bridge.run.metadata["graph_shim_discovery"]["status_path"])
    assert receipt.is_file()
    assert not receipt.is_relative_to(root.parent)


def test_stages_are_revalidated_with_current_submission(bridge):
    dispatch(bridge, [entry()])
    assert bridge.delivery_status()["state"] == "WAITING_FOR_OPERATORS"
    for stage in ("torch", "xpu", "integration"):
        done = complete(bridge, stage)
    assert bridge.delivery_status()["state"] == "OPERATORS_READY"
    Path(done.output["evidence"]["service_regression"]).write_text("tampered")
    status = bridge.delivery_status()
    assert status["state"] == "OPERATORS_BLOCKED"
    assert any("hash mismatch" in error for error in status["errors"])


def test_report_tampering_blocks_delivery_even_after_stages_pass(bridge):
    dispatch(bridge, [])
    report = Path(bridge.run.metadata["graph_discovery"]["reports"][0]["path"])
    report.write_text('{"gaps": [{"operator": "new_missing_operator"}]}')
    assert bridge.delivery_status()["state"] == "OPERATORS_BLOCKED"


def test_failed_stage_and_pending_diagnosis_cannot_deliver(bridge):
    dispatch(bridge, [entry()])
    task = bridge.scheduler.claim_ready("worker")[0]
    bridge.scheduler.fail(task.task_id, worker_id="worker", lease_token=task.lease_token, error="bad")
    status = bridge.delivery_status()
    assert status["state"] == "OPERATORS_BLOCKED"
    assert any("diagnosis" in error for error in status["errors"])


def test_other_runs_cannot_satisfy_delivery(bridge):
    bridge.scheduler.create_run(
        run_id="other", model_id="m", metadata={"evidence_mode": "simulation"},
    )
    from engine.discovery import operator_specs_from_report
    spec = operator_specs_from_report({"entries": [entry()]}, model_id="m", backend="kunlun")[0]
    bridge.scheduler.discover_operator("other", spec)
    for stage in ("torch", "xpu", "integration"):
        task = bridge.scheduler.claim_ready("worker", stage=stage)[0]
        bridge.scheduler.complete(
            task.task_id, worker_id="worker", lease_token=task.lease_token,
            result=simulation_result(task, bridge.artifact_root),
        )
    assert bridge.delivery_status()["task_ids"] == []
    assert bridge.delivery_status()["state"] == "WAITING_FOR_OPERATORS"


def facts_for_delivery(bridge):
    facts = {
        name: RunPaths(bridge.artifact_root, "r").allocate_attempt(name).output
        for name in FACT_STATUS
    }
    for name, root in facts.items():
        write(root / FACT_STATUS[name], {"state": "PASS", "run_id": "r", "evidence_mode": "simulation"})
    facts["EnvironmentProof"] = Path(bridge.run.environment["environment_proof"]["artifact_root"])
    dispatch(bridge, [])
    facts["OperatorTaskDispatch"] = Path(bridge.run.metadata["graph_discovery"]["status_path"]).parent
    service = facts["DeploymentProof"]
    write(service.parent / "input" / "scheduler_snapshot.json", bridge.delivery_status())
    write(service / "service.json", {"health": 200, "answer": "fixture only"})
    write(service / "status.json", {
        "state": "DEPLOYMENT_READY", "pod": "test-pod",
        "checks": {"pod_ready": True, "health_check": 200, "chat_completion": "non_empty",
                   "expected_backend": "kunlun", "unexpected_fallback": False},
        "artifacts": ["service.json"],
    })
    accuracy = facts["AccuracyDifferential"]
    write(accuracy / "accuracy_status.json", {"state": "ACCURACY_PASS"})
    write(accuracy / "accuracy_differential.json", {
        "status": "ACCURACY_PASS", "subject": "m", "threshold_source": "unit fixture",
        "reference": {"implementation": "unit fixture", "device": "cpu"},
        "candidate": {"device": "simulated-xpu"}, "metric": "top-1",
        "cases": [{"top1_candidate": 1, "top1_reference": 1, "candidate_top5": [1],
                   "reference_top5": [1], "top1_match": True}],
    })
    stamp_accuracy(bridge, accuracy, service)
    freeze_baseline(
        service / "status.json", accuracy / "accuracy_differential.json",
        facts["ServingBaseline"], "m", {},
    )
    integration_decision(
        facts["ServingBaseline"] / "baseline_manifest.json", None,
        facts["OperatorIntegration"], "m", scheduler_state=bridge.state, run_id="r",
    )
    shim = facts["TorchShimRegistry"]
    write(shim / "shim_status.json", {"state": "HANDOFF_CLEAR"})
    write(shim / "torch_shim_registry.json", {
        "state": "HANDOFF_CLEAR", "plugin": "fixture", "files_scanned": 1,
        "signals": [], "entries": [], "unmapped_signals": [],
    })
    return facts


def stamp_accuracy(bridge, accuracy, service):
    status = service / "status.json"
    write(accuracy.parent / "input/scheduler_snapshot.json", {
        **bridge.delivery_status(),
        "service_proof": {"path": str(status.resolve()),
                          "sha256": hashlib.sha256(status.read_bytes()).hexdigest()},
    })


def test_final_receipt_preserves_environment_semantics_and_binds_files(bridge):
    facts = facts_for_delivery(bridge)
    result = bridge.finalize(facts)
    assert result["state"] == "SIMULATION_PASS"
    assert result["verdict"] != "FUNCTIONAL_READY"
    assert bridge.run.status == "ENVIRONMENT_READY"
    assert bridge.delivery_status()["state"] == "OPERATORS_READY"
    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert receipt["facts"]["DeploymentProof"]["files"]
    assert result["receipt_sha256"] == hashlib.sha256(Path(result["receipt_path"]).read_bytes()).hexdigest()
    assert Path(result["manifest_path"]).is_file()
    assert bridge.run.metadata["graph_delivery"]["receipt_path"] == result["receipt_path"]
    assert bridge.finalize(facts)["receipt_path"] != result["receipt_path"]
    (facts["DeploymentProof"] / "service.json").unlink()
    with pytest.raises(ValueError, match="regular"):
        bridge.finalize(facts)
    assert bridge.run.metadata["graph_delivery"]["state"] == "OPERATORS_BLOCKED"


def test_fresh_regression_facts_do_not_replace_frozen_baseline(bridge):
    facts = facts_for_delivery(bridge)
    original = facts["DeploymentProof"]
    fresh = RunPaths(bridge.artifact_root, "r").allocate_attempt("fresh-service").output
    for filename in ("status.json", "service.json"):
        (fresh / filename).write_bytes((original / filename).read_bytes())
    write(fresh.parent / "input" / "scheduler_snapshot.json", bridge.delivery_status())
    facts["DeploymentProof"] = fresh
    old_accuracy = facts["AccuracyDifferential"]
    accuracy = RunPaths(bridge.artifact_root, "r").allocate_attempt("fresh-accuracy").output
    for filename in ("accuracy_status.json", "accuracy_differential.json"):
        (accuracy / filename).write_bytes((old_accuracy / filename).read_bytes())
    stamp_accuracy(bridge, accuracy, fresh)
    facts["AccuracyDifferential"] = accuracy
    result = bridge.finalize(facts)
    assert result["facts"]["ServingBaseline"]["service_source"]["path"] == str(original / "status.json")
    assert result["facts"]["DeploymentProof"]["root"] == str(fresh)
    (fresh / "service.json").unlink()
    with pytest.raises(ValueError, match="regular"):
        bridge.finalize(facts)


def test_delivery_snapshot_is_deterministic_and_binds_attempt_evidence(bridge):
    dispatch(bridge, [entry()])
    before = bridge.delivery_status()
    assert before["snapshot_token"] == bridge.delivery_status()["snapshot_token"]
    task = complete(bridge, "torch")
    after = bridge.delivery_status()
    assert after["snapshot_token"] != before["snapshot_token"]
    result = next(item for item in after["task_results"] if item["task_id"] == task.task_id)
    assert result["attempt"] == task.attempt
    assert result["evidence_sha256"] == task.output["evidence_sha256"]
    assert result["submission"] == task.output["_submission"]


@pytest.mark.parametrize("capture_waiting", [False, True])
def test_pre_operator_service_snapshot_cannot_deliver(bridge, capture_waiting):
    facts = facts_for_delivery(bridge)
    dispatch(bridge, [entry()])
    facts["OperatorTaskDispatch"] = Path(bridge.run.metadata["graph_discovery"]["status_path"]).parent
    if capture_waiting:
        write(facts["DeploymentProof"].parent / "input" / "scheduler_snapshot.json",
              bridge.delivery_status())
    for stage in ("torch", "xpu", "integration"):
        complete(bridge, stage)
    integration_decision(
        facts["ServingBaseline"] / "baseline_manifest.json", None,
        facts["OperatorIntegration"], "m", scheduler_state=bridge.state, run_id="r",
    )
    with pytest.raises(ValueError, match="snapshot|must start after"):
        bridge.finalize(facts)
    assert bridge.run.metadata["graph_delivery"]["state"] == "OPERATORS_BLOCKED"


def test_final_service_requires_persisted_scheduler_snapshot(bridge):
    facts = facts_for_delivery(bridge)
    (facts["DeploymentProof"].parent / "input" / "scheduler_snapshot.json").unlink()
    with pytest.raises(ValueError, match="scheduler_snapshot.json"):
        bridge.finalize(facts)


@pytest.mark.parametrize("fault", ["missing", "waiting", "other-service"])
def test_final_accuracy_requires_matching_service_and_ready_snapshot(bridge, fault):
    facts = facts_for_delivery(bridge)
    snapshot = facts["AccuracyDifferential"].parent / "input/scheduler_snapshot.json"
    if fault == "missing":
        snapshot.unlink()
    else:
        payload = json.loads(snapshot.read_text())
        if fault == "waiting":
            payload["state"] = "WAITING_FOR_OPERATORS"
        else:
            payload["service_proof"]["path"] = "/external/other-service/status.json"
        write(snapshot, payload)
    with pytest.raises(ValueError, match="scheduler_snapshot|final accuracy"):
        bridge.finalize(facts)
    assert bridge.run.metadata["graph_delivery"]["state"] == "OPERATORS_BLOCKED"


def test_discovery_cannot_overtake_delivery_publication(bridge, monkeypatch):
    from engine.discovery import operator_specs_from_report

    facts = facts_for_delivery(bridge)
    other = TaskScheduler(bridge.state)
    spec = operator_specs_from_report(
        {"entries": [entry("late")]}, model_id="m", backend="kunlun",
        model_revision="1", plugin_revision="p",
        environment=bridge.run.environment["environment_proof"],
    )[0]
    started = threading.Event()
    future = None
    pool = ThreadPoolExecutor(max_workers=1)

    def discover():
        started.set()
        return other.discover_operator("r", spec)

    allocate = RunPaths.allocate_attempt

    def interleave(paths, task_id):
        nonlocal future
        if task_id == "model-adaptation-delivery":
            future = pool.submit(discover)
            assert started.wait(2)
            done, _ = wait([future], timeout=0.1)
            assert not done, "discovery overtook an uncommitted delivery"
        return allocate(paths, task_id)

    monkeypatch.setattr(RunPaths, "allocate_attempt", interleave)
    try:
        with pool:
            result = bridge.finalize(facts)
            assert future is not None
            future.result(timeout=5)
    finally:
        other.store.close()
    events = bridge.scheduler.store.events("r")
    delivery_index = next(index for index, event in enumerate(events)
                          if event.event_type == "graph_delivery"
                          and event.payload.get("receipt_path") == result["receipt_path"])
    discovered_index = next(index for index, event in enumerate(events)
                            if event.event_type == "operator_discovered")
    assert delivery_index < discovered_index
    assert bridge.delivery_status()["state"] == "WAITING_FOR_OPERATORS"


@pytest.mark.parametrize("kind", ["DeploymentProof", "AccuracyDifferential"])
def test_recovery_captures_each_regression_attempt_before_execution(bridge, monkeypatch, kind):
    from runners import graph_runner

    facts = facts_for_delivery(bridge)
    paths = RunPaths(bridge.artifact_root, "r")
    journal = paths.journal
    environment = {"hardware": "P800"}
    graph_runner.record_fact(
        journal, graph_runner.NODES["service_proof"], "m",
        facts["DeploymentProof"], environment,
    )
    failed = paths.allocate_attempt("failed-regression").output
    write(failed / FACT_STATUS[kind], {"state": "READINESS_TIMEOUT", "reason": "readiness timeout"})
    spec = {"produces": kind, "state_file": FACT_STATUS[kind],
            "command": ["unit-regression-boundary", "{artifacts}"]}
    observed = []

    def execute(command, *, log_path, **kwargs):
        out = Path(command[1])
        snapshot = out.parent / "input/scheduler_snapshot.json"
        assert json.loads(snapshot.read_text())["snapshot_token"] == bridge.delivery_status()["snapshot_token"]
        observed.append(snapshot)
        for source in facts[kind].iterdir():
            if source.is_file():
                (out / source.name).write_bytes(source.read_bytes())
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("unit external boundary")
        return SimpleNamespace(returncode=0, crash_log=None)

    monkeypatch.setattr(graph_runner.evidence, "run_logged", execute)
    args = SimpleNamespace(
        run_paths=paths, artifact_root=paths.root, subject="m", journal=journal,
        brain="rule", decide_command=None, decide_timeout=1, recovery_budget=1,
        watch_interval=0,
    )
    outcome = graph_runner.attempt_recovery(
        args, node="regression", spec=spec, context={"subject": "m"},
        artifacts=failed, environment=environment, state="READINESS_TIMEOUT",
        skill={
            "id": "fixture",
            "task_type": "service_proof",
            "tools": [],
            "verification": "fixture-validator",
            "method": None,
        },
        bridge=bridge,
    )
    assert outcome.status == "RECOVERED"
    assert len(observed) == 1
    recovered = Path(outcome.final_artifacts)
    assert observed[0] == recovered.parent / "input/scheduler_snapshot.json"
    if kind == "AccuracyDifferential":
        bridge._accuracy_snapshot(recovered, facts["DeploymentProof"], bridge.delivery_status())
    else:
        bridge._service_snapshot(recovered, bridge.delivery_status())


def test_premature_finalize_persists_failure_and_rejects_foreign_fact(bridge, tmp_path):
    with pytest.raises(ValueError, match="missing graph fact"):
        bridge.finalize({})
    assert bridge.run.metadata["graph_delivery"]["state"] == "OPERATORS_BLOCKED"
    facts = facts_for_delivery(bridge)
    facts["DeploymentProof"] = tmp_path / "foreign"
    with pytest.raises(ValueError, match="outside"):
        bridge.finalize(facts)


def test_scheduled_integrate_waits_and_rejects_unfrozen_baseline(bridge):
    facts = facts_for_delivery(bridge)
    dispatch(bridge, [entry()])
    baseline = facts["ServingBaseline"] / "baseline_manifest.json"
    out = RunPaths(bridge.artifact_root, "r").allocate_attempt("integrate").output
    kwargs = {"scheduler_state": bridge.state, "run_id": "r"}
    assert integration_decision(baseline, None, out, "m", **kwargs)["state"] == "WAITING_FOR_OPERATORS"
    payload = json.loads(baseline.read_text())
    payload["status"] = "PENDING"
    write(baseline, payload)
    assert integration_decision(baseline, None, out, "m", **kwargs)["state"] == "OPERATORS_BLOCKED"


def test_scheduled_validators_reject_state_only_claims():
    assert validate_dispatch({"state": "DISPATCHED", "run_id": "r", "requests": ["file-only"]})
    assert validate_integration({"state": "OPERATORS_READY"})


def test_dispatch_validator_checks_persisted_task_identity_and_report_hashes(bridge):
    status = dispatch(bridge, [entry()])
    assert status["validator"] == {"passed": True, "errors": []}
    assert validate_dispatch(status) == []
    forged = {**status, "task_ids": ["other-run:operator:torch"], "requests": ["other-run:operator:torch"]}
    assert any("owned by this run" in error for error in validate_dispatch(forged))
    Path(status["reports"][0]["path"]).write_text('{"gaps":[]}')
    assert any("changed" in error for error in validate_dispatch(status))


def test_integration_validator_recomputes_live_waiting_and_ready_gates(bridge):
    facts = facts_for_delivery(bridge)
    original = json.loads((facts["OperatorIntegration"] / "integration_status.json").read_text())
    assert original["validator"] == {"passed": True, "errors": []}
    assert validate_integration(original) == []
    dispatch(bridge, [entry()])
    assert any("live scheduler gate" in error for error in validate_integration(original))
    status = integration_decision(
        facts["ServingBaseline"] / "baseline_manifest.json", None,
        facts["OperatorIntegration"], "m", scheduler_state=bridge.state, run_id="r",
    )
    assert status["state"] == "WAITING_FOR_OPERATORS"
    assert status["validator"] == {"passed": True, "errors": []}
    assert validate_integration({**status, "state": "OPERATORS_READY"})


def test_dispatch_executes_validator_before_accepting_report(bridge, monkeypatch):
    monkeypatch.setattr("engine.graph_bridge.validate_dispatch", lambda report: ["independent rejection"])
    status = dispatch(bridge, [])
    assert status["state"] == "DISPATCH_BLOCKED"
    assert status["validator"]["passed"] is False
    assert bridge.run.metadata["graph_discovery"]["state"] == "DISPATCH_BLOCKED"


def test_integration_executes_validator_before_publishing_readiness(bridge, monkeypatch):
    facts = facts_for_delivery(bridge)
    monkeypatch.setattr(
        "operations.operators.operator_lifecycle.validate_integration",
        lambda report: ["independent rejection"],
    )
    status = integration_decision(
        facts["ServingBaseline"] / "baseline_manifest.json", None,
        facts["OperatorIntegration"], "m", scheduler_state=bridge.state, run_id="r",
    )
    assert status["state"] == "OPERATORS_BLOCKED"
    assert status["validator"]["passed"] is False
