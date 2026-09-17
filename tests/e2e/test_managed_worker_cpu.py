"""Production CPU probe integration, not a scheduler/hardware completion claim.

Real CPU cases require optional torch. The real-run environment rejection is
mandatory without torch; no prepared-environment report is synthesized here.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from core.paths import REPO_ROOT
from core.storage import ArtifactStore
from operations.validation.managed_worker import grade_observations, validation_plan
from tests.e2e.test_model_adaptation import source_snapshot


pytestmark = pytest.mark.local_e2e


class CpuScenario:
    def __init__(self, root):
        self.root = root
        self.store = ArtifactStore(root)
        self.state = root / "state.sqlite"
        self.run_id = "real-cpu-environment-gate"
        self.step = 0
        self.env = {key: value for key, value in os.environ.items()
                    if key != "PYTHONPATH" and not key.startswith("INFER_FORGE_E2E")}
        self.env.update(PYTHONDONTWRITEBYTECODE="1", INFER_FORGE_STATE_ROOT=str(root / "state"))

    def process(self, entry, arguments, expected=0):
        result = subprocess.run([sys.executable, str(REPO_ROOT / entry), *arguments],
                                cwd=self.root, env=self.env, capture_output=True,
                                text=True, check=False, timeout=45)
        self.step += 1
        log = self.store.write_text(f"driver/{self.step:03d}.log",
                                    f"exit={result.returncode}\n{result.stdout}\n{result.stderr}")
        assert result.returncode == expected, f"{log}\n{result.stdout}\n{result.stderr}"
        return result

    def cli(self, *arguments, expected=0):
        result = self.process("cli/adaptation.py", ["--state", str(self.state), *arguments], expected)
        return json.loads(result.stdout)


@pytest.fixture
def cpu_scenario(tmp_path, request):
    before = source_snapshot()
    case = CpuScenario(tmp_path)
    request.node.user_properties.append(("artifact_root", str(tmp_path)))
    yield case
    case.store.register(identity={"scenario": "cpu_probe_integration", "case": request.node.name},
                        outcome="RECORDED")
    assert source_snapshot() == before


def spec_and_candidate(case, geometry, *, wrong=False):
    reference = case.store.write_text("oracle/reference.py",
        "def run_case(inputs):\n    return {'y': (inputs['x'] + inputs['x']).contiguous()}\n")
    marker = case.root / "candidate-call-count.txt"
    source = (
        "from pathlib import Path\n"
        "def run_case(inputs):\n"
        f"    marker = Path({str(marker)!r})\n"
        "    count = int(marker.read_text()) if marker.exists() else 0\n"
        "    marker.write_text(str(count + 1))\n"
        f"    result = (inputs['x'] * {3 if wrong else 2}).contiguous()\n"
        "    inputs['x'].fill_(999)\n"
        "    return {'y': result}\n"
    )
    candidate = case.store.write_text("candidate/candidate.py", source)
    input_layout = "noncontiguous" if geometry == "strided" else "contiguous"
    dimensions = [2, 3] if geometry == "strided" else ["N", 3]
    tensor = {"shape": dimensions, "dtype": "float64", "layout": input_layout}
    output = {**tensor, "layout": "contiguous"}
    nonempty = {
        "id": "nonempty", "bindings": {"N": 2},
        "inputs": {"x": {**tensor, "shape": [2, 3],
                          "values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}},
        "outputs": {"y": {**output, "shape": [2, 3]}},
    }
    if geometry == "strided":
        nonempty["inputs"]["x"]["strides"] = [6, 2]
        cases = [nonempty]
    else:
        cases = [{"id": "empty", "bindings": {"N": 0},
                  "inputs": {"x": {**tensor, "shape": [0, 3], "values": []}},
                  "outputs": {"y": {**output, "shape": [0, 3]}}}, nonempty]
    contract = {
        "schema_version": 1, "candidate_entry": "candidate.py",
        "reference": {"entry": str(reference), "files": {
            str(reference): hashlib.sha256(reference.read_bytes()).hexdigest()},
            "provenance": "Independent addition oracle for the CPU probe fixture"},
        "cases": cases, "thresholds": {"y": {"max_relative_l2": 1e-8}},
        "negative_control": {"kind": "offset", "value": 1.0},
    }
    spec = {"inputs": [{"name": "x", **tensor}], "outputs": [{"name": "y", **output}],
            "semantics": {"formula": "y = 2 * x", "validation": contract}}
    return spec, candidate, marker


def execute_probes(case, geometry, *, wrong=False):
    spec, candidate, marker = spec_and_candidate(case, geometry, wrong=wrong)
    plan = validation_plan(spec, "torch")
    case.store.write_json("validation-plan.json", plan)
    raw, paths, requests = {}, {}, {}
    for role in ("candidate", "reference", "control"):
        request = {"schema_version": 1, "validation_id": "cpu-probe-only", "role": role,
                   "plan": plan, "evidence_mode": "real"}
        if role == "control":
            request["reference_observations"] = str(paths["reference"])
        else:
            entry = candidate if role == "candidate" else Path(plan["contract"]["reference"]["entry"])
            request["entry"] = {"path": str(entry), "sha256": hashlib.sha256(entry.read_bytes()).hexdigest()}
        request_path = case.store.write_json(f"requests/{role}.json", request)
        output = case.root / "observations" / f"{role}.json"
        result = case.process("cli/validation/worker_probe.py",
                              ["--request", str(request_path), "--output", str(output)])
        assert json.loads(result.stdout)["status"] == "MEASURED"
        raw[role], paths[role], requests[role] = json.loads(output.read_text()), output, request_path
    report = grade_observations(plan, **raw, evidence_mode="real", validation_id="cpu-probe-only")
    case.store.write_json("cpu-probe-grade.json", {
        "scope": "CPU_probe_integration_only", "scheduler_accepted": False,
        "hardware_ready": False, "grade": report,
    })
    return plan, raw, report, paths, requests, marker


@pytest.mark.parametrize("geometry", ["empty", "strided"])
def test_cpu_probe_cli_measures_geometry_and_rejects_reused_output(cpu_scenario, geometry):
    pytest.importorskip("torch", reason="actual CPU probe integration requires optional torch")
    case = cpu_scenario
    plan, raw, report, paths, requests, marker = execute_probes(case, geometry)
    assert report["verdict"] == "PASS"
    assert all(item["evidence_mode"] == "real" and item["stage"] == "torch" for item in raw.values())
    if geometry == "empty":
        assert raw["candidate"]["cases"][0]["outputs"]["y"]["shape"] == [0, 3]
        assert report["cases"][0]["metrics"]["y"]["numerical"] == "not_applicable_empty"
    else:
        for role in ("candidate", "reference"):
            observed = raw[role]["cases"][0]["input_metadata"]["x"]
            assert observed["strides"] == [6, 2] and observed["layout"] == "noncontiguous"
    calls = marker.read_text()
    assert int(calls) == len(plan["contract"]["cases"])
    observations_before = {role: path.read_bytes() for role, path in paths.items()}
    # The standalone probe is intentionally not a receipt/replay API. A new
    # process must reject an already-owned output without re-running the entry.
    rejected = case.process("cli/validation/worker_probe.py",
                            ["--request", str(requests["candidate"]),
                             "--output", str(paths["candidate"])], expected=6)
    assert "fresh output path" in rejected.stderr
    assert marker.read_text() == calls
    assert {role: path.read_bytes() for role, path in paths.items()} == observations_before
    assert not case.state.exists()


def test_cpu_probe_cli_wrong_numerics_are_not_accepted(cpu_scenario):
    pytest.importorskip("torch", reason="actual CPU probe integration requires optional torch")
    case = cpu_scenario
    _, _, report, _, _, _ = execute_probes(case, "empty", wrong=True)
    assert report["verdict"] == "FAIL"
    assert any(item["name"].endswith(":numerical") and item["passed"] is False
               for item in report["checks"])
    assert not case.state.exists()


def test_real_cpu_run_cannot_disable_the_environment_gate(cpu_scenario):
    case = cpu_scenario
    metadata = case.store.write_json("metadata.json", {"evidence_mode": "real", "environment_required": False})
    created = case.cli("create-run", "--run-id", case.run_id, "--model", "cpu-probe-only-model",
                       "--model-revision", "cpu-fixture-model-v1", "--plugin-revision", "cpu-fixture-plugin-v1",
                       "--backend", "cpu", "--metadata", str(metadata), "--worker-protocol", "managed-v2",
                       "--artifact-root", str(case.root / "run"))
    assert created["run"]["metadata"]["environment_required"] is True
    assert created["run"]["status"] == "WAITING_FOR_ENVIRONMENT"
    spec, _, marker = spec_and_candidate(case, "empty")
    gap = case.store.write_json("gap.json", {"entries": [{"operator_id": "scale", **spec,
        "evidence": {"mode": "real", "call_site": "cpu_probe_fixture.run_case"}}]})
    rejected = case.cli("discover", "--run-id", case.run_id, "--report", str(gap), expected=2)
    assert "environment proof is required" in rejected["error"]
    assert case.cli("claim", "--run-id", case.run_id, "--worker", "cpu-controller",
                    "--stage", "torch", "--limit", "1")["tasks"] == []
    status = case.cli("status", "--run-id", case.run_id, "--events")
    assert status["tasks"] == []
    assert not status["run"]["metadata"].get("execution_records")
    assert not marker.exists()
    # A second CLI process observes the same gate; no simulated proof is imported.
    replay = case.cli("status", "--run-id", case.run_id, "--events")
    for key in ("run", "tasks", "events"):
        assert replay[key] == status[key]
