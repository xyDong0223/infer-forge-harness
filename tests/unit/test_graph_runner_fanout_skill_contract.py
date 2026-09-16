import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import graph_runner  # noqa: E402
import cli.workflow.graph as _graph_runner_cli  # noqa: E402


def test_graph_execute_passes_skill_contract_to_fanout_discovery(tmp_path, monkeypatch):
    workflow = [{"id": "fanout", "task": "fixture", "on_success": "DELIVERED",
                 "on_failure": "REWORK"}]
    spec = {
        "produces": "CapabilityEvaluation",
        "command": ["child", "--out", "{artifacts}"],
        "fan_out": {"list": ["list"], "var": "dimension",
                    "aggregate": ["aggregate", "--out", "{artifacts}"]},
        "state_file": "evaluation_status.json",
    }
    monkeypatch.setattr(graph_runner, "load_workflow", lambda _: workflow)
    monkeypatch.setattr(graph_runner, "node_task_type", lambda _: "fixture_fanout")
    monkeypatch.setitem(graph_runner.NODES, "fixture_fanout", spec)
    monkeypatch.setattr(graph_runner.skill_registry, "resolve_for_context",
                        lambda *_: {"id": "fixture", "verification": [], "tools": []})
    observed = {}

    def list_dimensions(command, *, cwd, text, capture_output, env):
        observed["env"] = env
        return SimpleNamespace(returncode=0, stdout='["quantization"]', stderr="")

    def run(command, *, log_path, **kwargs):
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        if command[0] == "aggregate":
            (out / "evaluation_status.json").write_text(
                json.dumps({"state": "EVALUATION_PASS"}), encoding="utf-8"
            )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fixture console", encoding="utf-8")
        return SimpleNamespace(returncode=0, crash_log=None)

    monkeypatch.setattr(graph_runner.subprocess, "run", list_dimensions)
    monkeypatch.setattr(graph_runner.evidence, "run_logged", run)
    monkeypatch.setattr(sys, "argv", [
        "graph", "--subject", "demo", "--artifact-root", str(tmp_path / "run"),
        "--watch-interval", "0", "--json", "--execute",
    ])
    assert _graph_runner_cli.main() == 0
    packet_path = Path(observed["env"]["INFER_FORGE_SKILL_CONTRACT"])
    assert packet_path.is_file()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["id"] == "fixture"
    assert packet["task_type"] == "fixture_fanout"
