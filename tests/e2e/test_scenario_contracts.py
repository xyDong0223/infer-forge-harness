"""Every registered capability has a mandatory executable local scenario."""

import ast

import pytest
import yaml

from core.paths import REPO_ROOT


pytestmark = pytest.mark.local_e2e


def test_scenario_registry_is_nonempty_and_executable():
    scenarios = sorted((REPO_ROOT / "tests/e2e/scenarios").glob("*.yaml"))
    assert scenarios
    for path in scenarios:
        scenario = yaml.safe_load(path.read_text())
        assert scenario["schema_version"] == 1
        assert scenario["capability"] == path.stem
        assert (REPO_ROOT / scenario["workflow"]).is_file()
        assert (REPO_ROOT / scenario["entrypoint"]).is_file()
        workflow = yaml.safe_load((REPO_ROOT / scenario["workflow"]).read_text())
        assert workflow["spec"]["nodes"]
        assert workflow["metadata"]["regression_scenario"] == path.relative_to(REPO_ROOT).as_posix()
        local = scenario["tiers"]["local"]
        assert local["required"] is True
        assert local["evidence_mode"] == "simulation"
        assert local["cases"]
        module = ast.parse((REPO_ROOT / local["test"]).read_text())
        tests = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        assert set(local["cases"]) <= tests
        for tier in ("device_smoke", "real_model"):
            optional = scenario["tiers"][tier]
            assert optional["required"] is False
            assert optional["evidence_mode"] == "real"
            assert (REPO_ROOT / optional["test"]).is_file()
