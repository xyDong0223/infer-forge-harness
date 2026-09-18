"""Skeleton workflows must stay non-executable until they register a scenario."""

import yaml

from core.paths import REPO_ROOT
from runners.graph_runner import node_task_type


def _workflows():
    return sorted((REPO_ROOT / "workflows").glob("*.yaml"))


def test_registered_workflows_reference_existing_scenarios():
    """A workflow claiming a regression scenario must point at a real file."""
    for path in _workflows():
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenario = (workflow.get("metadata") or {}).get("regression_scenario")
        if scenario is not None:
            assert (REPO_ROOT / scenario).is_file(), f"{path.name}: missing {scenario}"


def test_unregistered_workflows_have_no_executable_nodes():
    """A workflow without metadata.regression_scenario is a skeleton. Every node
    must be PLANNED or validator-only so the Graph Runner stops with NO_CONTRACT
    instead of resolving a registered executor and touching external systems."""
    for path in _workflows():
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (workflow.get("metadata") or {}).get("regression_scenario"):
            continue
        nodes = workflow["spec"]["nodes"]
        assert nodes, f"{path.name}: empty skeleton"
        for node in nodes:
            assert node.get("task") in (None, "PLANNED"), (
                f"{path.name}:{node['id']} references {node.get('task')!r} without a "
                "registered regression_scenario; use task: PLANNED until the capability "
                "ships its mandatory local E2E scenario"
            )
            assert node_task_type(node) is None
