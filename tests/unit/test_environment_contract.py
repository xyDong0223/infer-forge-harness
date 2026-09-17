"""Contract generation is driven by harness resources, not a model example."""

import hashlib
from pathlib import Path

import pytest
import yaml

from cli.adaptation import _environment_command, _parser
from core.paths import REPO_ROOT
from operations.deployment.environment_contract import build_environment_contract
from runners.graph_runner import NODES, Unresolved, resolve


def test_environment_is_generated_from_profile_not_target_settings():
    profile = yaml.safe_load((REPO_ROOT / "config/clusters/p800-cluster.yaml").read_text())
    previous = {"context": {"model": {"name": "AnotherModel", "path": "/wrong/weights"}},
                "execution": {"resource_name": "someone-else-target", "server_log": "/workspace/reproof.log",
                              "commands": {"serve": ["launch-the-wrong-model"]}}}
    contract = build_environment_contract("team-member", previous=previous)
    assert contract["context"]["model"]["path"] == profile["validation"]["base_model"]["path"]
    assert contract["execution"]["resource_name"] == "team-member-environment-base"
    assert contract["execution"]["server_log"] == "/workspace/reproof.log"
    assert "launch-the-wrong-model" not in str(contract)
    assert "AnotherModel" not in str(contract)
    for path, digest in contract["metadata"]["sources"].items():
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest


def test_adaptation_environment_needs_no_seed_yaml():
    args = _parser().parse_args(["environment", "--run-id", "example", "--user-id", "team-member"])
    assert args.contract is None
    command = _environment_command(None, Path("/external/run"), None, user_id=args.user_id)
    assert command[2] == "--execute"
    assert command[command.index("--user-id") + 1] == "team-member"


def test_service_cannot_bypass_missing_plan_with_a_manual_contract(tmp_path):
    with pytest.raises(Unresolved, match="MAT-005"):
        resolve(NODES["service_proof"], {
            "subject": "model", "contract_instance": "/external/handwritten.yaml",
            "pod": "team-member-pod", "artifacts": str(tmp_path),
        }, tmp_path / "journal.jsonl", {"hardware": "p800"})
