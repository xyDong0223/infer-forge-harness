"""Static discovery/configuration checks, not proof of an Agent's behavior."""

import re

import pytest
import yaml

from core.paths import REPO_ROOT


SKILL = REPO_ROOT / ".agents/skills/infer-forge-adaptation/SKILL.md"
ROLES = REPO_ROOT / ".codex/agents"


def test_adaptation_skill_is_discoverable_and_uses_existing_references():
    text = SKILL.read_text()
    frontmatter = text.split("---", 2)[1]
    metadata = yaml.safe_load(frontmatter)
    assert metadata["name"] == SKILL.parent.name
    assert isinstance(metadata["description"], str) and metadata["description"].strip()
    assert set(metadata) == {"name", "description"}

    references = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    assert references
    for reference in references:
        target = (SKILL.parent / reference).resolve()
        assert target.is_relative_to(REPO_ROOT)
        assert target.is_file(), reference


@pytest.mark.parametrize("role", ["implementer", "diagnoser", "validator"])
def test_project_agent_configs_inherit_user_execution_settings(role):
    tomllib = pytest.importorskip("tomllib", reason="TOML parser is in Python 3.11+")
    path = ROLES / f"infer-forge-{role}.toml"
    config = tomllib.loads(path.read_text())
    assert set(config) == {"name", "description", "developer_instructions"}
    assert config["name"] == path.stem
    for field in ("description", "developer_instructions"):
        assert isinstance(config[field], str) and config[field].strip()
