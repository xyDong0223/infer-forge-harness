"""User identities are supplied inputs, never derived from machine accounts."""

import pytest

from core.user_identity import resolve_user_id


def test_input_precedence_and_missing_identity(monkeypatch):
    monkeypatch.setenv("USER_ID", "legacy-owner")
    assert resolve_user_id(" team-member ", "contract-owner") == "team-member"
    assert resolve_user_id(recorded="contract-owner") == "contract-owner"
    assert resolve_user_id() == "legacy-owner"
    assert resolve_user_id(" ") == ""
    monkeypatch.delenv("USER_ID")
    monkeypatch.setenv("USER", "must-not-infer")
    assert resolve_user_id() == ""


@pytest.mark.parametrize("value", ["../other", "${USER_ID}", "-owner", "a b", "Owner"])
def test_invalid_resource_owner_is_rejected(value):
    with pytest.raises(ValueError, match="user_id"):
        resolve_user_id(value)
