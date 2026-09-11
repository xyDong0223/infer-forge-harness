import pytest

from orchestration.discovery import (
    IncompleteOperatorEvidence,
    operator_spec_from_entry,
    operator_specs_from_report,
)


def _entry():
    return {
        "name": "paged_decode",
        "inputs": [{"name": "q", "dtype": "float16", "shape": [1, 8], "layout": "contiguous"}],
        "outputs": [{"name": "out", "dtype": "float16", "shape": [1, 8], "layout": "contiguous"}],
        "semantics_basis": "torch reference captured from the failing call",
        "location": "plugin/attention.py:42",
    }


def test_discovery_requires_measured_tensor_contract():
    entry = _entry()
    entry.pop("inputs")
    with pytest.raises(IncompleteOperatorEvidence, match="inputs"):
        operator_spec_from_entry(entry, model_id="m", backend="P800")


def test_discovery_rejects_unmapped_signals():
    with pytest.raises(ValueError, match="unmapped"):
        operator_specs_from_report(
            {"unmapped_signals": [{"symbol": "unknown"}], "entries": []},
            model_id="m",
            backend="P800",
        )


def test_discovery_preserves_report_evidence_and_is_idempotent():
    report = {"plugin": "vllm_kunlun", "version": "p1", "entries": [_entry()]}
    first = operator_specs_from_report(report, model_id="m", backend="P800")[0]
    second = operator_specs_from_report(report, model_id="m", backend="P800")[0]
    assert first.operator_key == second.operator_key
    assert first.evidence["plugin"] == "vllm_kunlun"
