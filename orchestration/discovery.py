"""Turn shim and operator-gap reports into strict :class:`OperatorSpec` objects.

The shim scanner deliberately reports *signals*, rather than tensor contracts.
This module is the boundary between that report and the orchestration scheduler:
it only creates a spec when the report carries measured input/output schemas and
semantic evidence.  In particular, it never invents a tensor shape, dtype,
layout, or reference implementation from a function name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import IOSpec, OperatorSpec


class DiscoveryError(ValueError):
    """The report cannot safely be converted into an operator specification."""


class IncompleteOperatorEvidence(DiscoveryError):
    """A candidate is missing input/output schema or semantic evidence."""


def _text(value: Any, field: str, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IncompleteOperatorEvidence(f"{context}: {field} is required")
    return value.strip()


def _io_specs(value: Any, field: str, *, context: str) -> list[IOSpec]:
    if not isinstance(value, (list, tuple)) or not value:
        raise IncompleteOperatorEvidence(
            f"{context}: {field} must be a non-empty list with measured tensor schemas"
        )
    result: list[IOSpec] = []
    for index, item in enumerate(value):
        item_context = f"{context}.{field}[{index}]"
        if isinstance(item, IOSpec):
            result.append(item)
            continue
        if not isinstance(item, Mapping):
            raise IncompleteOperatorEvidence(f"{item_context} must be an object")
        missing = [key for key in ("name", "dtype", "shape", "layout") if key not in item]
        if missing:
            raise IncompleteOperatorEvidence(
                f"{item_context} is missing required fields: {', '.join(missing)}"
            )
        # IOSpec performs the final value checks.  Convert its errors to the
        # discovery-specific exception so callers can classify the report.
        try:
            result.append(IOSpec.from_dict(dict(item)))
        except (TypeError, ValueError) as exc:
            raise IncompleteOperatorEvidence(f"{item_context}: {exc}") from exc
    return result


def _semantics(entry: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    raw = entry.get("semantics")
    if raw is not None and not isinstance(raw, Mapping):
        raise IncompleteOperatorEvidence(f"{context}: semantics must be an object")

    semantics = dict(raw or {})
    basis = entry.get("semantics_basis")
    if basis is not None:
        basis = _text(basis, "semantics_basis", context=context)
        semantics.setdefault("basis", basis)

    # A non-empty arbitrary dictionary is not enough: it could be metadata
    # copied from the registry without describing the operator's behaviour.
    evidence_keys = {
        "reference",
        "reference_impl",
        "reference_implementation",
        "formula",
        "description",
        "basis",
        "semantics_basis",
        "evidence",
    }
    has_evidence = any(
        key in semantics and semantics[key] not in (None, "", [], {})
        for key in evidence_keys
    )
    if not has_evidence:
        raise IncompleteOperatorEvidence(
            f"{context}: semantic evidence is required (reference, formula, description, "
            "or semantics_basis)"
        )
    return semantics


def _entries(report: Any) -> list[Mapping[str, Any]]:
    if isinstance(report, Mapping):
        values = report.get("entries")
        if values is None:
            values = report.get("gaps")
        if values is None:
            values = report.get("findings")
        if values is None:
            # A single registry entry is useful for callers handling one gap.
            values = [report] if any(k in report for k in ("name", "operator_id", "symbol")) else []
    else:
        values = report
    if not isinstance(values, (list, tuple)):
        raise DiscoveryError("report entries/gaps must be a list")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise DiscoveryError(f"entries[{index}] must be an object")
        result.append(item)
    return result


def operator_spec_from_entry(
    entry: Mapping[str, Any],
    *,
    model_id: str,
    backend: str,
    model_revision: str = "unknown",
    plugin_revision: str = "unknown",
    environment: Mapping[str, Any] | None = None,
    report: Mapping[str, Any] | None = None,
) -> OperatorSpec:
    """Build one spec from a registry/gap entry.

    ``inputs`` and ``outputs`` must be explicit measured schemas.  Semantic
    evidence must be supplied as ``semantics`` or ``semantics_basis``.  Missing
    evidence raises :class:`IncompleteOperatorEvidence` instead of producing a
    speculative spec.
    """

    if not isinstance(entry, Mapping):
        raise DiscoveryError("operator entry must be an object")
    context = str(entry.get("name") or entry.get("operator_id") or entry.get("symbol") or "entry")
    operator_id = _text(
        entry.get("operator_id") or entry.get("name") or entry.get("symbol"),
        "operator id",
        context=context,
    )
    model_id = _text(model_id, "model_id", context=context)
    backend = _text(backend, "backend", context=context)
    inputs = _io_specs(entry.get("inputs"), "inputs", context=context)
    outputs = _io_specs(entry.get("outputs"), "outputs", context=context)
    semantics = _semantics(entry, context=context)

    location = entry.get("location")
    if location is not None:
        location = _text(location, "location", context=context)
    evidence: dict[str, Any] = dict(entry.get("evidence") or {})
    for key in ("location", "replaced_kernel", "call_frequency", "semantics_basis", "source", "class"):
        if key in entry and entry[key] not in (None, "", [], {}):
            evidence.setdefault(key, entry[key])
    if report:
        signals = [
            signal
            for signal in report.get("signals", [])
            if isinstance(signal, Mapping)
            and signal.get("symbol") == operator_id
        ]
        if signals:
            evidence.setdefault("signals", [dict(signal) for signal in signals])
        for key in ("plugin", "plugin_path", "scanned_in"):
            if report.get(key) not in (None, ""):
                evidence.setdefault(key, report[key])

    return OperatorSpec(
        operator_id=operator_id,
        model_id=model_id,
        model_revision=_text(model_revision, "model_revision", context=context),
        plugin_revision=_text(plugin_revision, "plugin_revision", context=context),
        backend=backend,
        inputs=inputs,
        outputs=outputs,
        semantics=semantics,
        evidence=evidence,
        environment=dict(environment or {}),
    )


def operator_specs_from_report(
    report: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    model_id: str,
    backend: str,
    model_revision: str = "unknown",
    plugin_revision: str | None = None,
    environment: Mapping[str, Any] | None = None,
    include_waived: bool = False,
) -> list[OperatorSpec]:
    """Convert all actionable entries in a shim/gap report.

    Unmapped scanner signals are rejected because they have no declared tensor
    or semantic contract.  Waived entries are skipped by default; requesting
    ``include_waived`` still validates them and therefore cannot bypass the
    evidence gate.
    """

    report_mapping = report if isinstance(report, Mapping) else None
    if report_mapping and report_mapping.get("unmapped_signals"):
        raise DiscoveryError(
            "report contains unmapped shim signals; declare their input/output and semantic "
            "evidence before creating an OperatorSpec"
        )
    entries = _entries(report)
    resolved_plugin_revision = plugin_revision
    if resolved_plugin_revision is None and report_mapping:
        resolved_plugin_revision = report_mapping.get("plugin_revision") or report_mapping.get("version")
    resolved_plugin_revision = resolved_plugin_revision or "unknown"

    specs: list[OperatorSpec] = []
    for index, entry in enumerate(entries):
        status = entry.get("status")
        if status == "WAIVED" and not include_waived:
            continue
        if status == "WAIVED" and not entry.get("reason"):
            raise DiscoveryError(f"entries[{index}] is WAIVED without a reason")
        specs.append(
            operator_spec_from_entry(
                entry,
                model_id=model_id,
                backend=backend,
                model_revision=model_revision,
                plugin_revision=resolved_plugin_revision,
                environment=environment,
                report=report_mapping,
            )
        )
    return specs


def load_report(path: str | Path) -> Any:
    """Load a JSON shim/gap report from disk."""

    report_path = Path(path)
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"cannot load discovery report {report_path}: {exc}") from exc


# Concise aliases for integrations and callers migrating from the scanner.
from_shim_entry = operator_spec_from_entry
specs_from_shim_report = operator_specs_from_report
discover_operator_specs = operator_specs_from_report


__all__ = [
    "DiscoveryError",
    "IncompleteOperatorEvidence",
    "operator_spec_from_entry",
    "operator_specs_from_report",
    "discover_operator_specs",
    "from_shim_entry",
    "specs_from_shim_report",
    "load_report",
]
