"""Validate a TorchShimRegistry.

The contract this enforces: every torch shim that stands in for a vendor kernel
must be declared, and every declared, non-waived shim must have an operator
request. GLM-5.2 served with three hot-path torch shims and zero operator
requests because nothing refused the in-between state.

The validator therefore refuses three things: a signal the registry cannot
explain (an undeclared shim), a declared entry missing the facts an operator
request needs (location, replaced kernel, call frequency, semantics basis), and
a state that disagrees with its own content -- HANDOFF_CLEAR over unmapped
signals, or DISPATCHED without a request id.
"""

from __future__ import annotations

from typing import Any

ENTRY_STATUSES = {"REGISTERED", "DISPATCHED", "WAIVED"}
REQUIRED_ENTRY_FIELDS = ("name", "location", "replaced_kernel", "call_frequency", "semantics_basis")


def validate_shim_handoff(report: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    state = report.get("state")
    if state not in ("HANDOFF_CLEAR", "HANDOFF_FOUND"):
        errors.append(f"state {state!r} is neither HANDOFF_CLEAR nor HANDOFF_FOUND")

    plugin = report.get("plugin")
    if not plugin:
        errors.append("plugin is required: shims belong to a specific installed package")

    scanned = report.get("files_scanned")
    if not isinstance(scanned, int) or scanned <= 0:
        errors.append("files_scanned must be a positive count: an empty scan proves nothing")

    signals = report.get("signals")
    if signals is None:
        errors.append("signals is required, even when empty")
    else:
        for index, signal in enumerate(signals):
            for field in ("kind", "file", "symbol", "line"):
                if not signal.get(field) and signal.get(field) != 0:
                    errors.append(f"signals[{index}].{field} is required")

    entries = report.get("entries")
    if entries is None:
        errors.append("entries is required, even when empty")
    else:
        for index, entry in enumerate(entries):
            for field in REQUIRED_ENTRY_FIELDS:
                if not entry.get(field):
                    errors.append(
                        f"entries[{index}].{field} is required: an operator request without "
                        "it makes the next Task guess what to build"
                    )
            status = entry.get("status")
            if status not in ENTRY_STATUSES:
                errors.append(f"entries[{index}].status {status!r} is not one of {sorted(ENTRY_STATUSES)}")
            if status == "WAIVED" and not entry.get("reason"):
                errors.append(f"entries[{index}] is WAIVED without a reason")
            if status == "DISPATCHED" and not entry.get("request_id"):
                errors.append(f"entries[{index}] is DISPATCHED without a request_id")

    unmapped = report.get("unmapped_signals")
    if unmapped is None:
        errors.append("unmapped_signals is required, even when empty")
    elif unmapped and state == "HANDOFF_CLEAR":
        errors.append(
            "state is HANDOFF_CLEAR but signals exist that no registry entry explains: "
            "declare them, dispatch them, or waive them"
        )

    if state == "HANDOFF_CLEAR":
        for index, entry in enumerate(entries or []):
            if entry.get("status") == "REGISTERED":
                errors.append(
                    f"entries[{index}] {entry.get('name')!r} is still REGISTERED in a "
                    "HANDOFF_CLEAR report: dispatch it or waive it"
                )

    dispatched_now = [
        entry for entry in entries or [] if entry.get("dispatched_this_run")
    ]
    if dispatched_now:
        dispatch = report.get("dispatch")
        if not dispatch:
            errors.append("entries were dispatched this run but there is no dispatch record")
        elif dispatch.get("request_count") != len(dispatched_now):
            errors.append(
                f"dispatch.request_count {dispatch.get('request_count')} does not match the "
                f"{len(dispatched_now)} entries dispatched this run"
            )

    return errors
