"""Independent MAT-020 acceptance checks.

A handoff is the one artifact that leaves this project, so the rule is that it
must not overstate what was established. The specific failure to prevent is a
ticket that reads as standalone-reproducible when it is not: the vendor tries the
arguments directly, they pass, and the ticket comes back rejected with the real
problem untouched.
"""

from __future__ import annotations

from typing import Any

BINARY_LAYERS = {"kunlun_ops_vendor", "torch_xmlir_vendor"}


def validate_handoff(ticket: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    acceptance = contract.get("acceptance") or {}

    if ticket.get("state") != "HANDOFF_READY":
        errors.append(f"state is {ticket.get('state')!r}, not HANDOFF_READY")
    if ticket.get("layer") not in BINARY_LAYERS:
        errors.append(
            f"layer {ticket.get('layer')!r} is modifiable here, so the work is ours, not a handoff"
        )
    for field in ("symbol", "error_text", "expected", "observed", "reproduction"):
        if not ticket.get(field):
            errors.append(f"{field} is required in a ticket someone else has to act on")
    if not ticket.get("arguments"):
        errors.append("arguments must be included: a symptom without them cannot be reproduced")
    if acceptance.get("require_environment_fingerprint") and not ticket.get("environment_fingerprint"):
        errors.append("environment_fingerprint is required: the failure holds for one combination")

    reproduces = ticket.get("reproduces_in_isolation")
    if reproduces not in (True, False):
        errors.append("reproduces_in_isolation must be stated explicitly, not left unsaid")
    else:
        text = str(ticket.get("reproduction", "")).lower()
        if not reproduces and "only occurs inside the server" not in text:
            errors.append(
                "this failure does not reproduce standalone, so the ticket must say the reproduction "
                "requires the full serving run"
            )
        if reproduces and "directly" not in text:
            errors.append("a standalone-reproducible failure must say how to call it directly")
    if acceptance.get("require_ruled_out") and not ticket.get("ruled_out"):
        errors.append("ruled_out must list what was already eliminated, or the vendor repeats it")
    if ticket.get("owner") != "vendor":
        errors.append("a handoff is owned by the vendor by definition")
    return errors
