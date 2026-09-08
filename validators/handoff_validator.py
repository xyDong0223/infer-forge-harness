"""Independent MAT-020 acceptance checks.

A handoff is the one artifact that leaves this project, so the rule is that it
must not overstate what was established. The specific failure to prevent is a
ticket that reads as standalone-reproducible when it is not: the vendor tries the
arguments directly, they pass, and the ticket comes back rejected with the real
problem untouched.

There are two forms. A single symbol on a binary layer is a ticket, checked by
`validate_handoff`. A model that stops against several walls across several layers is
a work package, checked by `validate_handoff_package`, and it invites the opposite
error: reading like a finished adaptation with follow-ups.
"""

from __future__ import annotations

from typing import Any

BINARY_LAYERS = {"kunlun_ops_vendor", "torch_xmlir_vendor"}
OWNED_LAYERS = {"vllm_upstream", "vllm_kunlun_plugin"}
ALL_LAYERS = BINARY_LAYERS | OWNED_LAYERS
OWNERS = {"us", "upstream", "vendor"}
KINDS = {"observed_failure", "unreached_dependency"}


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


def _validate_finding(finding: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    where = finding.get("id") or f"finding[{index}]"

    for field in ("id", "symbol", "site", "minimal_fix", "verified_against"):
        if not finding.get(field):
            errors.append(f"{where}: {field} is required in an item someone else has to act on")
    site = str(finding.get("site") or "")
    if site and ":" not in site:
        errors.append(f"{where}: site {site!r} must name a file and a line, not just a file")

    layer = finding.get("layer")
    if layer not in ALL_LAYERS:
        errors.append(f"{where}: layer must be one of {sorted(ALL_LAYERS)}, got {layer!r}")
    owner = finding.get("owner")
    if owner not in OWNERS:
        errors.append(f"{where}: owner must be one of {sorted(OWNERS)}, got {owner!r}")
    elif owner == "vendor" and layer in OWNED_LAYERS:
        errors.append(f"{where}: {layer} is modifiable here, so the vendor cannot own it")
    elif owner == "us" and layer in BINARY_LAYERS:
        errors.append(f"{where}: {layer} ships as a binary, so this is a ticket or a workaround, not ours")
    elif owner == "upstream":
        if layer != "vllm_upstream":
            errors.append(f"{where}: only a vllm_upstream item can be filed upstream, not {layer!r}")
        if not finding.get("generality"):
            errors.append(
                f"{where}: an item filed upstream must state who else it hits, or it reads as our "
                "platform's private problem and gets closed"
            )

    kind = finding.get("kind")
    if kind not in KINDS:
        errors.append(f"{where}: kind must be one of {sorted(KINDS)}, got {kind!r}")
    elif kind == "observed_failure":
        if not finding.get("error_text"):
            errors.append(f"{where}: an observed failure must carry the error verbatim")
        if finding.get("reproduces_in_isolation") not in (True, False):
            errors.append(f"{where}: reproduces_in_isolation must be stated explicitly, not left unsaid")
    else:
        # The failure to prevent: a list where predicted work is indistinguishable
        # from measured work, so the reader trusts both equally.
        if finding.get("error_text"):
            errors.append(
                f"{where}: this item was never reached, so it cannot carry an error_text — "
                "a predicted symptom presented as an observed one is the whole problem"
            )
        if not finding.get("blocked_by"):
            errors.append(f"{where}: an unreached item must say what stopped the run before it")

    if finding.get("stand_in") and not finding.get("stand_in_is_not_a_fix"):
        errors.append(
            f"{where}: a stand-in must be labelled as one; otherwise the reader takes the item as done"
        )
    return errors


def validate_handoff_package(package: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    """Acceptance for the multi-finding form: a port work package, not one ticket.

    A single symbol on a binary layer is a ticket. A model that walks into several
    walls across several layers is a work package, and the failure it invites is the
    opposite of the ticket's: not overstating reproducibility, but reading like a
    finished adaptation with follow-ups. So the package has to say where the run
    stopped and must not claim the subject serves.
    """
    errors: list[str] = []
    acceptance = contract.get("acceptance") or {}

    if package.get("state") != "HANDOFF_READY":
        errors.append(f"state is {package.get('state')!r}, not HANDOFF_READY")
    if not package.get("subject"):
        errors.append("subject is required: a package without a model is not actionable")
    if acceptance.get("require_environment_fingerprint") and not package.get("environment_fingerprint"):
        errors.append("environment_fingerprint is required: the walls hold for one combination")

    findings = package.get("findings") or []
    if not findings:
        errors.append("a package with no findings is not a handoff")
    seen: set[str] = set()
    for index, finding in enumerate(findings):
        errors += _validate_finding(finding, index)
        identifier = finding.get("id")
        if identifier in seen:
            errors.append(f"duplicate finding id {identifier!r}")
        seen.add(identifier)

    status = package.get("subject_status") or {}
    if status.get("serves_requests") not in (True, False):
        errors.append("subject_status.serves_requests must be stated explicitly")
    elif status.get("serves_requests"):
        errors.append("if the subject serves requests this is not a handoff but a deployment proof")
    if not status.get("stopped_at"):
        errors.append(
            "subject_status.stopped_at is required: without it the package reads as a completed "
            "adaptation with minor follow-ups"
        )
    if acceptance.get("require_ruled_out") and not package.get("ruled_out"):
        errors.append("ruled_out must list what was already eliminated, or the reader repeats it")
    return errors
