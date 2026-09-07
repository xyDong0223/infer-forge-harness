"""Independent MAT-006 acceptance checks.

The point of triage is attribution: which layer owns the failure, and therefore
who can fix it. The expensive mistake is claiming a vendor gap from a symptom,
because that ends with a support ticket nobody can act on. So an attribution is
only accepted when an isolated reproduction was actually attempted, and the two
possible outcomes lead to different verdicts:

  isolated call fails too   -> VENDOR_KERNEL_GAP, reproducible, ticket-ready
  isolated call passes      -> RUNTIME_STATE_DEPENDENT, not a shape bug; the
                               reproduction is the server, so a workaround must
                               be validated in the server

Qwen3-8B on P800 is the second case, which is why it exists as a verdict.
"""

from __future__ import annotations

from typing import Any

LAYERS = {"vllm_upstream", "vllm_kunlun_plugin", "kunlun_ops_vendor", "torch_xmlir_vendor"}
OWNED_LAYERS = {"vllm_upstream", "vllm_kunlun_plugin"}
VERDICTS = {
    "VENDOR_KERNEL_GAP",
    "RUNTIME_STATE_DEPENDENT",
    "PLUGIN_DEFECT",
    "CONFIGURATION",
    "UNKNOWN",
}


def validate_triage(report: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    acceptance = contract.get("acceptance") or {}

    if report.get("state") != "TRIAGE_READY":
        errors.append(f"state is {report.get('state')!r}, not TRIAGE_READY")
    if not report.get("failing_symbol"):
        errors.append("failing_symbol must name the call that failed")
    if not report.get("error_text"):
        errors.append("error_text must quote what the runtime actually said")

    layer = report.get("layer")
    if layer not in LAYERS:
        errors.append(f"layer must be one of {sorted(LAYERS)}, got {layer!r}")

    verdict = report.get("verdict")
    if verdict not in VERDICTS:
        errors.append(f"unknown verdict {verdict!r}")

    isolated = report.get("isolated_reproduction")
    if not isinstance(isolated, dict):
        errors.append(
            "isolated_reproduction is required: an attribution without a reproduction attempt "
            "is a guess"
        )
    else:
        attempted = isolated.get("attempted")
        reproduced = isolated.get("reproduced")
        if attempted is not True:
            errors.append("isolated_reproduction.attempted must be True before attributing a layer")
        if reproduced not in (True, False):
            errors.append("isolated_reproduction.reproduced must be recorded as True or False")
        if not isolated.get("cases"):
            errors.append("isolated_reproduction.cases must record what was actually run")
        if verdict == "VENDOR_KERNEL_GAP" and reproduced is not True:
            errors.append(
                "VENDOR_KERNEL_GAP requires the isolated call to fail as well; otherwise the "
                "verdict is RUNTIME_STATE_DEPENDENT"
            )
        if verdict == "RUNTIME_STATE_DEPENDENT" and reproduced is not False:
            errors.append(
                "RUNTIME_STATE_DEPENDENT means the isolated call passed; a reproducing case makes "
                "it a plain vendor gap"
            )

    if acceptance.get("require_captured_arguments") and not report.get("captured_arguments"):
        errors.append(
            "the failing call's real arguments must be captured; a symptom without arguments "
            "cannot be reproduced by anyone else"
        )
    if acceptance.get("require_owner") :
        owner = report.get("owner")
        if not owner:
            errors.append("owner must state who can fix this")
        elif layer in OWNED_LAYERS and owner == "vendor":
            errors.append(f"{layer} is modifiable here, so the owner cannot be the vendor")
        elif layer and layer not in OWNED_LAYERS and owner == "us":
            errors.append(
                f"{layer} ships as a binary, so the fix is a workaround or a ticket, not ours"
            )

    workaround = report.get("workaround")
    if workaround:
        if not workaround.get("description"):
            errors.append("a workaround must describe itself")
        if workaround.get("validated_in_server") is not True:
            errors.append(
                "a workaround is only a workaround once it is validated in the server: the "
                "isolated call already passed, so isolation proves nothing here"
            )
    return errors
