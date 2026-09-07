"""Independent MAT-007 acceptance checks.

A patch that makes the server start is not automatically a correct patch. The two
questions that decide whether it may stay are whether it computes the same thing
as the kernel it replaces, and whether it can be switched off again — without the
second, nothing can ever be compared against it, including the failure it was
written for.
"""

from __future__ import annotations

from typing import Any


def _chosen(contract: dict[str, Any]) -> list[dict[str, Any]]:
    mechanisms = ((contract.get("checks") or {}).get("mechanisms")) or []
    return [mechanism for mechanism in mechanisms if mechanism.get("chosen")]


def validate_patch_placement(report: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    checks = contract.get("checks") or {}
    acceptance = contract.get("acceptance") or {}

    if report.get("state") != "PATCH_PLACED":
        errors.append(f"state is {report.get('state')!r}, not PATCH_PLACED")

    chosen = _chosen(contract)
    if len(chosen) != 1:
        errors.append("exactly one placement mechanism must be chosen in the contract")
    for mechanism in ((checks.get("mechanisms")) or []):
        if not mechanism.get("why"):
            errors.append(f"mechanism {mechanism.get('id')!r} must record why it was or was not chosen")
    if chosen and report.get("mechanism") != chosen[0].get("id"):
        errors.append(
            f"the report placed the patch via {report.get('mechanism')!r} but the contract chose "
            f"{chosen[0].get('id')!r}"
        )

    if acceptance.get("require_numerical_evidence"):
        numerical = report.get("numerical") or {}
        threshold = float((checks.get("numerical_equivalence") or {}).get("threshold", 0.9999))
        similarity = numerical.get("min_cosine_similarity")
        if not numerical.get("reference"):
            errors.append("numerical.reference must name the kernel the patch was compared against")
        if not numerical.get("cases"):
            errors.append("numerical.cases must record the geometries that were compared")
        if not isinstance(similarity, (int, float)):
            errors.append("numerical.min_cosine_similarity is required: replacing a kernel without "
                          "comparing it is a rewrite, not a patch")
        elif similarity < threshold:
            errors.append(
                f"min cosine similarity {similarity} is below the contract's {threshold}"
            )

    if acceptance.get("require_server_validation"):
        server = report.get("server_validation") or {}
        if server.get("state") != "DEPLOYMENT_READY":
            errors.append(
                "the patch must be validated by a service proof: isolation already passed before "
                "the patch existed, so it proves nothing here"
            )

    if acceptance.get("require_reversibility"):
        reversibility = report.get("reversibility") or {}
        if not reversibility.get("backup_path"):
            errors.append("reversibility.backup_path must record what --remove will restore")
        if reversibility.get("runtime_switch") != checks.get("reversibility", {}).get("runtime_switch"):
            errors.append("the runtime switch must match the contract")
        if checks.get("reversibility", {}).get("must_reproduce_original_failure") and not reversibility.get(
            "reproduces_original_failure"
        ):
            errors.append(
                "turning the patch off must still reproduce the original failure; otherwise the "
                "evidence for needing it has been lost"
            )

    scope = report.get("scope") or {}
    if scope.get("routes_only") != (checks.get("scope") or {}).get("routes_only"):
        errors.append(
            "the routing scope must match the contract: widening it silently changes results on "
            "paths the fallback does not implement"
        )
    limitations = report.get("limitations") or []
    if not limitations:
        errors.append("a patch with no stated limitation is a claim nobody can check")
    return errors
