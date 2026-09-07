"""Independent MAT-002 acceptance checks.

The scan's value is that it can say "this is not a gap". That makes one failure
mode worse than a wrong answer: silently downgrading an unreachable upstream into
`ABSENT`, which sends the next Task to write a model file that may already exist.
"""

from __future__ import annotations

from typing import Any

VERDICTS = {"KUNLUN_OOT", "UPSTREAM_GENERIC", "MAIN_ONLY", "PR_PENDING", "ABSENT", "UNKNOWN_UPSTREAM"}
# An ABSENT verdict is only credible when both upstream lookups actually ran.
ABSENT_EVIDENCE = {"main_lookup": "NOT_FOUND", "pr_lookup": "NOT_FOUND"}


def validate_support_card(scan: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    acceptance = contract.get("acceptance") or {}

    if scan.get("state") != "SCAN_READY":
        errors.append(f"scan state is {scan.get('state')!r}, not SCAN_READY")
    if not scan.get("scanned_in"):
        errors.append("scanned_in must record the pod: a scan of an unnamed runtime is not evidence")

    results = scan.get("results") or []
    if not results:
        errors.append("no architecture was classified")

    for entry in results:
        arch = entry.get("architecture", "?")
        verdict = entry.get("verdict")
        if verdict not in VERDICTS:
            errors.append(f"{arch}: unknown verdict {verdict!r}")
            continue
        if not entry.get("meaning"):
            errors.append(f"{arch}: a verdict without its meaning invites the wrong next action")
        if verdict == "ABSENT":
            for key, expected in ABSENT_EVIDENCE.items():
                if entry.get(key) != expected:
                    errors.append(
                        f"{arch}: ABSENT claimed while {key}={entry.get(key)!r}; an unreachable "
                        "upstream must stay UNKNOWN_UPSTREAM"
                    )
        if verdict == "PR_PENDING" and not entry.get("pull_requests"):
            errors.append(f"{arch}: PR_PENDING must cite the pull requests it found")
        if verdict == "KUNLUN_OOT" and not entry.get("in_kunlun_oot"):
            errors.append(f"{arch}: KUNLUN_OOT claimed but the installed registry does not list it")
        if verdict == "UPSTREAM_GENERIC" and not entry.get("in_installed_vllm"):
            errors.append(f"{arch}: UPSTREAM_GENERIC claimed but the installed vLLM does not know it")

    if acceptance.get("require_installed_registry_evidence"):
        for entry in results:
            if not isinstance(entry.get("kunlun_oot_archs"), list):
                errors.append(
                    f"{entry.get('architecture', '?')}: the installed OOT registry contents must be "
                    "recorded, not summarised"
                )
    return errors
