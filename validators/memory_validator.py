"""Independent acceptance checks for a memory budget report.

The Tool collects and computes; this decides. A budget that cannot be traced
back to the cards is rejected even when every number looks plausible, because a
memory claim without a device counter behind it is not evidence.
"""

from __future__ import annotations

from typing import Any

# A P800 that shows no unattributed memory has almost certainly been reported
# from a log that does not belong to the snapshot: the driver, the XPU runtime
# and the allocator always hold something.
MIN_UNATTRIBUTED_MIB = 64


def validate_memory_budget(budget: dict[str, Any], thresholds: dict[str, Any] | None = None) -> list[str]:
    """Return acceptance errors for one rank's budget.

    Recognised thresholds:
      min_free_mib          headroom that must remain on the card
      max_utilization_pct   upper bound on used/total
      min_kv_cache_tokens   KV pool capacity the deployment must reach
      max_card_spread_mib   tolerated imbalance across ranks
    """
    thresholds = thresholds or {}
    errors: list[str] = []

    if not budget.get("hbm_total_matches_device"):
        errors.append("xpu_smi total HBM disagrees with the device spec in the catalog")
    if not budget.get("reconciled"):
        errors.append(
            "log-derived memory does not reconcile with the xpu_smi snapshot: "
            f"unattributed={budget.get('unattributed_mib')} MiB vs "
            f"analyzer other={budget.get('analyzer_other_mib')} MiB"
        )
    unattributed = budget.get("unattributed_mib")
    if not isinstance(unattributed, int):
        errors.append("unattributed_mib is missing: the snapshot was not applied")
    elif unattributed < MIN_UNATTRIBUTED_MIB:
        errors.append(
            f"unattributed memory is {unattributed} MiB, below {MIN_UNATTRIBUTED_MIB} MiB: "
            "the log and the snapshot probably describe different processes"
        )
    if not (budget.get("evidence") or {}).get("xpu_smi"):
        errors.append("evidence.xpu_smi must record the device snapshot path")

    if "min_free_mib" in thresholds and budget.get("measured_free_mib", 0) < thresholds["min_free_mib"]:
        errors.append(
            f"free memory {budget.get('measured_free_mib')} MiB is below the required "
            f"{thresholds['min_free_mib']} MiB of headroom"
        )
    if "max_utilization_pct" in thresholds and budget.get("utilization_pct", 100) > thresholds["max_utilization_pct"]:
        errors.append(
            f"utilization {budget.get('utilization_pct')}% exceeds "
            f"{thresholds['max_utilization_pct']}%"
        )
    if "min_kv_cache_tokens" in thresholds:
        tokens = budget.get("kv_cache_tokens")
        if not isinstance(tokens, int):
            errors.append("kv_cache_tokens is unknown: the log has no KV cache size line")
        elif tokens < thresholds["min_kv_cache_tokens"]:
            errors.append(
                f"KV pool holds {tokens} tokens, below the required "
                f"{thresholds['min_kv_cache_tokens']}"
            )
    if "max_card_spread_mib" in thresholds and budget.get("card_spread_mib", 0) > thresholds["max_card_spread_mib"]:
        errors.append(
            f"per-card spread {budget.get('card_spread_mib')} MiB exceeds "
            f"{thresholds['max_card_spread_mib']} MiB"
        )
    return errors
