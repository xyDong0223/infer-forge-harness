"""Validate a ToyBringupReport.

The report is only worth anything if it says which stage was reached and proves the
run was the same code path as the real one. Two things are therefore refused: a
report that claims completion without a decode of more than one token, and a report
whose toy config shrank a dimension that selects kernels.
"""

from __future__ import annotations

from typing import Any

STAGES = ("CONFIG_DERIVED", "ENGINE_CONSTRUCTED", "PREFILL_OK", "DECODE_OK")

# Shrinking any of these changes which kernel or backend is selected, which would
# make a pass here say nothing about the real model.
DIMENSIONS_THAT_SELECT_KERNELS = (
    "hidden_size",
    "num_attention_heads",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "kv_lora_rank",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
)


def validate_bringup_report(
    report: dict[str, Any], contract: dict[str, Any], real_config: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []

    stage = report.get("stage")
    if stage != "NOTHING_RAN" and stage not in STAGES:
        errors.append(f"stage {stage!r} is not one of {STAGES} or NOTHING_RAN")

    passed = report.get("stages_passed")
    if not isinstance(passed, list):
        errors.append("stages_passed is required: a bring-up report is a position, not a verdict")
    else:
        if passed != list(STAGES[: len(passed)]):
            errors.append(f"stages_passed {passed} is not a prefix of {list(STAGES)}")
        if report.get("complete") and passed != list(STAGES):
            errors.append("complete is true but not every stage passed")

    if not report.get("complete") and not report.get("error"):
        errors.append("an incomplete bring-up must carry the error that stopped it")

    config = report.get("config") or {}
    if not config.get("architectures"):
        errors.append("config.architectures is required: the report must name what it brought up")

    depth = config.get("num_hidden_layers") or {}
    if not depth.get("toy") or not depth.get("real"):
        errors.append("config.num_hidden_layers must record both the real and the toy depth")
    elif depth["toy"] > depth["real"]:
        errors.append("the toy model is deeper than the real one")

    kept = config.get("kept_dimensions") or {}
    if real_config:
        for key in DIMENSIONS_THAT_SELECT_KERNELS:
            if key in real_config and key in kept and kept[key] != real_config[key]:
                errors.append(
                    f"{key} was changed from {real_config[key]} to {kept[key]}: shrinking a "
                    "kernel-selecting dimension makes this run a different code path"
                )
    elif not kept:
        errors.append(
            "config.kept_dimensions is required: without it nothing shows the toy run kept the "
            "dimensions that decide which kernel is selected"
        )

    if report.get("complete"):
        decode = report.get("decode") or {}
        if not isinstance(decode.get("count"), int) or decode["count"] < 2:
            errors.append(
                "a complete bring-up needs more than one decoded token: one token only proves "
                "prefill, and the decode path is where paged attention and the indexer differ"
            )

    return errors
