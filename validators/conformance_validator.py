"""Independent MAT-009 acceptance checks.

The claim is narrow and easy to overstate: a parser pair handles this model's output
format. Three ways that goes wrong, all seen while building the probe:

- The control does not discriminate. The first version removed the markers, and
  Qwen3's parser correctly reports unmarked text as reasoning-in-progress, so the
  control "passed" and proved nothing. A conformance verdict now requires a control
  that actually behaved differently.
- The sample is invented. A marker that never appears in the model's own chat
  template makes the whole check a statement about a format the model does not emit.
- A registry name is mistaken for a working parser, the same substitution MAT-003
  already warns about on the model side.
"""

from __future__ import annotations

from typing import Any

STATES = {"CONFORMANT", "NONCONFORMANT", "PARSER_ABSENT", "EVALUATION_INCONCLUSIVE",
          "EVALUATION_ERROR"}


def validate_conformance(report: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    state = report.get("state")
    if state not in STATES:
        errors.append(f"state {state!r} is not one of {sorted(STATES)}")
    if not report.get("registries"):
        errors.append("registries are required: which parsers the installed runtime has is "
                      "the first fact this Task establishes")
    if not report.get("profile_source"):
        errors.append("profile_source is required: the samples must be traceable to the contract")

    if state in ("PARSER_ABSENT", "EVALUATION_ERROR", "EVALUATION_INCONCLUSIVE"):
        if not report.get("error"):
            errors.append(f"{state} must carry the reason")
        return errors

    cases = report.get("cases") or []
    if not cases:
        errors.append("at least one parser check is required")
    for case in cases:
        label = case.get("case", "?")
        for field in ("parser", "implementation", "control"):
            if not case.get(field):
                errors.append(f"case {label!r} is missing {field}")
        control = case.get("control") or {}
        if control.get("discriminates") is None:
            errors.append(f"case {label!r} does not say whether its control discriminated")

    acceptance = contract.get("acceptance") or {}
    if acceptance.get("require_template_confirms_markers"):
        if report.get("template_confirms_markers") is not True:
            errors.append(
                "the model's chat template did not confirm the sample's markers, so the "
                "sample is not evidence about this model"
            )

    if state == "CONFORMANT":
        for case in cases:
            label = case.get("case", "?")
            if not (case.get("control") or {}).get("discriminates"):
                errors.append(
                    f"case {label!r} is CONFORMANT with a control that behaved the same way, "
                    "so the check cannot tell a working parser from one that always fires"
                )
            if label.startswith("reasoning") and not case.get("separated"):
                errors.append(f"case {label!r} did not separate reasoning from content")
            if label.startswith("tool"):
                if not case.get("tools_called"):
                    errors.append(f"case {label!r} reported no tool call")
                if case.get("arguments_are_json") is not True:
                    errors.append(f"case {label!r} produced arguments that are not JSON")
                if case.get("name_matches") is False:
                    errors.append(f"case {label!r} parsed a different function name than expected")
    return errors
