"""Independent MAT-008 acceptance checks.

The claim this guards is the strongest one in the evidence scale: `EXERCISED` says
the capability ran and was numerically correct. Two ways that claim goes bad, both
seen in this repo:

- A comparison that cannot fail. The first version of the quantization probe gated
  on cosine similarity, and cosine is invariant to the uniform per-channel factor
  that `scale_mm.py`'s `mul_(127.0)` supplies — so the wrong answer scored 0.99999.
  A pass therefore requires a negative control that actually missed the gate.
- A dimension that was declared and never run. `EVALUATION_UNIMPLEMENTED` is a
  legitimate state and must survive validation, but it must never read as exercised.
"""

from __future__ import annotations

from typing import Any

STATES = {
    "EXERCISED_PASS",
    "EXERCISED_FAIL",
    "EVALUATION_INCONCLUSIVE",
    "EVALUATION_UNIMPLEMENTED",
    "EVALUATION_ERROR",
}


def validate_evaluation(report: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    state = report.get("state")
    if state not in STATES:
        errors.append(f"state {state!r} is not one of {sorted(STATES)}")
    dimension = report.get("dimension")
    if not dimension:
        errors.append("dimension is required: the contract is parameterized by it")

    if state == "EVALUATION_UNIMPLEMENTED":
        if not report.get("reason"):
            errors.append("an unimplemented dimension must say why")
        if report.get("cases"):
            errors.append("an unimplemented dimension reported cases, which contradicts its state")
        return errors

    if state == "EVALUATION_ERROR":
        if not report.get("error"):
            errors.append("EVALUATION_ERROR must carry the error it hit")
        return errors

    if not report.get("operators"):
        errors.append("operators is required: exercised means named operators ran")
    if not report.get("exercised_in"):
        errors.append("exercised_in is required: a run without a host is not evidence")
    if not report.get("threshold_source"):
        errors.append("threshold_source is required: a report without it is an opinion")

    cases = report.get("cases") or []
    if not cases:
        errors.append("at least one comparison case is required")
    for case in cases:
        for field in ("case", "reference", "cosine", "relative_l2"):
            if case.get(field) is None:
                errors.append(f"case {case.get('case', '?')!r} is missing {field}")

    control = report.get("control") or {}
    if not control:
        errors.append("a negative control is required: a check that cannot fail proves nothing")

    if state == "EXERCISED_PASS":
        if not control.get("discriminates"):
            errors.append(
                "EXERCISED_PASS requires the negative control to miss the gate; this control "
                "passed too, so the probe is blind to what it claims to check"
            )
        thresholds = report.get("thresholds") or {}
        floor, ceiling = thresholds.get("min_cosine"), thresholds.get("max_relative_l2")
        if floor is None or ceiling is None:
            errors.append("EXERCISED_PASS requires both thresholds to be recorded")
        elif cases:
            primary = cases[0]
            if float(primary.get("relative_l2", 1.0)) > float(ceiling):
                errors.append(
                    f"relative L2 {primary.get('relative_l2')} exceeds {ceiling} but the state is "
                    "EXERCISED_PASS"
                )
            if float(primary.get("cosine", 0.0)) < float(floor):
                errors.append(
                    f"cosine {primary.get('cosine')} is below {floor} but the state is EXERCISED_PASS"
                )
        declared = ((contract.get("checks") or {}).get("dimensions") or {}).get(dimension or "")
        if declared and float(declared.get("max_relative_l2", 0)) != float(
            (report.get("thresholds") or {}).get("max_relative_l2", -1)
        ):
            errors.append(
                "the report's relative-L2 threshold does not match the contract's, so the gate "
                "was not the declared one"
            )
    return errors
