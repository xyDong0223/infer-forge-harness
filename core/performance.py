"""Platform-neutral performance gates and metric comparison."""

from collections.abc import Iterable
import math
import re

from core.contracts import GateResult, Metric


_DIRECTIONS = {"higher_is_better", "lower_is_better"}


def _legacy_direction(name: str) -> str | None:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    lower = bool(tokens & {"latency", "tpot", "ttft"})
    higher = "throughput" in tokens
    if lower == higher:
        return None
    return "lower_is_better" if lower else "higher_is_better"


def _index_metrics(metrics: Iterable[Metric]) -> tuple[dict, set]:
    indexed = {}
    duplicates = set()
    for metric in metrics:
        key = (metric.name, tuple(sorted(metric.labels.items())))
        if key in indexed:
            duplicates.add(key)
        indexed[key] = metric
    return indexed, duplicates


def compare_metrics(
    baseline: Iterable[Metric],
    candidate: Iterable[Metric],
    *,
    max_regression: float = 0.05,
) -> list[GateResult]:
    """Compare every name/label pair; baseline entries are required candidates.

    Explicit directions take precedence over legacy name inference. Missing
    baselines are UNKNOWN, missing candidates FAIL, and invalid or ambiguous
    comparisons are INCOMPARABLE. A zero baseline has no relative-regression
    denominator and is also INCOMPARABLE.
    """
    if not math.isfinite(max_regression) or max_regression < 0:
        raise ValueError("max_regression must be finite and nonnegative")
    base, duplicate_base = _index_metrics(baseline)
    current, duplicate_current = _index_metrics(candidate)
    results = []
    for key in dict.fromkeys((*current, *base)):
        name, labels = key
        reference = base.get(key)
        metric = current.get(key)
        context = f" labels={dict(labels)}" if labels else ""
        if key in duplicate_base or key in duplicate_current:
            results.append(GateResult(name, "INCOMPARABLE", "duplicate metric key" + context))
            continue
        if metric is None:
            results.append(GateResult(name, "FAIL", "missing candidate" + context))
            continue
        if reference is None:
            results.append(GateResult(name, "UNKNOWN", "missing baseline" + context))
            continue
        if reference.unit != metric.unit:
            results.append(GateResult(
                name, "INCOMPARABLE",
                f"unit mismatch: baseline={reference.unit!r} candidate={metric.unit!r}" + context,
            ))
            continue
        if not math.isfinite(reference.value) or not math.isfinite(metric.value):
            results.append(GateResult(
                name, "INCOMPARABLE",
                f"nonfinite value: baseline={reference.value} candidate={metric.value}" + context,
            ))
            continue
        explicit = {m.direction for m in (reference, metric) if m.direction is not None}
        if explicit - _DIRECTIONS or len(explicit) > 1:
            results.append(GateResult(
                name, "INCOMPARABLE", "invalid or conflicting metric direction" + context,
            ))
            continue
        direction = next(iter(explicit)) if explicit else _legacy_direction(name)
        if direction is None:
            results.append(GateResult(
                name, "INCOMPARABLE", "metric direction is required for this name" + context,
            ))
            continue
        if reference.value == 0:
            results.append(GateResult(
                name, "INCOMPARABLE", "zero baseline; relative regression is undefined" + context,
            ))
            continue
        delta = metric.value - reference.value
        regression = (
            delta if direction == "lower_is_better" else -delta
        ) / abs(reference.value)
        passed = regression <= max_regression
        results.append(GateResult(
            name, "PASS" if passed else "FAIL",
            f"baseline={reference.value} candidate={metric.value} unit={metric.unit!r} "
            f"direction={direction} regression={regression} max_regression={max_regression}"
            + context,
        ))
    return results or [GateResult("metrics", "UNKNOWN", "no baseline or candidate metrics")]
