"""Platform-neutral performance gates and metric comparison."""

from collections.abc import Iterable

from core.contracts import GateResult, Metric


def compare_metrics(
    baseline: Iterable[Metric],
    candidate: Iterable[Metric],
    *,
    max_regression: float = 0.05,
) -> list[GateResult]:
    base = {(m.name, tuple(sorted(m.labels.items()))): m for m in baseline}
    results = []
    for metric in candidate:
        key = (metric.name, tuple(sorted(metric.labels.items())))
        reference = base.get(key)
        if reference is None:
            results.append(GateResult(metric.name, "UNKNOWN", "no baseline"))
            continue
        # Higher throughput is good; lower latency is good.
        ratio = metric.value / reference.value if reference.value else 0.0
        regression = reference.value / metric.value - 1 if metric.value else float("inf")
        if "latency" in metric.name.lower() or "tpot" in metric.name.lower():
            passed = regression <= max_regression
        else:
            passed = ratio >= 1 - max_regression
        results.append(GateResult(
            metric.name, "PASS" if passed else "FAIL",
            f"baseline={reference.value} candidate={metric.value}",
        ))
    return results
