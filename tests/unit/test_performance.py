import pytest

from core.contracts import Metric
from core.performance import compare_metrics


@pytest.mark.parametrize("name", ["latency", "p99_latency_ms", "TPOT", "mean_ttft_ms"])
@pytest.mark.parametrize(
    ("value", "verdict"),
    [(200.0, "FAIL"), (106.0, "FAIL"), (105.0, "PASS"), (100.0, "PASS"), (50.0, "PASS")],
)
def test_lower_is_better_regression_uses_candidate_increase(name, value, verdict):
    gates = compare_metrics([Metric(name, 100.0, "ms")], [Metric(name, value, "ms")])
    assert gates[0].verdict == verdict


@pytest.mark.parametrize(
    ("value", "verdict"),
    [(94.0, "FAIL"), (95.0, "PASS"), (100.0, "PASS"), (150.0, "PASS"), (0.0, "FAIL")],
)
def test_throughput_regression_uses_candidate_decrease(value, verdict):
    gates = compare_metrics(
        [Metric("output_throughput", 100.0, "tok/s")],
        [Metric("output_throughput", value, "tok/s")],
    )
    assert gates[0].verdict == verdict


@pytest.mark.parametrize("explicit_side", ["baseline", "candidate", "both"])
@pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
def test_explicit_direction_supports_custom_names_on_either_side(explicit_side, direction):
    baseline = Metric(
        "custom", 100.0, "units",
        direction=direction if explicit_side in ("baseline", "both") else None,
    )
    candidate = Metric(
        "custom", 110.0, "units",
        direction=direction if explicit_side in ("candidate", "both") else None,
    )
    verdict = "PASS" if direction == "higher_is_better" else "FAIL"
    assert compare_metrics([baseline], [candidate])[0].verdict == verdict


def test_explicit_direction_overrides_legacy_name_inference():
    gates = compare_metrics(
        [Metric("latency", 100.0, "score", direction="higher_is_better")],
        [Metric("latency", 110.0, "score")],
    )
    assert gates[0].verdict == "PASS"


@pytest.mark.parametrize("name", ["memory_usage", "throughput_latency", "notthroughput"])
def test_unrecognized_or_ambiguous_name_requires_direction(name):
    gates = compare_metrics([Metric(name, 100.0, "units")], [Metric(name, 110.0, "units")])
    assert gates[0].verdict == "INCOMPARABLE"
    assert "direction" in gates[0].reason


@pytest.mark.parametrize(
    ("baseline_direction", "candidate_direction"),
    [("higher_is_better", "lower_is_better"), ("sideways", None), (None, "")],
)
def test_invalid_or_conflicting_directions_are_incomparable(
    baseline_direction, candidate_direction
):
    gates = compare_metrics(
        [Metric("throughput", 100.0, "tok/s", direction=baseline_direction)],
        [Metric("throughput", 110.0, "tok/s", direction=candidate_direction)],
    )
    assert gates[0].verdict == "INCOMPARABLE"
    assert "direction" in gates[0].reason


def test_missing_baseline_is_unknown():
    gates = compare_metrics([], [Metric("throughput", 100.0, "tok/s")])
    assert gates[0].verdict == "UNKNOWN"
    assert "missing baseline" in gates[0].reason


def test_missing_required_candidate_fails():
    baseline = [Metric("throughput", 100.0, "tok/s"), Metric("ttft", 10.0, "ms")]
    gates = compare_metrics(baseline, [baseline[0]])
    assert [gate.verdict for gate in gates] == ["PASS", "FAIL"]
    assert gates[1].name == "ttft"
    assert "missing candidate" in gates[1].reason


def test_empty_required_candidate_set_fails():
    gates = compare_metrics([Metric("throughput", 100.0, "tok/s")], [])
    assert gates[0].verdict == "FAIL"


def test_both_empty_sets_have_an_explicit_unknown_gate():
    gates = compare_metrics([], [])
    assert len(gates) == 1
    assert gates[0].verdict == "UNKNOWN"


def test_unit_mismatch_is_incomparable_without_implicit_conversion():
    gates = compare_metrics(
        [Metric("ttft", 100.0, "ms")], [Metric("ttft", 0.05, "s")]
    )
    assert gates[0].verdict == "INCOMPARABLE"
    assert "unit mismatch" in gates[0].reason


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_nonfinite_values_cannot_pass(value, side):
    valid = [Metric("latency", 100.0, "ms")]
    invalid = [Metric("latency", value, "ms")]
    baseline, candidate = (invalid, valid) if side == "baseline" else (valid, invalid)
    gates = compare_metrics(baseline, candidate)
    assert gates[0].verdict == "INCOMPARABLE"
    assert "nonfinite" in gates[0].reason


@pytest.mark.parametrize("name", ["throughput", "latency"])
@pytest.mark.parametrize("candidate", [0.0, 1.0])
def test_zero_baseline_has_no_defined_relative_regression(name, candidate):
    gates = compare_metrics(
        [Metric(name, 0.0, "units")], [Metric(name, candidate, "units")]
    )
    assert gates[0].verdict == "INCOMPARABLE"
    assert "zero baseline" in gates[0].reason


def test_zero_candidate_latency_is_an_improvement():
    gates = compare_metrics(
        [Metric("latency", 100.0, "ms")], [Metric("latency", 0.0, "ms")]
    )
    assert gates[0].verdict == "PASS"


def test_labels_are_part_of_required_metric_identity():
    baseline = [Metric("ttft", 10.0, "ms", {"batch": "1"})]
    candidate = [Metric("ttft", 10.0, "ms", {"batch": "2"})]
    gates = compare_metrics(baseline, candidate)
    assert [gate.verdict for gate in gates] == ["UNKNOWN", "FAIL"]
    assert "'batch': '2'" in gates[0].reason
    assert "'batch': '1'" in gates[1].reason


def test_label_order_does_not_change_metric_identity_and_iterators_are_supported():
    gates = compare_metrics(
        iter([Metric("ttft", 10.0, "ms", {"batch": "1", "length": "8"})]),
        iter([Metric("ttft", 10.0, "ms", {"length": "8", "batch": "1"})]),
    )
    assert gates[0].verdict == "PASS"


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_duplicate_keys_cannot_silently_hide_regression(side):
    metrics = [Metric("throughput", 100.0, "tok/s")]
    duplicates = [Metric("throughput", 1.0, "tok/s"), *metrics]
    baseline, candidate = (duplicates, metrics) if side == "baseline" else (metrics, duplicates)
    gates = compare_metrics(baseline, candidate)
    assert gates[0].verdict == "INCOMPARABLE"
    assert "duplicate" in gates[0].reason


@pytest.mark.parametrize("threshold", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_regression_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="max_regression"):
        compare_metrics([], [], max_regression=threshold)


def test_metric_positional_labels_remain_backwards_compatible():
    metric = Metric("throughput", 100.0, "tok/s", {"batch": "1"})
    assert metric.labels == {"batch": "1"}
    assert metric.direction is None
