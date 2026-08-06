from itertools import product

import numpy as np
import pytest

from qecdecoder.benchmark import (
    empirical_logical_error_rate,
    estimate_crossing_point,
    repetition_code_theoretical_logical_error_rate,
    wilson_confidence_interval,
)


def test_theoretical_formula_matches_brute_force_enumeration() -> None:
    distance = 5
    p = 0.15
    threshold = (distance + 1) // 2
    expected = 0.0
    for bits in product([0, 1], repeat=distance):
        k = sum(bits)
        if k >= threshold:
            expected += (p**k) * ((1 - p) ** (distance - k))
    actual = repetition_code_theoretical_logical_error_rate(distance, p)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_theoretical_formula_edge_cases() -> None:
    assert repetition_code_theoretical_logical_error_rate(5, 0.0) == 0.0
    assert repetition_code_theoretical_logical_error_rate(5, 1.0) == pytest.approx(1.0)


def test_theoretical_formula_monotonic_in_p() -> None:
    distance = 7
    rates = [
        repetition_code_theoretical_logical_error_rate(distance, p)
        for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    assert rates == sorted(rates)


def test_theoretical_formula_rejects_even_distance() -> None:
    with pytest.raises(ValueError):
        repetition_code_theoretical_logical_error_rate(4, 0.1)


def test_empirical_logical_error_rate_all_correct() -> None:
    predicted = np.zeros((10, 1), dtype=bool)
    true = np.zeros((10, 1), dtype=bool)
    assert empirical_logical_error_rate(predicted, true) == 0.0


def test_empirical_logical_error_rate_all_wrong() -> None:
    predicted = np.zeros((10, 1), dtype=bool)
    true = np.ones((10, 1), dtype=bool)
    assert empirical_logical_error_rate(predicted, true) == 1.0


def test_empirical_logical_error_rate_shape_mismatch_raises() -> None:
    predicted = np.zeros((10, 1), dtype=bool)
    true = np.zeros((5, 1), dtype=bool)
    with pytest.raises(ValueError):
        empirical_logical_error_rate(predicted, true)


def test_wilson_interval_contains_point_estimate() -> None:
    ci_low, ci_high = wilson_confidence_interval(successes=50, total=1000)
    assert ci_low < 0.05 < ci_high


def test_wilson_interval_shrinks_with_more_data() -> None:
    narrow_low, narrow_high = wilson_confidence_interval(successes=500, total=10_000)
    wide_low, wide_high = wilson_confidence_interval(successes=50, total=1_000)
    assert (narrow_high - narrow_low) < (wide_high - wide_low)


def test_wilson_interval_bounds_are_valid_probabilities() -> None:
    ci_low, ci_high = wilson_confidence_interval(successes=0, total=100)
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_wilson_interval_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_confidence_interval(successes=0, total=0)
    with pytest.raises(ValueError):
        wilson_confidence_interval(successes=11, total=10)


def test_estimate_crossing_point_finds_linear_crossing() -> None:
    xs = np.array([0.0, 0.1, 0.2, 0.3])
    a = np.array([0.0, 0.1, 0.2, 0.3])  # increasing
    b = np.array([0.3, 0.2, 0.1, 0.0])  # decreasing, crosses a between x=0.1 and 0.2
    crossing = estimate_crossing_point(xs, a, b)
    assert crossing == pytest.approx(0.15, abs=1e-9)


def test_estimate_crossing_point_returns_none_when_no_crossing() -> None:
    xs = np.array([0.0, 0.1, 0.2])
    a = np.array([0.0, 0.1, 0.2])
    b = np.array([1.0, 1.1, 1.2])
    assert estimate_crossing_point(xs, a, b) is None
