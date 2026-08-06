from itertools import product

import numpy as np
import pytest

from qecdecoder.benchmark import (
    empirical_logical_error_rate,
    repetition_code_theoretical_logical_error_rate,
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
