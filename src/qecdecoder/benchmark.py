"""Logical error rate estimation and theoretical baselines."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def empirical_logical_error_rate(
    predicted_flips: np.ndarray, true_flips: np.ndarray
) -> float:
    """Fraction of shots where the decoder's prediction disagrees with the truth.

    A shot with multiple observables counts as one logical failure if any
    observable is predicted incorrectly.
    """
    if predicted_flips.shape != true_flips.shape:
        raise ValueError(
            f"shape mismatch: predicted {predicted_flips.shape} vs true {true_flips.shape}"
        )
    disagreement = np.any(predicted_flips != true_flips, axis=1)
    return float(np.mean(disagreement))


def wilson_confidence_interval(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default: 95% CI).

    More reliable than a naive normal-approximation interval when the
    observed proportion is close to 0 or 1, which is the common case for
    logical error rates well below threshold.
    """
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be in [0, {total}], got {successes}")
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    half_width = (z / denom) * ((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5)
    return max(0.0, center - half_width), min(1.0, center + half_width)


def estimate_crossing_point(
    physical_error_rates: np.ndarray,
    logical_error_rates_a: np.ndarray,
    logical_error_rates_b: np.ndarray,
) -> float | None:
    """Rough estimate of where two logical-error-rate curves cross, via
    linear interpolation of their difference.

    Intended as a quick threshold estimate from two code distances (below
    the crossing, the larger distance has lower logical error rate; above
    it, the ordering flips). This is not a rigorous finite-size-scaling
    threshold fit -- that needs more distances and more shots per point.
    Returns None if the two curves don't cross within the given range.
    """
    physical_error_rates = np.asarray(physical_error_rates, dtype=float)
    diffs = np.asarray(logical_error_rates_a, dtype=float) - np.asarray(
        logical_error_rates_b, dtype=float
    )
    sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    i = int(sign_changes[0])
    x0, x1 = physical_error_rates[i], physical_error_rates[i + 1]
    d0, d1 = diffs[i], diffs[i + 1]
    t = -d0 / (d1 - d0)
    return float(x0 + t * (x1 - x0))


def repetition_code_theoretical_logical_error_rate(
    distance: int, bit_flip_rate: float
) -> float:
    """Closed-form logical failure probability for a distance-d repetition code.

    Assumes independent per-qubit bit-flip noise and optimal decoding
    (MWPM on a 1D chain is equivalent to majority vote): failure occurs
    exactly when more than half the data qubits flip. Valid for odd
    distance only.
    """
    if distance % 2 == 0:
        raise ValueError(f"closed-form formula requires odd distance, got {distance}")
    if not 0.0 <= bit_flip_rate <= 1.0:
        raise ValueError(f"bit_flip_rate must be in [0, 1], got {bit_flip_rate}")
    majority_threshold = (distance + 1) // 2
    return float(binom.sf(majority_threshold - 1, distance, bit_flip_rate))
