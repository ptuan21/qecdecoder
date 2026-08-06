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
