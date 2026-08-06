"""Noise model helpers for code-capacity-style Stim circuits."""

from __future__ import annotations


def code_capacity_effective_bit_flip_rate(depolarizing_rate: float) -> float:
    """Effective bit-flip probability seen by a CSS code's X-type stabilizers
    under single-qubit depolarizing noise.

    Under DEPOLARIZE1(p), each qubit independently gets X, Y, or Z with
    probability p/3 each. X and Y both flip the value read out by a
    Z-basis logical/stabilizer; Z alone is invisible to it. The two error
    channels decouple (CSS code property), so the effective bit-flip rate
    seen by the X-detecting side of the code is 2p/3.
    """
    if not 0.0 <= depolarizing_rate <= 1.0:
        raise ValueError(f"depolarizing_rate must be in [0, 1], got {depolarizing_rate}")
    return 2.0 * depolarizing_rate / 3.0
