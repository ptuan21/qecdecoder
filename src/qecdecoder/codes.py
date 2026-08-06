"""Stim circuit builders for the codes studied in this project."""

from __future__ import annotations

import stim


def repetition_code_circuit(
    distance: int, rounds: int, physical_error_rate: float
) -> stim.Circuit:
    """Build a repetition-code memory circuit with data-qubit depolarizing noise."""
    _validate_distance(distance)
    _validate_rounds(rounds)
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=physical_error_rate,
    )


def rotated_surface_code_circuit(
    distance: int, rounds: int, physical_error_rate: float
) -> stim.Circuit:
    """Build a rotated surface-code memory-Z circuit with data-qubit depolarizing noise."""
    _validate_distance(distance)
    _validate_rounds(rounds)
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=physical_error_rate,
    )


def _validate_distance(distance: int) -> None:
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {distance}")


def _validate_rounds(rounds: int) -> None:
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
