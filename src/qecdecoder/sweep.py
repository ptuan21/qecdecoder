"""Sweep a decoder's logical error rate over code distance and physical
error rate.
"""

from __future__ import annotations

import stim
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from qecdecoder.baseline import build_matching, decode_batch
from qecdecoder.benchmark import empirical_logical_error_rate, wilson_confidence_interval
from qecdecoder.codes import rotated_surface_code_circuit
from qecdecoder.simulate import sample_dataset

DecodeFn = Callable[[stim.Circuit, np.ndarray], np.ndarray]
"""A decoder: (circuit, detector_syndromes) -> predicted observable flips."""

CircuitBuilder = Callable[[int, int, float], stim.Circuit]
"""(distance, rounds, physical_error_rate) -> circuit, e.g. rotated_surface_code_circuit."""


@dataclass(frozen=True)
class SweepPoint:
    """One (distance, physical_error_rate) result of a decoder sweep."""

    distance: int
    physical_error_rate: float
    logical_error_rate: float
    ci_low: float
    ci_high: float
    num_shots: int


def run_sweep(
    decode_fn: DecodeFn,
    distances: Sequence[int],
    physical_error_rates: Sequence[float],
    num_shots: int,
    *,
    rounds: int = 1,
    seed: int | None = None,
    circuit_builder: CircuitBuilder = rotated_surface_code_circuit,
) -> list[SweepPoint]:
    """Run `decode_fn` over every (distance, physical_error_rate) pair.

    Each (distance, physical_error_rate) combination gets its own seeded
    sample so results are reproducible. `circuit_builder` defaults to the
    code-capacity-style rotated surface code; pass a different builder
    (e.g. `codes.rotated_surface_code_circuit_level_noise`) for other noise
    models.
    """
    points: list[SweepPoint] = []
    combo_index = 0
    for distance in distances:
        for physical_error_rate in physical_error_rates:
            circuit = circuit_builder(distance, rounds, physical_error_rate)
            shot_seed = None if seed is None else seed + combo_index
            combo_index += 1
            dataset = sample_dataset(circuit, num_shots=num_shots, seed=shot_seed)
            predictions = decode_fn(circuit, dataset.detector_syndromes)

            logical_error_rate = empirical_logical_error_rate(
                predictions, dataset.observable_flips
            )
            failures = round(logical_error_rate * num_shots)
            ci_low, ci_high = wilson_confidence_interval(failures, num_shots)

            points.append(
                SweepPoint(
                    distance=distance,
                    physical_error_rate=physical_error_rate,
                    logical_error_rate=logical_error_rate,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    num_shots=num_shots,
                )
            )
    return points


def _mwpm_decode_fn(circuit: stim.Circuit, detector_syndromes: np.ndarray) -> np.ndarray:
    matching = build_matching(circuit)
    return decode_batch(matching, detector_syndromes)


def run_mwpm_sweep(
    distances: Sequence[int],
    physical_error_rates: Sequence[float],
    num_shots: int,
    *,
    rounds: int = 1,
    seed: int | None = None,
    circuit_builder: CircuitBuilder = rotated_surface_code_circuit,
) -> list[SweepPoint]:
    """Run the MWPM baseline over every (distance, physical_error_rate) pair."""
    return run_sweep(
        _mwpm_decode_fn,
        distances,
        physical_error_rates,
        num_shots,
        rounds=rounds,
        seed=seed,
        circuit_builder=circuit_builder,
    )
