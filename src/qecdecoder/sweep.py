"""Sweep the MWPM baseline's logical error rate over code distance and
physical error rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qecdecoder.baseline import build_matching, decode_batch
from qecdecoder.benchmark import empirical_logical_error_rate, wilson_confidence_interval
from qecdecoder.codes import rotated_surface_code_circuit
from qecdecoder.simulate import sample_dataset


@dataclass(frozen=True)
class SweepPoint:
    """One (distance, physical_error_rate) result of an MWPM sweep."""

    distance: int
    physical_error_rate: float
    logical_error_rate: float
    ci_low: float
    ci_high: float
    num_shots: int


def run_mwpm_sweep(
    distances: Sequence[int],
    physical_error_rates: Sequence[float],
    num_shots: int,
    *,
    rounds: int = 1,
    seed: int | None = None,
) -> list[SweepPoint]:
    """Run the MWPM baseline over every (distance, physical_error_rate) pair.

    Uses a rotated surface code with code-capacity-style depolarizing
    noise. Each (distance, physical_error_rate) combination gets its own
    seeded sample so results are reproducible.
    """
    points: list[SweepPoint] = []
    combo_index = 0
    for distance in distances:
        for physical_error_rate in physical_error_rates:
            circuit = rotated_surface_code_circuit(
                distance=distance, rounds=rounds, physical_error_rate=physical_error_rate
            )
            shot_seed = None if seed is None else seed + combo_index
            combo_index += 1
            dataset = sample_dataset(circuit, num_shots=num_shots, seed=shot_seed)
            matching = build_matching(circuit)
            predictions = decode_batch(matching, dataset.detector_syndromes)

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
