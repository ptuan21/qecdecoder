"""Sampling syndromes and logical-observable flips from Stim circuits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass(frozen=True)
class SampledDataset:
    """Batch of sampled detector syndromes and true logical-observable flips."""

    detector_syndromes: np.ndarray
    """Shape (num_shots, num_detectors), dtype bool."""

    observable_flips: np.ndarray
    """Shape (num_shots, num_observables), dtype bool."""


def sample_dataset(
    circuit: stim.Circuit, num_shots: int, *, seed: int | None = None
) -> SampledDataset:
    """Sample detector syndromes and true logical-observable flips from a circuit."""
    if num_shots < 1:
        raise ValueError(f"num_shots must be >= 1, got {num_shots}")
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(num_shots, separate_observables=True)
    return SampledDataset(detector_syndromes=detectors, observable_flips=observables)
