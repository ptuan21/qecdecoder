"""PyMatching-based MWPM baseline decoder."""

from __future__ import annotations

import numpy as np
import pymatching
import stim


def build_matching(circuit: stim.Circuit) -> pymatching.Matching:
    """Build an MWPM decoder from a circuit's detector error model."""
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def decode_batch(matching: pymatching.Matching, detector_syndromes: np.ndarray) -> np.ndarray:
    """Decode a batch of detector syndromes into predicted observable flips."""
    return matching.decode_batch(detector_syndromes)
