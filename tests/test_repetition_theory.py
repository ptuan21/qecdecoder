"""Phase 1 milestone: validate the full simulate -> decode -> benchmark
pipeline against the closed-form repetition-code theory before moving on
to the surface code, where no such simple closed form exists.
"""

from qecdecoder.baseline import build_matching, decode_batch
from qecdecoder.benchmark import (
    empirical_logical_error_rate,
    repetition_code_theoretical_logical_error_rate,
)
from qecdecoder.codes import repetition_code_circuit
from qecdecoder.noise import code_capacity_effective_bit_flip_rate
from qecdecoder.simulate import sample_dataset


def test_mwpm_decoding_matches_repetition_code_theory() -> None:
    distance = 5
    depolarizing_rate = 0.3
    num_shots = 40_000

    circuit = repetition_code_circuit(
        distance=distance, rounds=1, physical_error_rate=depolarizing_rate
    )
    dataset = sample_dataset(circuit, num_shots=num_shots, seed=2024)
    matching = build_matching(circuit)
    predictions = decode_batch(matching, dataset.detector_syndromes)

    empirical = empirical_logical_error_rate(predictions, dataset.observable_flips)

    bit_flip_rate = code_capacity_effective_bit_flip_rate(depolarizing_rate)
    theoretical = repetition_code_theoretical_logical_error_rate(distance, bit_flip_rate)

    # Binomial standard error at this sample size, with a generous 6-sigma
    # margin plus a floor so the test is meaningful but not flaky.
    std_error = (theoretical * (1 - theoretical) / num_shots) ** 0.5
    tolerance = max(6 * std_error, 0.01)

    assert abs(empirical - theoretical) < tolerance
