import numpy as np

from qecdecoder.baseline import build_matching, decode_batch
from qecdecoder.codes import repetition_code_circuit
from qecdecoder.simulate import sample_dataset


def test_decode_batch_output_shape() -> None:
    circuit = repetition_code_circuit(distance=5, rounds=1, physical_error_rate=0.1)
    dataset = sample_dataset(circuit, num_shots=100, seed=1)
    matching = build_matching(circuit)
    predictions = decode_batch(matching, dataset.detector_syndromes)
    assert predictions.shape == dataset.observable_flips.shape


def test_zero_noise_decodes_to_no_flips() -> None:
    circuit = repetition_code_circuit(distance=5, rounds=1, physical_error_rate=0.0)
    dataset = sample_dataset(circuit, num_shots=50, seed=2)
    matching = build_matching(circuit)
    predictions = decode_batch(matching, dataset.detector_syndromes)
    assert not predictions.any()


def test_all_zero_syndrome_decodes_to_no_flips() -> None:
    circuit = repetition_code_circuit(distance=7, rounds=1, physical_error_rate=0.3)
    matching = build_matching(circuit)
    empty_syndrome = np.zeros((1, circuit.num_detectors), dtype=bool)
    predictions = decode_batch(matching, empty_syndrome)
    assert not predictions.any()
