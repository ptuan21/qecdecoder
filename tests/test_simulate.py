from qecdecoder.codes import repetition_code_circuit
from qecdecoder.simulate import sample_dataset


def test_sample_dataset_shapes() -> None:
    circuit = repetition_code_circuit(distance=5, rounds=1, physical_error_rate=0.1)
    dataset = sample_dataset(circuit, num_shots=100, seed=42)
    assert dataset.detector_syndromes.shape == (100, circuit.num_detectors)
    assert dataset.observable_flips.shape == (100, circuit.num_observables)
    assert dataset.detector_syndromes.dtype == bool
    assert dataset.observable_flips.dtype == bool


def test_sample_dataset_is_deterministic_given_seed() -> None:
    circuit = repetition_code_circuit(distance=5, rounds=1, physical_error_rate=0.2)
    first = sample_dataset(circuit, num_shots=50, seed=123)
    second = sample_dataset(circuit, num_shots=50, seed=123)
    assert (first.detector_syndromes == second.detector_syndromes).all()
    assert (first.observable_flips == second.observable_flips).all()


def test_zero_noise_circuit_has_no_flips() -> None:
    circuit = repetition_code_circuit(distance=5, rounds=1, physical_error_rate=0.0)
    dataset = sample_dataset(circuit, num_shots=200, seed=7)
    assert not dataset.detector_syndromes.any()
    assert not dataset.observable_flips.any()
