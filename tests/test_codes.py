import pytest
import stim

from qecdecoder.codes import (
    repetition_code_circuit,
    rotated_surface_code_circuit,
    rotated_surface_code_circuit_level_noise,
)


def test_repetition_code_circuit_returns_stim_circuit() -> None:
    circuit = repetition_code_circuit(distance=5, rounds=1, physical_error_rate=0.1)
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_observables == 1


def test_rotated_surface_code_circuit_returns_stim_circuit() -> None:
    circuit = rotated_surface_code_circuit(distance=3, rounds=1, physical_error_rate=0.05)
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_observables == 1


def test_rotated_surface_code_circuit_level_noise_returns_stim_circuit() -> None:
    circuit = rotated_surface_code_circuit_level_noise(
        distance=3, rounds=3, physical_error_rate=0.005
    )
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_observables == 1
    # Multi-round circuit-level noise needs more detectors than a single
    # code-capacity round (one per stabilizer per round, roughly).
    assert circuit.num_detectors > 8


@pytest.mark.parametrize("bad_distance", [0, 1, 2, 4, -3])
def test_rejects_invalid_distance(bad_distance: int) -> None:
    with pytest.raises(ValueError):
        repetition_code_circuit(distance=bad_distance, rounds=1, physical_error_rate=0.1)
    with pytest.raises(ValueError):
        rotated_surface_code_circuit(distance=bad_distance, rounds=1, physical_error_rate=0.1)
    with pytest.raises(ValueError):
        rotated_surface_code_circuit_level_noise(
            distance=bad_distance, rounds=1, physical_error_rate=0.1
        )


def test_rejects_invalid_rounds() -> None:
    with pytest.raises(ValueError):
        repetition_code_circuit(distance=3, rounds=0, physical_error_rate=0.1)
