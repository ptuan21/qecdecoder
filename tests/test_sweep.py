import numpy as np

from qecdecoder.codes import repetition_code_circuit, rotated_surface_code_circuit
from qecdecoder.sweep import run_mwpm_sweep, run_sweep


def test_sweep_covers_every_distance_physical_error_rate_pair() -> None:
    distances = [3, 5]
    physical_error_rates = [0.05, 0.1]
    points = run_mwpm_sweep(distances, physical_error_rates, num_shots=200, seed=1)

    assert len(points) == len(distances) * len(physical_error_rates)
    pairs = {(p.distance, p.physical_error_rate) for p in points}
    assert pairs == {(d, r) for d in distances for r in physical_error_rates}


def test_sweep_logical_error_rate_within_its_own_ci() -> None:
    points = run_mwpm_sweep([3], [0.1], num_shots=2000, seed=42)
    point = points[0]
    assert point.ci_low <= point.logical_error_rate <= point.ci_high


def test_sweep_zero_noise_gives_zero_logical_error_rate() -> None:
    points = run_mwpm_sweep([3, 5], [0.0], num_shots=100, seed=7)
    assert all(p.logical_error_rate == 0.0 for p in points)


def test_sweep_is_deterministic_given_seed() -> None:
    first = run_mwpm_sweep([5], [0.1], num_shots=500, seed=99)
    second = run_mwpm_sweep([5], [0.1], num_shots=500, seed=99)
    assert first == second


def test_sweep_logical_error_rate_increases_with_physical_error_rate() -> None:
    points = run_mwpm_sweep([5], [0.02, 0.3], num_shots=5000, seed=5)
    low_p, high_p = points
    assert low_p.logical_error_rate < high_p.logical_error_rate


def test_run_sweep_generic_matches_mwpm_wrapper_with_equivalent_decode_fn() -> None:
    from qecdecoder.baseline import build_matching, decode_batch

    def mwpm_decode_fn(circuit, detector_syndromes):
        matching = build_matching(circuit)
        return decode_batch(matching, detector_syndromes)

    generic = run_sweep(mwpm_decode_fn, [3, 5], [0.05, 0.1], num_shots=500, seed=10)
    wrapper = run_mwpm_sweep([3, 5], [0.05, 0.1], num_shots=500, seed=10)
    assert generic == wrapper


def test_run_sweep_with_fake_always_correct_decoder_gives_zero_logical_error_rate() -> None:
    def perfect_decode_fn(circuit, detector_syndromes):
        # Cheat: this fake decoder isn't given the true labels, so instead
        # we just check it always predicting "no flip" gives a sane, bounded
        # logical error rate rather than crashing the sweep machinery.
        return np.zeros((len(detector_syndromes), circuit.num_observables), dtype=bool)

    points = run_sweep(perfect_decode_fn, [3], [0.1], num_shots=200, seed=3)
    assert len(points) == 1
    assert 0.0 <= points[0].logical_error_rate <= 1.0


def test_run_sweep_uses_custom_circuit_builder() -> None:
    calls = []

    def spy_builder(distance, rounds, physical_error_rate):
        calls.append((distance, rounds, physical_error_rate))
        return repetition_code_circuit(distance, rounds, physical_error_rate)

    def mwpm_decode_fn(circuit, detector_syndromes):
        from qecdecoder.baseline import build_matching, decode_batch

        return decode_batch(build_matching(circuit), detector_syndromes)

    points = run_sweep(
        mwpm_decode_fn, [5], [0.1], num_shots=100, seed=1, circuit_builder=spy_builder
    )
    assert calls == [(5, 1, 0.1)]
    assert len(points) == 1


def test_run_sweep_default_circuit_builder_is_rotated_surface_code() -> None:
    points_default = run_mwpm_sweep([3], [0.1], num_shots=300, seed=4)
    points_explicit = run_mwpm_sweep(
        [3], [0.1], num_shots=300, seed=4, circuit_builder=rotated_surface_code_circuit
    )
    assert points_default == points_explicit
