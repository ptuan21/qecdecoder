from qecdecoder.sweep import run_mwpm_sweep


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
