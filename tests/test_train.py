import numpy as np

from qecdecoder.train import TrainConfig, make_gnn_decode_fn, train_gnn_decoder
from qecdecoder.codes import rotated_surface_code_circuit
from qecdecoder.simulate import sample_dataset


def _tiny_config(**overrides) -> TrainConfig:
    defaults = dict(
        distance=3,
        physical_error_rates=[0.1, 0.2],
        num_train_shots_per_rate=150,
        num_val_shots_per_rate=50,
        hidden_channels=8,
        num_layers=2,
        batch_size=64,
        epochs=2,
        seed=0,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_train_gnn_decoder_runs_and_produces_history() -> None:
    result = train_gnn_decoder(_tiny_config())
    assert len(result.history) == 2
    for stats in result.history:
        assert np.isfinite(stats.train_loss)
        assert np.isfinite(stats.val_loss)
        assert 0.0 <= stats.val_logical_error_rate <= 1.0
    assert result.val_logical_error_rate == result.history[-1].val_logical_error_rate


def test_train_gnn_decoder_is_deterministic_given_seed() -> None:
    result_a = train_gnn_decoder(_tiny_config(seed=1))
    result_b = train_gnn_decoder(_tiny_config(seed=1))
    losses_a = [s.train_loss for s in result_a.history]
    losses_b = [s.train_loss for s in result_b.history]
    assert losses_a == losses_b


def test_make_gnn_decode_fn_output_shape() -> None:
    result = train_gnn_decoder(_tiny_config())
    decode_fn = make_gnn_decode_fn(result.model)

    circuit = rotated_surface_code_circuit(distance=3, rounds=1, physical_error_rate=0.15)
    dataset = sample_dataset(circuit, num_shots=20, seed=99)
    predictions = decode_fn(circuit, dataset.detector_syndromes)
    assert predictions.shape == dataset.observable_flips.shape
    assert predictions.dtype == bool


def test_make_gnn_decode_fn_handles_zero_edge_graph_at_zero_noise() -> None:
    """physical_error_rate=0 has no error mechanisms at all, so the
    decoding graph has zero edges -- entirely out-of-distribution for the
    trained model. The only correct (and well-defined) answer is "no flip".
    """
    result = train_gnn_decoder(_tiny_config())
    decode_fn = make_gnn_decode_fn(result.model)

    circuit = rotated_surface_code_circuit(distance=3, rounds=1, physical_error_rate=0.0)
    dataset = sample_dataset(circuit, num_shots=20, seed=99)
    predictions = decode_fn(circuit, dataset.detector_syndromes)
    assert not predictions.any()
