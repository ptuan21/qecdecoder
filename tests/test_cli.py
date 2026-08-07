from pathlib import Path

import pytest
import yaml

from qecdecoder.cli import _resolve_circuit_builder, _resolve_per_distance, _resolve_rounds, main
from qecdecoder.codes import rotated_surface_code_circuit, rotated_surface_code_circuit_level_noise


def test_sweep_command_writes_csv_and_plot(tmp_path: Path, monkeypatch) -> None:
    config = {
        "distances": [3, 5],
        "physical_error_rates": [0.05, 0.15],
        "num_shots": 200,
        "rounds": 1,
        "seed": 1,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        ["qecdecoder", "sweep", str(config_path), "--output-dir", str(output_dir), "--name", "t"],
    )
    main()

    assert (output_dir / "t.csv").exists()
    assert (output_dir / "t.png").exists()
    csv_lines = (output_dir / "t.csv").read_text().splitlines()
    assert len(csv_lines) == 1 + len(config["distances"]) * len(config["physical_error_rates"])


def test_resolve_circuit_builder_defaults_to_code_capacity() -> None:
    assert _resolve_circuit_builder({}) is rotated_surface_code_circuit


def test_resolve_circuit_builder_selects_circuit_level() -> None:
    builder = _resolve_circuit_builder({"noise_model": "circuit_level"})
    assert builder is rotated_surface_code_circuit_level_noise


def test_resolve_circuit_builder_rejects_unknown_noise_model() -> None:
    with pytest.raises(ValueError):
        _resolve_circuit_builder({"noise_model": "not_a_real_model"})


def test_resolve_rounds_uses_config_value_for_every_distance_when_present() -> None:
    config = {"rounds": 1}
    assert _resolve_rounds(config, distance=3) == 1
    assert _resolve_rounds(config, distance=5) == 1


def test_resolve_rounds_defaults_to_distance_when_absent() -> None:
    config = {}
    assert _resolve_rounds(config, distance=3) == 3
    assert _resolve_rounds(config, distance=5) == 5


def test_resolve_per_distance_uses_scalar_for_every_distance() -> None:
    config = {"hidden_channels": 32}
    assert _resolve_per_distance(config, "hidden_channels", 3, 16) == 32
    assert _resolve_per_distance(config, "hidden_channels", 9, 16) == 32


def test_resolve_per_distance_uses_mapping_override() -> None:
    config = {"hidden_channels": {7: 48, 9: 64}}
    assert _resolve_per_distance(config, "hidden_channels", 7, 32) == 48
    assert _resolve_per_distance(config, "hidden_channels", 9, 32) == 64


def test_resolve_per_distance_mapping_falls_back_to_default_for_missing_distance() -> None:
    config = {"hidden_channels": {9: 64}}
    assert _resolve_per_distance(config, "hidden_channels", 3, 32) == 32


def test_resolve_per_distance_falls_back_to_default_when_key_absent() -> None:
    assert _resolve_per_distance({}, "hidden_channels", 3, 32) == 32


def test_sweep_command_with_circuit_level_noise_model(tmp_path: Path, monkeypatch) -> None:
    config = {
        "noise_model": "circuit_level",
        "distances": [3],
        "physical_error_rates": [0.005, 0.01],
        "num_shots": 500,
        "seed": 1,
        # rounds intentionally omitted -> defaults to rounds == distance
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        ["qecdecoder", "sweep", str(config_path), "--output-dir", str(output_dir), "--name", "t"],
    )
    main()

    csv_lines = (output_dir / "t.csv").read_text().splitlines()
    assert len(csv_lines) == 1 + len(config["distances"]) * len(config["physical_error_rates"])


def test_gnn_benchmark_command_supports_per_distance_overrides(tmp_path: Path, monkeypatch) -> None:
    config = {
        "distances": [3, 5],
        "train_physical_error_rates": [0.1, 0.2],
        "eval_physical_error_rates": [0.1, 0.2],
        "num_train_shots_per_rate": {3: 100, 5: 200},
        "num_val_shots_per_rate": 50,
        "num_eval_shots": 100,
        "hidden_channels": {5: 16},
        "num_layers": 2,
        "batch_size": 32,
        "epochs": 1,
        "seed": 1,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "qecdecoder",
            "gnn-benchmark",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--name",
            "t",
            "--device",
            "cpu",
        ],
    )
    main()

    assert (output_dir / "t.csv").exists()
    assert (output_dir / "t_d3_gnn.pt").exists()
    assert (output_dir / "t_d5_gnn.pt").exists()
