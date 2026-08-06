from pathlib import Path

import pytest
import yaml

from qecdecoder.cli import _resolve_circuit_builder, _resolve_rounds, main
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
