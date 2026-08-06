from pathlib import Path

import yaml

from qecdecoder.cli import main


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
