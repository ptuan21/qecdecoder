"""Command-line entry points for qecdecoder experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
import yaml

from qecdecoder.benchmark import estimate_crossing_point
from qecdecoder.codes import rotated_surface_code_circuit, rotated_surface_code_circuit_level_noise
from qecdecoder.sweep import CircuitBuilder, SweepPoint, run_mwpm_sweep, run_sweep
from qecdecoder.train import TrainConfig, make_gnn_decode_fn, train_gnn_decoder

_CIRCUIT_BUILDERS: dict[str, CircuitBuilder] = {
    "code_capacity": rotated_surface_code_circuit,
    "circuit_level": rotated_surface_code_circuit_level_noise,
}


def _resolve_circuit_builder(config: dict) -> CircuitBuilder:
    noise_model = config.get("noise_model", "code_capacity")
    try:
        return _CIRCUIT_BUILDERS[noise_model]
    except KeyError:
        raise ValueError(
            f"unknown noise_model {noise_model!r}, expected one of {sorted(_CIRCUIT_BUILDERS)}"
        ) from None


def _resolve_rounds(config: dict, distance: int) -> int:
    """A config-specified `rounds` applies to every distance (Phase 2/3
    behavior). Without one, default to rounds == distance -- the standard
    "memory experiment" convention for circuit-level noise."""
    return config["rounds"] if "rounds" in config else distance


def _resolve_per_distance(config: dict, key: str, distance: int, default: int) -> int:
    """A config value can be a single number (applies to every distance,
    Phase 2-4 behavior) or a {distance: value} mapping for per-distance
    overrides -- e.g. more training shots/capacity for larger distances
    that need it. Missing key or missing distance in a mapping both fall
    back to `default`.
    """
    value = config.get(key, default)
    if isinstance(value, dict):
        return value.get(distance, default)
    return value


def _run_sweep_command(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    circuit_builder = _resolve_circuit_builder(config)
    points: list[SweepPoint] = []
    for distance in config["distances"]:
        points += run_mwpm_sweep(
            [distance],
            config["physical_error_rates"],
            config["num_shots"],
            rounds=_resolve_rounds(config, distance),
            seed=config.get("seed"),
            circuit_builder=circuit_builder,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(points, output_dir / f"{args.name}.csv")
    _plot(points, output_dir / f"{args.name}.png")
    print(f"Wrote {len(points)} points to {output_dir}/{args.name}.{{csv,png}}")
    _print_crossing_estimates(points)


def _print_crossing_estimates(points: Sequence[SweepPoint]) -> None:
    """Rough threshold estimate between each pair of adjacent code distances."""
    distances = sorted({point.distance for point in points})
    for lower, higher in zip(distances, distances[1:]):
        lower_points = sorted(
            (p for p in points if p.distance == lower), key=lambda p: p.physical_error_rate
        )
        higher_points = sorted(
            (p for p in points if p.distance == higher), key=lambda p: p.physical_error_rate
        )
        lower_xs = [p.physical_error_rate for p in lower_points]
        higher_xs = [p.physical_error_rate for p in higher_points]
        if lower_xs != higher_xs:
            continue  # crossing estimate needs a shared physical-error-rate grid
        crossing = estimate_crossing_point(
            lower_xs,
            [p.logical_error_rate for p in lower_points],
            [p.logical_error_rate for p in higher_points],
        )
        if crossing is not None:
            print(f"Estimated threshold between d={lower} and d={higher}: p ~ {crossing:.4f}")
        else:
            print(f"No crossing found between d={lower} and d={higher} in the swept range")


def _write_csv(points: Sequence[SweepPoint], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "distance",
                "physical_error_rate",
                "logical_error_rate",
                "ci_low",
                "ci_high",
                "num_shots",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.distance,
                    point.physical_error_rate,
                    point.logical_error_rate,
                    point.ci_low,
                    point.ci_high,
                    point.num_shots,
                ]
            )


def _plot(points: Sequence[SweepPoint], path: Path) -> None:
    fig, ax = plt.subplots()
    distances = sorted({point.distance for point in points})
    for distance in distances:
        subset = sorted(
            (point for point in points if point.distance == distance),
            key=lambda point: point.physical_error_rate,
        )
        xs = [point.physical_error_rate for point in subset]
        ys = [point.logical_error_rate for point in subset]
        y_err_low = [point.logical_error_rate - point.ci_low for point in subset]
        y_err_high = [point.ci_high - point.logical_error_rate for point in subset]
        ax.errorbar(
            xs, ys, yerr=[y_err_low, y_err_high], label=f"d={distance}", marker="o", capsize=3
        )
    ax.set_xlabel("physical error rate (depolarizing p)")
    ax.set_ylabel("logical error rate")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Rotated surface code: MWPM baseline")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_gnn_benchmark_command(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get("seed", 0)
    eval_physical_error_rates = config["eval_physical_error_rates"]
    num_eval_shots = config["num_eval_shots"]
    circuit_builder = _resolve_circuit_builder(config)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    series: dict[str, list[SweepPoint]] = {}
    for distance in config["distances"]:
        rounds = _resolve_rounds(config, distance)
        train_config = TrainConfig(
            distance=distance,
            physical_error_rates=config["train_physical_error_rates"],
            rounds=rounds,
            num_train_shots_per_rate=_resolve_per_distance(
                config, "num_train_shots_per_rate", distance, 50_000
            ),
            num_val_shots_per_rate=_resolve_per_distance(
                config, "num_val_shots_per_rate", distance, 2_000
            ),
            hidden_channels=_resolve_per_distance(config, "hidden_channels", distance, 32),
            num_layers=_resolve_per_distance(config, "num_layers", distance, 3),
            batch_size=config.get("batch_size", 512),
            epochs=_resolve_per_distance(config, "epochs", distance, 15),
            learning_rate=config.get("learning_rate", 1e-3),
            seed=seed,
        )
        print(
            f"Training GNN for d={distance} (rounds={rounds}, "
            f"shots/rate={train_config.num_train_shots_per_rate}, "
            f"hidden={train_config.hidden_channels}) "
            f"at p={list(train_config.physical_error_rates)}..."
        )
        result = train_gnn_decoder(train_config, device=device, circuit_builder=circuit_builder)
        print(f"  final val logical error rate: {result.val_logical_error_rate:.4f}")
        torch.save(result.model.state_dict(), output_dir / f"{args.name}_d{distance}_gnn.pt")

        gnn_decode_fn = make_gnn_decode_fn(result.model, device=device)
        series[f"GNN d={distance}"] = run_sweep(
            gnn_decode_fn,
            [distance],
            eval_physical_error_rates,
            num_eval_shots,
            rounds=rounds,
            seed=seed + 1_000,
            circuit_builder=circuit_builder,
        )
        series[f"MWPM d={distance}"] = run_mwpm_sweep(
            [distance],
            eval_physical_error_rates,
            num_eval_shots,
            rounds=rounds,
            seed=seed + 2_000,
            circuit_builder=circuit_builder,
        )

    _write_labeled_csv(series, output_dir / f"{args.name}.csv")
    _plot_labeled(series, output_dir / f"{args.name}.png")
    print(f"Wrote {output_dir}/{args.name}.{{csv,png}}")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _write_labeled_csv(series: dict[str, list[SweepPoint]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "decoder",
                "distance",
                "physical_error_rate",
                "logical_error_rate",
                "ci_low",
                "ci_high",
                "num_shots",
            ]
        )
        for label, points in series.items():
            for point in points:
                writer.writerow(
                    [
                        label,
                        point.distance,
                        point.physical_error_rate,
                        point.logical_error_rate,
                        point.ci_low,
                        point.ci_high,
                        point.num_shots,
                    ]
                )


def _plot_labeled(series: dict[str, list[SweepPoint]], path: Path) -> None:
    fig, ax = plt.subplots()
    for label, points in series.items():
        subset = sorted(points, key=lambda point: point.physical_error_rate)
        xs = [point.physical_error_rate for point in subset]
        ys = [point.logical_error_rate for point in subset]
        y_err_low = [point.logical_error_rate - point.ci_low for point in subset]
        y_err_high = [point.ci_high - point.logical_error_rate for point in subset]
        linestyle = "--" if label.startswith("GNN") else "-"
        ax.errorbar(
            xs,
            ys,
            yerr=[y_err_low, y_err_high],
            label=label,
            marker="o",
            capsize=3,
            linestyle=linestyle,
        )
    ax.set_xlabel("physical error rate (depolarizing p)")
    ax.set_ylabel("logical error rate")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Rotated surface code: GNN vs MWPM")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(prog="qecdecoder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep_parser = subparsers.add_parser(
        "sweep", help="Run an MWPM logical-error-rate sweep from a YAML config."
    )
    sweep_parser.add_argument("config", type=str, help="Path to a sweep config YAML file.")
    sweep_parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to write CSV/plot output.",
    )
    sweep_parser.add_argument(
        "--name", type=str, default="sweep", help="Base filename for output CSV/plot."
    )
    sweep_parser.set_defaults(func=_run_sweep_command)

    gnn_parser = subparsers.add_parser(
        "gnn-benchmark",
        help="Train a GNN decoder per distance and benchmark it against MWPM.",
    )
    gnn_parser.add_argument("config", type=str, help="Path to a GNN benchmark config YAML file.")
    gnn_parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to write CSV/plot/model checkpoint output.",
    )
    gnn_parser.add_argument(
        "--name", type=str, default="gnn_benchmark", help="Base filename for output CSV/plot."
    )
    gnn_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="torch device: 'auto' (use cuda if available), 'cpu', or 'cuda'.",
    )
    gnn_parser.set_defaults(func=_run_gnn_benchmark_command)

    args = parser.parse_args()
    args.func(args)
