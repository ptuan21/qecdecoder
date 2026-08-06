"""Command-line entry points for qecdecoder experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import yaml

from qecdecoder.benchmark import estimate_crossing_point
from qecdecoder.sweep import SweepPoint, run_mwpm_sweep


def _run_sweep_command(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    points = run_mwpm_sweep(
        distances=config["distances"],
        physical_error_rates=config["physical_error_rates"],
        num_shots=config["num_shots"],
        rounds=config.get("rounds", 1),
        seed=config.get("seed"),
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

    args = parser.parse_args()
    args.func(args)
