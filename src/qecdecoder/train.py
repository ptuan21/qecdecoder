"""Train the GNN decoder for a single code distance at a fixed physical
error rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim
import torch
from torch import nn

from qecdecoder.benchmark import empirical_logical_error_rate
from qecdecoder.codes import rotated_surface_code_circuit
from qecdecoder.graph import DecodingGraph, build_decoding_graph
from qecdecoder.model import GNNDecoder, syndromes_to_model_inputs
from qecdecoder.simulate import SampledDataset, sample_dataset
from qecdecoder.sweep import DecodeFn


@dataclass(frozen=True)
class TrainConfig:
    distance: int
    physical_error_rate: float
    rounds: int = 1
    num_train_shots: int = 50_000
    num_val_shots: int = 10_000
    hidden_channels: int = 32
    num_layers: int = 3
    batch_size: int = 512
    epochs: int = 15
    learning_rate: float = 1e-3
    seed: int = 0


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_logical_error_rate: float


@dataclass(frozen=True)
class TrainResult:
    model: GNNDecoder
    graph: DecodingGraph
    history: list[EpochStats] = field(default_factory=list)

    @property
    def val_logical_error_rate(self) -> float:
        return self.history[-1].val_logical_error_rate


def train_gnn_decoder(config: TrainConfig, *, device: torch.device | None = None) -> TrainResult:
    """Train a GNNDecoder on syndromes sampled at `config.physical_error_rate`."""
    device = device or torch.device("cpu")
    circuit = rotated_surface_code_circuit(
        config.distance, config.rounds, config.physical_error_rate
    )
    graph = build_decoding_graph(circuit)

    train_data = sample_dataset(circuit, config.num_train_shots, seed=config.seed)
    val_data = sample_dataset(circuit, config.num_val_shots, seed=config.seed + 1)

    torch.manual_seed(config.seed)  # seeds GNNDecoder's weight initialization
    model = GNNDecoder(config.hidden_channels, config.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(config.seed)

    history: list[EpochStats] = []
    for epoch in range(config.epochs):
        train_loss = _train_one_epoch(
            model, optimizer, loss_fn, graph, train_data, config.batch_size, rng
        )
        val_loss, val_ler = _evaluate(model, loss_fn, graph, val_data, config.batch_size, device)
        history.append(
            EpochStats(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_logical_error_rate=val_ler)
        )

    return TrainResult(model=model, graph=graph, history=history)


def _train_one_epoch(
    model: GNNDecoder,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    graph: DecodingGraph,
    train_data: SampledDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    device = next(model.parameters()).device
    model.train()
    num_train = len(train_data.detector_syndromes)
    perm = rng.permutation(num_train)
    total_loss = 0.0
    for start in range(0, num_train, batch_size):
        idx = perm[start : start + batch_size]
        x, edge_index, edge_attr, batch_vec = syndromes_to_model_inputs(
            graph, train_data.detector_syndromes[idx]
        )
        y = torch.from_numpy(train_data.observable_flips[idx, 0].astype(np.float32))
        x, edge_index, edge_attr, batch_vec, y = (
            t.to(device) for t in (x, edge_index, edge_attr, batch_vec, y)
        )
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_attr, batch_vec)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    return total_loss / num_train


def _evaluate(
    model: GNNDecoder,
    loss_fn: nn.Module,
    graph: DecodingGraph,
    dataset: SampledDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    num_shots = len(dataset.detector_syndromes)
    total_loss = 0.0
    predictions = np.zeros((num_shots, 1), dtype=bool)
    with torch.no_grad():
        for start in range(0, num_shots, batch_size):
            chunk = slice(start, start + batch_size)
            x, edge_index, edge_attr, batch_vec = syndromes_to_model_inputs(
                graph, dataset.detector_syndromes[chunk]
            )
            y = torch.from_numpy(dataset.observable_flips[chunk, 0].astype(np.float32))
            x, edge_index, edge_attr, batch_vec, y = (
                t.to(device) for t in (x, edge_index, edge_attr, batch_vec, y)
            )
            logits = model(x, edge_index, edge_attr, batch_vec)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.shape[0]
            predictions[chunk, 0] = (torch.sigmoid(logits) > 0.5).cpu().numpy()
    val_loss = total_loss / num_shots
    val_ler = empirical_logical_error_rate(predictions, dataset.observable_flips)
    return val_loss, val_ler


def make_gnn_decode_fn(
    model: GNNDecoder, *, device: torch.device | None = None, batch_size: int = 2000
) -> DecodeFn:
    """Wrap a trained GNNDecoder as a `sweep.DecodeFn`.

    Rebuilds the decoding graph from whichever circuit it's called with,
    rather than reusing the training-time graph: edge weights depend on
    the circuit's physical error rate, so evaluating at a different rate
    than training needs fresh weights even though the graph topology (and
    the model's parameters, which are graph-size-independent) stay the same.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    def decode_fn(circuit: stim.Circuit, detector_syndromes: np.ndarray) -> np.ndarray:
        graph = build_decoding_graph(circuit)
        num_shots = len(detector_syndromes)
        predictions = np.zeros((num_shots, 1), dtype=bool)
        if graph.edge_index.shape[1] == 0:
            # No error mechanisms at all (e.g. physical_error_rate == 0):
            # there is no decoding problem to solve and no graph structure
            # for the model to reason over -- always "no flip" is exactly
            # correct here, and is also the only well-defined answer, since
            # this input is entirely out-of-distribution for a model trained
            # on graphs that do have edges.
            return predictions
        with torch.no_grad():
            for start in range(0, num_shots, batch_size):
                chunk = slice(start, start + batch_size)
                x, edge_index, edge_attr, batch_vec = syndromes_to_model_inputs(
                    graph, detector_syndromes[chunk]
                )
                x, edge_index, edge_attr, batch_vec = (
                    t.to(device) for t in (x, edge_index, edge_attr, batch_vec)
                )
                logits = model(x, edge_index, edge_attr, batch_vec)
                predictions[chunk, 0] = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        return predictions

    return decode_fn
