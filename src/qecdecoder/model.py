"""GNN decoder: message passing over the fixed decoding graph, pooled to a
graph-level prediction of whether the logical observable flipped.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import TransformerConv, global_mean_pool

from qecdecoder.graph import DecodingGraph

_NUM_INPUT_CHANNELS = 2  # [detector fired, is boundary node]


class GNNDecoder(nn.Module):
    """Graph-level binary classifier: syndrome -> P(logical observable flipped)."""

    def __init__(self, hidden_channels: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(_NUM_INPUT_CHANNELS, hidden_channels)
        self.convs = nn.ModuleList(
            [
                TransformerConv(hidden_channels, hidden_channels, edge_dim=1)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Returns per-graph logits, shape (num_graphs,)."""
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(h + conv(h, edge_index, edge_attr)))
        pooled = global_mean_pool(h, batch)
        return self.head(pooled).squeeze(-1)


def syndromes_to_model_inputs(
    graph: DecodingGraph, detector_syndromes: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized conversion of a batch of syndromes into GNNDecoder inputs.

    Builds one graph per shot by replicating `graph`'s fixed topology
    `batch_size` times with node-id offsets, entirely with numpy -- no
    per-shot Python loop or `torch_geometric.data.Batch` machinery needed
    since every shot shares the exact same edge structure.

    Returns (x, edge_index, edge_attr, batch) ready to pass to
    `GNNDecoder.forward`.
    """
    batch_size, num_detectors = detector_syndromes.shape
    if num_detectors != graph.num_detectors:
        raise ValueError(
            f"detector_syndromes has {num_detectors} detectors, "
            f"graph has {graph.num_detectors}"
        )

    fired = np.zeros((batch_size, graph.num_nodes), dtype=np.float32)
    fired[:, : graph.num_detectors] = detector_syndromes.astype(np.float32)
    is_boundary = np.zeros((batch_size, graph.num_nodes), dtype=np.float32)
    is_boundary[:, graph.boundary_node] = 1.0
    x = np.stack([fired, is_boundary], axis=-1).reshape(batch_size * graph.num_nodes, 2)

    num_edges = graph.edge_index.shape[1]
    offsets = (np.arange(batch_size) * graph.num_nodes).reshape(batch_size, 1, 1)
    edge_index = graph.edge_index[np.newaxis, :, :] + offsets
    edge_index = edge_index.transpose(1, 0, 2).reshape(2, batch_size * num_edges)

    edge_attr = np.tile(graph.edge_weight, batch_size).reshape(-1, 1)
    batch_vec = np.repeat(np.arange(batch_size), graph.num_nodes)

    return (
        torch.from_numpy(x),
        torch.from_numpy(edge_index).long(),
        torch.from_numpy(edge_attr),
        torch.from_numpy(batch_vec).long(),
    )
