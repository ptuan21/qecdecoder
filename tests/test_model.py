import numpy as np
import torch

from qecdecoder.codes import rotated_surface_code_circuit
from qecdecoder.graph import build_decoding_graph
from qecdecoder.model import GNNDecoder, syndromes_to_model_inputs


def _graph_and_syndromes(distance: int = 3, num_shots: int = 4):
    circuit = rotated_surface_code_circuit(distance=distance, rounds=1, physical_error_rate=0.1)
    graph = build_decoding_graph(circuit)
    rng = np.random.default_rng(0)
    syndromes = rng.integers(0, 2, size=(num_shots, graph.num_detectors)).astype(bool)
    return graph, syndromes


def test_syndromes_to_model_inputs_shapes() -> None:
    graph, syndromes = _graph_and_syndromes(distance=3, num_shots=5)
    x, edge_index, edge_attr, batch = syndromes_to_model_inputs(graph, syndromes)

    num_edges = graph.edge_index.shape[1]
    assert x.shape == (5 * graph.num_nodes, 2)
    assert edge_index.shape == (2, 5 * num_edges)
    assert edge_attr.shape == (5 * num_edges, 1)
    assert batch.shape == (5 * graph.num_nodes,)
    assert batch.max().item() == 4


def test_syndromes_to_model_inputs_rejects_wrong_detector_count() -> None:
    graph, _ = _graph_and_syndromes(distance=3)
    bad_syndromes = np.zeros((3, graph.num_detectors + 1), dtype=bool)
    try:
        syndromes_to_model_inputs(graph, bad_syndromes)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_gnn_decoder_forward_pass_output_shape() -> None:
    graph, syndromes = _graph_and_syndromes(distance=5, num_shots=8)
    x, edge_index, edge_attr, batch = syndromes_to_model_inputs(graph, syndromes)

    model = GNNDecoder(hidden_channels=8, num_layers=2)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr, batch)
    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()


def test_identical_syndromes_give_identical_logits_regardless_of_batch_position() -> None:
    graph, _ = _graph_and_syndromes(distance=3)
    syndromes = np.zeros((2, graph.num_detectors), dtype=bool)
    syndromes[0, 0] = True
    syndromes[1, 0] = True  # same syndrome in both batch slots

    x, edge_index, edge_attr, batch = syndromes_to_model_inputs(graph, syndromes)
    model = GNNDecoder(hidden_channels=8, num_layers=2)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr, batch)
    assert torch.allclose(logits[0], logits[1], atol=1e-6)
