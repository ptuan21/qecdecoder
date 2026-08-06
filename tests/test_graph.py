import pytest

from qecdecoder.codes import repetition_code_circuit, rotated_surface_code_circuit
from qecdecoder.graph import build_decoding_graph


def test_graph_shapes_for_surface_code_d3() -> None:
    circuit = rotated_surface_code_circuit(distance=3, rounds=1, physical_error_rate=0.1)
    graph = build_decoding_graph(circuit)

    assert graph.num_detectors == circuit.num_detectors
    assert graph.num_nodes == circuit.num_detectors + 1
    assert graph.boundary_node == circuit.num_detectors
    assert graph.edge_index.shape[0] == 2
    num_edges = graph.edge_index.shape[1]
    assert graph.edge_weight.shape == (num_edges,)
    assert graph.edge_flips_observable.shape == (num_edges,)
    assert num_edges > 0


def test_graph_edge_indices_within_bounds() -> None:
    circuit = rotated_surface_code_circuit(distance=5, rounds=1, physical_error_rate=0.15)
    graph = build_decoding_graph(circuit)
    assert graph.edge_index.min() >= 0
    assert graph.edge_index.max() < graph.num_nodes


def test_graph_has_at_least_one_observable_flipping_edge() -> None:
    circuit = rotated_surface_code_circuit(distance=3, rounds=1, physical_error_rate=0.1)
    graph = build_decoding_graph(circuit)
    assert graph.edge_flips_observable.any()


def test_graph_edge_weights_are_positive_and_finite() -> None:
    circuit = rotated_surface_code_circuit(distance=5, rounds=1, physical_error_rate=0.1)
    graph = build_decoding_graph(circuit)
    assert (graph.edge_weight > 0).all()
    assert (graph.edge_weight < float("inf")).all()


def test_graph_works_for_repetition_code() -> None:
    circuit = repetition_code_circuit(distance=5, rounds=1, physical_error_rate=0.1)
    graph = build_decoding_graph(circuit)
    assert graph.num_detectors == circuit.num_detectors
    assert graph.edge_index.shape[1] > 0


def test_graph_handles_separator_with_multiple_rounds() -> None:
    """rounds > 1 can produce DEM entries with `^` separators (one physical
    error decomposed into multiple graph edges) -- confirm these parse into
    valid, in-bounds edges instead of raising or silently corrupting data.
    """
    circuit = rotated_surface_code_circuit(distance=3, rounds=3, physical_error_rate=0.05)
    graph = build_decoding_graph(circuit)
    assert graph.edge_index.shape[1] > 0
    assert graph.edge_index.min() >= 0
    assert graph.edge_index.max() < graph.num_nodes


def test_rejects_multi_observable_circuit() -> None:
    import stim

    circuit = stim.Circuit(
        """
        M 0
        M 1
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )
    with pytest.raises(ValueError):
        build_decoding_graph(circuit)
