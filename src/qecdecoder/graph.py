"""Fixed decoding graph derived from a Stim circuit's detector error model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass(frozen=True)
class DecodingGraph:
    """Decoding graph for a single-observable circuit.

    Nodes are the circuit's detectors plus one virtual boundary node (the
    last node, index `num_detectors`). Edges come from the circuit's
    detector error model: each graph-like error mechanism becomes an edge
    between the two detectors it flips, or between one detector and the
    boundary node for a boundary-only error.
    """

    num_detectors: int
    edge_index: np.ndarray
    """Shape (2, num_edges), int64. Node ids in [0, num_nodes)."""

    edge_weight: np.ndarray
    """Shape (num_edges,), float32. Matching-style cost -log(p / (1 - p))."""

    edge_flips_observable: np.ndarray
    """Shape (num_edges,), bool. Whether this error mechanism flips observable 0."""

    @property
    def num_nodes(self) -> int:
        return self.num_detectors + 1

    @property
    def boundary_node(self) -> int:
        return self.num_detectors


def build_decoding_graph(circuit: stim.Circuit) -> DecodingGraph:
    """Build a decoding graph from a circuit's detector error model.

    Requires a graph-like DEM (every error mechanism touches at most two
    detectors, possibly after splitting on separators) -- true for the
    code-capacity and circuit-level noise models used in this project.
    """
    if circuit.num_observables != 1:
        raise ValueError(
            "build_decoding_graph only supports single-observable circuits, "
            f"got {circuit.num_observables}"
        )

    dem = circuit.detector_error_model(decompose_errors=True)
    num_detectors = circuit.num_detectors
    boundary_node = num_detectors

    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    flips_observable: list[bool] = []

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        weight = _edge_weight(probability)
        for component in _split_on_separators(instruction.targets_copy()):
            detector_ids = [t.val for t in component if t.is_relative_detector_id()]
            flips_obs = any(t.is_logical_observable_id() for t in component)
            if len(detector_ids) == 0:
                continue  # undetectable error component; no graph edge can represent it
            if len(detector_ids) == 1:
                i, j = detector_ids[0], boundary_node
            elif len(detector_ids) == 2:
                i, j = detector_ids
            else:
                raise ValueError(
                    f"non-graph-like DEM: error component touches {len(detector_ids)} detectors"
                )
            sources.append(i)
            targets.append(j)
            weights.append(weight)
            flips_observable.append(flips_obs)

    return DecodingGraph(
        num_detectors=num_detectors,
        edge_index=np.array([sources, targets], dtype=np.int64),
        edge_weight=np.array(weights, dtype=np.float32),
        edge_flips_observable=np.array(flips_observable, dtype=bool),
    )


def _split_on_separators(targets: list) -> list[list]:
    """Split a DEM instruction's targets into components at `^` separators.

    Stim represents one correlated physical error as several graph edges
    (sharing the error's probability) by joining their targets with a
    separator, e.g. "D4 D6 ^ D5 L0" is two edges: (D4, D6) and (D5, boundary
    with an observable flip).
    """
    groups: list[list] = [[]]
    for target in targets:
        if target.is_separator():
            groups.append([])
        else:
            groups[-1].append(target)
    return groups


def _edge_weight(probability: float) -> float:
    """Matching-style log-likelihood cost: -log(p / (1 - p))."""
    probability = min(max(probability, 1e-12), 1 - 1e-12)
    return float(-np.log(probability / (1 - probability)))
