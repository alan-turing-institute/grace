from __future__ import annotations

import enum
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Any
from dataclasses import dataclass
from scipy.spatial import Delaunay


@enum.unique
class GraphAttrs(str, enum.Enum):
    """These are key names for graph attributes used by grace."""

    NODE_X = "x"
    NODE_Y = "y"
    NODE_GROUND_TRUTH = "node_ground_truth"
    NODE_PREDICTION = "node_prediction"
    NODE_FEATURES = "features"
    NODE_CONFIDENCE = "confidence"
    EDGE_SOURCE = "source"
    EDGE_TARGET = "target"
    EDGE_GROUND_TRUTH = "edge_ground_truth"
    EDGE_PREDICTION = "edge_prediction"
    EDGE_PROPERTIES = "edge_properties"


class Annotation(enum.IntEnum):
    """Annotations for edges and nodes."""

    TRUE_NEGATIVE = 0
    TRUE_POSITIVE = 1
    UNKNOWN = 2


@dataclass
class Properties:
    """TODO: Fill in stuff."""

    properties_dict: dict[str, float] = None
    properties_keys: list[str] = None
    properties_vals: list[str] = None

    def __post_init__(self) -> None:
        if self.properties_dict is not None:
            self.properties_keys = self.property_keys
            self.properties_vals = self.property_vals
            assert len(self.properties_keys) == len(self.properties_vals)
        else:
            assert self.properties_keys is not None
            assert self.properties_vals is not None
            assert len(self.properties_keys) == len(self.properties_vals)

            # Create the dict & clean the keys & vals:
            self.properties_dict = self.create_dict()
            self.properties_keys = None
            self.properties_vals = None

    @property
    def property_keys(self) -> list[str]:
        return self.split_dict()["keys"]

    @property
    def property_vals(self) -> list[float]:
        return self.split_dict()["values"]

    def split_dict(self) -> dict[str, list]:
        keys_and_values = {
            "keys": list(self.properties_dict.keys()),
            "values": list(self.properties_dict.values()),
        }
        return keys_and_values

    def create_dict(self) -> dict[str, float]:
        return {
            k: v for k, v in zip(self.properties_keys, self.properties_vals)
        }


@dataclass
class Prediction:
    """Prediction dataclass all normalised softmax class probabilities.

    Parameters
    ----------
    softmax_probabs : npt.NDArray
        Array or normalised softmax probs as predicted by classifier.

    Methods
    -------
    label : Annotation
        Annotation class label with the highest probability.
    prob_TN : float
        Probability of true negative detection (normalised softmax).
    prob_TP : float
        Probability of true positive detection (normalised softmax).
    prob_UNKNOWN : float
        Probability of UNKNOWN label; should be 0 if excluded from training.

    Notes
    -----
    - Normalised probabilities of all classes must sum up to 1.
    - label return the Annotation (index) of the highest label prob.
    """

    softmax_probs: npt.NDArray

    def __post_init__(self):
        assert self.softmax_probs.ndim == 1
        assert len(self.softmax_probs) == len(Annotation)

        self.softmax_probs.shape[0] >= 2
        assert np.all(self.softmax_probs >= 0)
        assert np.all(self.softmax_probs <= 1)
        assert np.isclose(np.sum(self.softmax_probs), 1.0)

    @property
    def label(self) -> Annotation:
        return Annotation(np.argmax(self.softmax_probs))

    @property
    def prob_TN(self) -> float:
        return self.softmax_probs[Annotation.TRUE_NEGATIVE]

    @property
    def prob_TP(self) -> float:
        return self.softmax_probs[Annotation.TRUE_POSITIVE]

    @property
    def prob_UNKNOWN(self) -> float:
        return self.softmax_probs[Annotation.UNKNOWN]


def _map_annotation(annotation: int | Annotation) -> Annotation:
    if isinstance(annotation, Annotation):
        return Annotation
    if Annotation(annotation):
        return Annotation(annotation)
    return Annotation.UNKNOWN


def _sorted_vertices(vertices: npt.NDArray) -> set[tuple[int, int]]:
    ndim = len(vertices)
    edges = []
    for idx in range(ndim):
        edge = tuple(sorted([vertices[idx], vertices[(idx + 1) % ndim]]))
        edges.append(edge)
    return set(edges)


def edges_from_delaunay(tri: Delaunay) -> set[tuple[int, int]]:
    """Return the set of unique edges from a Delaunay graph.

    Parameters
    ----------
    tri : Delaunay
        An instance of a scipy Delaunay triangulation.

    Returns
    -------
    edges : set
        A set of unique edges {(source_id, target_id), ... }
    """
    edges = set()
    for idx in range(tri.nsimplex):
        edges.update(_sorted_vertices(tri.simplices[idx, ...]))
    return edges


def delaunay_edges_from_nodes(
    graph: nx.Graph, *, update_graph: bool = True
) -> set[tuple[int, int]]:
    """Create a Delaunay triangulation from a graph containing only nodes.

    Parameters
    ----------
    graph : nx.Graph
        The graph containing only nodes.
    update_graph : bool (default: True)
        An option to update the graph in-place with the new edges.

    Returns
    -------
    edges : set
        A set of unique edges {(source_id, target_id), ... }
    """

    if graph.number_of_edges() > 0:
        raise ValueError(f"Graph already contains edges ({graph})")

    points = [
        (node_attr[GraphAttrs.NODE_X], node_attr[GraphAttrs.NODE_Y])
        for _, node_attr in graph.nodes(data=True)
    ]
    points_arr = np.asarray(points)
    tri = Delaunay(points_arr)
    edges = edges_from_delaunay(tri)

    # add edge nodes
    if update_graph:
        edge_attrs = {
            GraphAttrs.EDGE_GROUND_TRUTH: Annotation.UNKNOWN,
        }
        graph.add_edges_from(edges, **edge_attrs)
    return edges


def remap_graph_dict(
    graph_dict: dict[str | GraphAttrs, Any]
) -> dict[GraphAttrs, Any]:
    """Remap the keys of a dictionary to the appropriate `GraphAttrs`."""
    graph_attrs_str_set = {attr.value for attr in GraphAttrs}
    keys = list(graph_dict.keys())
    for key in keys:
        if not isinstance(key, GraphAttrs):
            key = str(key).lower()
            if key in graph_attrs_str_set:
                new_key = GraphAttrs(key)
                graph_dict[new_key] = graph_dict.pop(key)

    # set the annotation type
    if GraphAttrs.NODE_GROUND_TRUTH in graph_dict:
        graph_dict[GraphAttrs.NODE_GROUND_TRUTH] = Annotation(
            graph_dict[GraphAttrs.NODE_GROUND_TRUTH]
        )
    return graph_dict


def graph_from_dataframe(
    df: pd.DataFrame,
    *,
    triangulate: bool = True,
) -> nx.Graph:
    """Return a NetworkX graph from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        A pandas dataframe containing at least (x, y) centroids and features
    triangulate : bool
        Compute the edges based on a delaunay triangulation.

    Returns
    -------
    G : nx.Graph
        A graph of the nodes connected by edges determined using Delaunay
        triangulation.
    """

    ground_truth_provided = GraphAttrs.NODE_GROUND_TRUTH in df.keys()

    # set graph nodes to unknown if not recognized
    if not ground_truth_provided:
        df[GraphAttrs.NODE_GROUND_TRUTH] = Annotation.UNKNOWN

    df[GraphAttrs.NODE_GROUND_TRUTH].apply(lambda x: _map_annotation(x))

    graph = nx.Graph()
    nodes = [
        (idx, remap_graph_dict(row.to_dict())) for idx, row in df.iterrows()
    ]

    # add graph nodes
    graph.add_nodes_from(nodes)

    # create edges
    if triangulate:
        delaunay_edges_from_nodes(graph, update_graph=True)
    return graph
