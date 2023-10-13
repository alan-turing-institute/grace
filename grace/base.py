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


@enum.unique
class EdgeProps(str, enum.Enum):
    """Ordered list of edge properties to be used for classifier training."""

    EDGE_LENGTH = "edge_length_nrm"
    EDGE_ORIENT = "edge_orient_rad"
    EAST_NEIGHBOUR_LENGTH = "east_to_mid_length_nrm"
    WEST_NEIGHBOUR_LENGTH = "west_to_mid_length_nrm"
    EAST_NEIGHBOUR_ORIENT = "east_to_mid_orient_rad"
    WEST_NEIGHBOUR_ORIENT = "west_to_mid_orient_rad"
    EAST_TRIANGLE_AREA = "east_triangle_area_nrm"
    WEST_TRIANGLE_AREA = "west_triangle_area_nrm"
    EAST_DISTANCE = "east_to_mid_length_rel"
    WEST_DISTANCE = "west_to_mid_length_rel"


@enum.unique
class PointCoords(str, enum.Enum):
    """Ordered list of relative point positions (x, y coords) in the graph."""

    SOUTH_POS_X_REL = "south_pos_x_rel"
    SOUTH_POS_Y_REL = "south_pos_y_rel"
    NORTH_POS_X_REL = "north_pos_x_rel"
    NORTH_POS_Y_REL = "north_pos_y_rel"
    MID_POS_X_REL = "mid_pos_x_rel"
    MID_POS_Y_REL = "mid_pos_y_rel"
    EAST_POS_X_REL = "east_pos_x_rel"
    EAST_POS_Y_REL = "east_pos_y_rel"
    WEST_POS_X_REL = "west_pos_x_rel"
    WEST_POS_Y_REL = "west_pos_y_rel"


@dataclass
class Properties:
    """Structure to organise the key: value pairs of EdgeProps properties."""

    properties_dict: dict[str, float] = None

    @property
    def property_keys(self) -> list[str]:
        return list(self.properties_dict.keys())

    @property
    def property_vals(self) -> list[float]:
        return list(self.properties_dict.values())

    @property
    def property_training_data(
        self, include_relative_coords: bool = False
    ) -> npt.NDArray:
        if include_relative_coords is False:
            return np.stack(
                [self.properties_dict[prop] for prop in EdgeProps], axis=0
            )
        else:
            keys = list(EdgeProps) + list(PointCoords)
            return np.stack(
                [self.properties_dict[prop] for prop in keys], axis=0
            )

    def from_keys_and_values(
        self, keys: list[str], values: list[float]
    ) -> None:
        assert len(keys) == len(values)
        assert all(isinstance(k, str) for k in keys)
        assert all(isinstance(v, (float, np.floating)) for v in values)
        self.properties_dict = {k: v for k, v in zip(keys, values)}


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
