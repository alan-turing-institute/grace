from __future__ import annotations

import enum
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Set, Tuple
from scipy.spatial import Delaunay


class GraphAttrs(str, enum.Enum):
    """These are key names for graph attributes used by grace."""

    NODE_X = "x"
    NODE_Y = "y"
    NODE_GROUND_TRUTH = "node_ground_truth"
    NODE_PREDICTION = "node_prediction"
    NODE_PROB_DETECTION = "prob_detection"
    NODE_FEATURES = "features"
    NODE_CONFIDENCE = "confidence"
    EDGE_PROB_LINK = "prob_link"
    EDGE_SOURCE = "source"
    EDGE_TARGET = "target"
    EDGE_GROUND_TRUTH = "edge_ground_truth"
    EDGE_CONFIDENCE = "confidence"


class Annotation(enum.IntEnum):
    TRUE_NEGATIVE = 0
    TRUE_POSITIVE = 1
    UNKNOWN = 2


def _sorted_vertices(vertices: npt.NDArray) -> Set[Tuple[int, int]]:
    ndim = len(vertices)
    edges = []
    for idx in range(ndim):
        edge = tuple(sorted([vertices[idx], vertices[(idx + 1) % ndim]]))
        edges.append(edge)
    return set(edges)


def edges_from_delaunay(tri: Delaunay) -> Set[Tuple[int, int]]:
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
) -> Set[Tuple[int, int]]:
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
            GraphAttrs.EDGE_PROB_LINK: 0.0,
            GraphAttrs.EDGE_GROUND_TRUTH: Annotation.UNKNOWN,
        }
        graph.add_edges_from(edges, **edge_attrs)
    return edges


def graph_from_dataframe(
    df: pd.DataFrame,
) -> nx.Graph:
    """Return a NetworkX graph from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        A pandas dataframe containing at least (x, y) centroids and features

    Returns
    -------
    G : nx.Graph
        A graph of the nodes connected by edges determined using Delaunay
        triangulation.
    """

    points = np.asarray(df.loc[:, [GraphAttrs.NODE_Y, GraphAttrs.NODE_X]])
    num_nodes = points.shape[0]

    graph = nx.Graph()

    nodes = [
        (
            idx,
            {
                GraphAttrs.NODE_X: points[idx, 0],
                GraphAttrs.NODE_Y: points[idx, 1],
                GraphAttrs.NODE_PROB_DETECTION: 0.0,
                # GraphAttrs.NODE_FEATURES: features[idx, ...],
            },
        )
        for idx in range(num_nodes)
    ]

    # add graph nodes
    graph.add_nodes_from(nodes)

    # create edges
    delaunay_edges_from_nodes(graph, update_graph=True)
    return graph
