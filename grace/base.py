from __future__ import annotations

import enum
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Any, Dict, Set, Tuple
from scipy.spatial import Delaunay


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


class Annotation(enum.IntEnum):
    """Annotations for edges and nodes."""

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
            GraphAttrs.EDGE_PREDICTION: 0.0,
            GraphAttrs.EDGE_GROUND_TRUTH: Annotation.UNKNOWN,
        }
        graph.add_edges_from(edges, **edge_attrs)
    return edges


def remap_graph_dict(
    graph_dict: Dict[str | GraphAttrs, Any]
) -> Dict[GraphAttrs, Any]:
    """Remap the keys of a dictionary to the appropriate `GraphAttrs`."""
    graph_attrs_str_set = {attr.value for attr in GraphAttrs}
    keys = list(graph_dict.keys())
    for key in keys:
        if not isinstance(key, GraphAttrs):
            key = str(key).lower()
            if key in graph_attrs_str_set:
                new_key = GraphAttrs(key)
                graph_dict[new_key] = graph_dict.pop(key)
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
