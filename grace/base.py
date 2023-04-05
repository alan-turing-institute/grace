from __future__ import annotations

import enum
import networkx as nx
import numpy as np
import pandas as pd

from typing import Set, Tuple
from scipy.spatial import Delaunay

import numpy.typing as npt


class GraphAttrs(str, enum.Enum):
    """These are key names for graph attributes used by grace."""

    NODE_X = "x"
    NODE_Y = "y"
    NODE_WIDTH = "width"
    NODE_HEIGHT = "height"
    NODE_GROUND_TRUTH = "ground_truth"
    NODE_PREDICTION = "prediction"
    NODE_PROB_DETECTION = "prob_detection"
    NODE_CONFIDENCE = "confidence"
    NODE_FEATURES = "features"
    EDGE_PROB_LINK = "prob_link"


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

    # TODO(arl): do we want some kind of schema to enforce what's required?
    # TODO(arl): what if we don't have any image features at this point?
    points = np.asarray(df.loc[:, [GraphAttrs.NODE_Y, GraphAttrs.NODE_X]])
    features = np.asarray(
        np.squeeze(np.asarray(df.loc[:, GraphAttrs.NODE_FEATURES]))
    )
    tri = Delaunay(points)
    edges = edges_from_delaunay(tri)
    num_nodes = points.shape[0]

    graph = nx.Graph()

    nodes = [
        (
            idx,
            {
                GraphAttrs.NODE_X: points[idx, 0],
                GraphAttrs.NODE_Y: points[idx, 1],
                GraphAttrs.NODE_PROB_DETECTION: 0.0,
                GraphAttrs.NODE_FEATURES: features[idx, ...],
            },
        )
        for idx in range(num_nodes)
    ]

    # add graph nodes
    graph.add_nodes_from(nodes)

    # add edge nodes
    graph.add_edges_from(edges, **{GraphAttrs.EDGE_PROB_LINK: 0.0})
    return graph
