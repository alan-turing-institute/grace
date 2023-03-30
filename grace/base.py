from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from typing import Set, Tuple
from scipy.spatial import Delaunay

import numpy.typing as npt


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
    points = np.asarray(df.loc[:, ["y", "x"]])
    features = np.asarray(np.squeeze(np.asarray(df.loc[:, "features"])))
    tri = Delaunay(points)
    edges = edges_from_delaunay(tri)

    assert features.shape[0] == points.shape[0]

    graph = nx.Graph()

    for idx, row in enumerate(df.iterrows()):
        graph.add_node(
            idx,
            x=points[idx, 0],
            y=points[idx, 1],
            features=features[idx, ...],
        )

    for edge in edges:
        graph.add_edge(*edge)

    return graph
