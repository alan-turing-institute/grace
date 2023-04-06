import dataclasses
from typing import List, Tuple, Union

import networkx as nx
import numpy as np

from grace.base import Annotation, GraphAttrs


@dataclasses.dataclass
class SpatialEdge:
    start: Union[tuple, np.ndarray]
    end: Union[tuple, np.ndarray]


def graph_to_napari_layers(graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a networkx graph to a napari compatible layer.

    Parameters
    ----------
    graph : nx.graph
        An instance of a networkx graph to be cut.

    Returns
    -------
    points : array (N, D)
        The spatial coordinates of the vertices or nodes of the graph.
    edges : array (N, D, D)
        The edges of the graph.
    """
    points = np.array(
        [
            (n[GraphAttrs.NODE_X], n[GraphAttrs.NODE_Y])
            for _, n in graph.nodes(data=True)
        ]
    )
    edges = np.array([(points[i, :], points[j, :]) for i, j in graph.edges])
    return points, edges


def cut_graph_using_mask(
    graph: nx.Graph,
    mask: np.ndarray,
    *,
    update_graph: bool = True,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Given a binary mask, cut the graph to contain only edges that are within
    the mask.

    Parameters
    ----------
    graph : nx.graph
        An instance of a networkx graph to be cut.
    mask : array
        A binary mask to filter points and edges in the graph.

    Returns
    -------
    indices : list
        The indices of points within the mask.
    enclosed_edges : set
        The edges of the graph within the mask.
    cut_edges : set
        The edges of the graph that cross the boundary of the mask.
    """

    # edges connected to node
    # G.edges(node)
    points_arr, _ = graph_to_napari_layers(graph)

    points_int = np.round(points_arr).astype(int)
    values_at_points = mask[tuple(points_int.T)]
    indices = np.nonzero(values_at_points)[0]

    # these are object indices, found within the masks
    # now get the simplices which contain these points

    enclosed_edges = set()
    cut_edges = set()

    # iterate over the indices, and find simplices containing those detections
    for idx in indices:
        adjacent_edges = graph.edges(idx)

        for edge in adjacent_edges:
            i, j = edge

            ray = SpatialEdge(
                points_int[i, :],
                points_int[j, :],
            )
            r = _ray_trace_along_edge(ray, mask)

            # get the index of the edge, and store that
            edge_idx = list(graph.edges()).index(tuple(sorted(edge)))

            if update_graph:
                annotation = (
                    Annotation.TRUE_POSITIVE if r else Annotation.TRUE_NEGATIVE
                )
                graph[i][j][GraphAttrs.EDGE_GROUND_TRUTH] = annotation

            if r:
                enclosed_edges.add(edge_idx)
            else:
                cut_edges.add(edge_idx)

    return indices, enclosed_edges, cut_edges


def _ray_trace_along_edge(ray: SpatialEdge, mask: np.ndarray) -> bool:
    """Ray trace along an edge in the corresponding pixel space and determine
    whether the ray crosses background pixels."""

    x0, y0 = ray.start
    x1, y1 = ray.end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1

    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1

    err = dx + dy

    while True:
        if mask[x0, y0] == 0:
            return False

        if x0 == x1 and y0 == y1:
            return True

        e2 = 2 * err

        if e2 >= dy:  # e_xy+e_x > 0
            err += dy
            x0 += sx

        if e2 <= dx:  # e_xy+e_y < 0
            err += dx
            y0 += sy
