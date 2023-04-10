from typing import List, Tuple

import enum
import networkx as nx
import numpy as np
import numpy.typing as npt

from grace.base import Annotation, GraphAttrs


class EdgeColor(str, enum.Enum):
    """Colour mapping for `Annotation`."""

    TRUE_POSITIVE = "green"
    TRUE_NEGATIVE = "magenta"
    UNKNOWN = "blue"


def color_edges(graph: nx.Graph) -> str:
    """Color an edge based on the set it belongs to."""
    edge_colors = []
    for source, target, edge_attr in graph.edges(data=True):
        edge_annotation = edge_attr[GraphAttrs.EDGE_GROUND_TRUTH].name
        color = EdgeColor[edge_annotation]
        edge_colors.append(color.value)
    return edge_colors


def graph_to_napari_layers(graph: nx.Graph) -> Tuple[npt.NDArray, npt.NDArray]:
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
            (node_attrs[GraphAttrs.NODE_Y], node_attrs[GraphAttrs.NODE_X])
            for _, node_attrs in graph.nodes(data=True)
        ]
    )
    edges = np.array([(points[i, :], points[j, :]) for i, j in graph.edges])
    return points, edges


def cut_graph_using_mask(
    graph: nx.Graph,
    mask: np.ndarray,
    *,
    update_graph: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    """Given a binary mask, cut the graph to contain only edges that are within
    the mask.

    Parameters
    ----------
    graph : nx.graph
        An instance of a networkx graph to be cut.
    mask : array
        A binary mask to filter points and edges in the graph.
    update_graph : bool, (default: True)
        Update edge attributes in place.

    Returns
    -------
    indices : list
        The indices of nodes within the mask.
    enclosed_edges : set
        The edges of the graph within the mask.
    cut_edges : set
        The edges of the graph that cross the boundary of the mask.
    """
    # these are object indices, found within the masks
    points_arr, _ = graph_to_napari_layers(graph)
    points_int = np.round(points_arr).astype(int)
    values_at_points = mask[tuple(points_int.T)]
    indices = np.nonzero(values_at_points)[0]

    # now get the simplices which contain these points
    enclosed_edges = set()
    cut_edges = set()

    # iterate over the indices, and find simplices containing those detections
    for idx in indices:
        adjacent_edges = graph.edges(idx)

        for edge in adjacent_edges:
            source, target = edge
            source_coords = points_int[source, :]
            target_coords = points_int[target, :]

            # check whether the ray exits the mask
            edge_contained = _ray_trace_along_edge(
                source_coords, target_coords, mask
            )

            # update the graph in-place
            if update_graph:
                annotation = (
                    Annotation.TRUE_POSITIVE
                    if edge_contained
                    else Annotation.TRUE_NEGATIVE
                )
                graph[source][target][
                    GraphAttrs.EDGE_GROUND_TRUTH
                ] = annotation

            # get the index of the edge, and store that
            edge_idx = list(graph.edges()).index(tuple(sorted(edge)))
            if edge_contained:
                enclosed_edges.add(edge_idx)
            else:
                cut_edges.add(edge_idx)

    return indices, enclosed_edges, cut_edges


def _ray_trace_along_edge(
    source_coords: npt.NDArray, target_coords: npt.NDArray, mask: npt.NDArray
) -> bool:
    """Ray trace along an edge in the corresponding pixel space and determine
    whether the ray crosses background pixels."""

    x0, y0 = source_coords
    x1, y1 = target_coords
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
