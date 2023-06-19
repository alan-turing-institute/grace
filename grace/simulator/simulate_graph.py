from __future__ import annotations

from typing import List

import dataclasses
import networkx as nx
import numpy as np
import numpy.typing as npt

from scipy.interpolate import interp1d
from scipy.spatial import Delaunay

from ..base import edges_from_delaunay, GraphAttrs


RNG = np.random.default_rng()


@dataclasses.dataclass
class DetectionNode:
    """A detection node object from a detection module.

    Parameters
    ----------
    x : float
    y : float
    features : array
    label : int
    object_idx : int
    """

    x: float
    y: float
    features: npt.NDArray
    label: int
    object_idx: int = 0

    def asdict(self) -> DetectionNode:
        return dataclasses.asdict(self)


def _line_motif(
    src: DetectionNode, dst: DetectionNode, *, density: float
) -> List[DetectionNode]:
    length = np.sqrt((dst.x - src.x) ** 2 + (dst.y - src.y) ** 2)
    n_points = int(length / density)

    dx = np.linspace(src.x, dst.x, n_points)
    dy = np.linspace(src.y, dst.y, n_points)

    label = src.label

    linfit = interp1d([0, 1], np.vstack([src.features, dst.features]), axis=0)
    dfeatures = linfit(np.linspace(0, 1, n_points))

    return [
        DetectionNode(x, y, f, label=label)
        for x, y, f in zip(dx, dy, dfeatures)
    ]


def _curve_motif(
    src: DetectionNode,
    dst: DetectionNode,
    *,
    density: float = 0.02,
) -> List[DetectionNode]:
    # length = np.sqrt((dst.x - src.x) ** 2 + (dst.y - src.y) ** 2)
    curvature = RNG.uniform(1, 10)
    curvature = curvature if RNG.integers(0, 2) < 0.5 else -curvature
    n_points = int((1 / density) / 4)

    # Calculate the midpoint between the start and end nodes
    mid_x = (src.x + dst.x) / 2
    mid_y = (src.y + dst.y) / 2

    # Calculate the perpendicular vector to the line between the start and end nodes
    dx = dst.x - src.x
    dy = dst.y - src.y
    perp_x = dy
    perp_y = -dx
    perp_norm = np.sqrt(perp_x**2 + perp_y**2)
    perp_x /= perp_norm
    perp_y /= perp_norm

    # Calculate the radius of curvature
    radius = 1 / curvature

    # Calculate the angle to the center of curvature
    cx = mid_x + radius * perp_x
    cy = mid_y + radius * perp_y
    angle = np.arctan2(mid_y - cy, mid_x - cx)

    # Generate the points along the curve
    angles = np.linspace(0, angle, n_points, endpoint=True)
    dx = cx + radius * np.cos(angles + np.pi / 2 * np.sign(curvature))
    dy = cy + radius * np.sin(angles + np.pi / 2 * np.sign(curvature))
    label = src.label
    dfeatures = np.repeat(src.features.reshape((1, -1)), n_points, axis=0)

    return [
        DetectionNode(x, y, f, label=label)
        for x, y, f in zip(dx, dy, dfeatures)
    ]


def _spiral_motif(
    src: DetectionNode,
    dst: DetectionNode = None,
    *,
    density: float = 0.02,
) -> List[DetectionNode]:
    curvature = RNG.uniform(5, 10)
    radius = 1 / curvature
    num_turns = np.abs(curvature / 3)
    n_points = int(((1 / density) / 4) * num_turns)

    angle_st = RNG.uniform(0, 2 * np.pi)
    angle_en = angle_st + num_turns * 2 * np.pi
    angles = np.linspace(angle_st, angle_en, n_points, endpoint=True)
    dx = src.x + radius * np.cos(angles) * (angles - angle_st) / (
        2 * np.pi * num_turns
    )
    dy = src.y + radius * np.sin(angles) * (angles - angle_st) / (
        2 * np.pi * num_turns
    )
    label = src.label
    dfeatures = np.repeat(src.features.reshape((1, -1)), n_points, axis=0)

    return [
        DetectionNode(x, y, f, label=label)
        for x, y, f in zip(dx, dy, dfeatures)
    ]


def _circle_motif(
    src: DetectionNode,
    dst: DetectionNode = None,
    *,
    density: float,
) -> List[DetectionNode]:
    radius = RNG.uniform(0.05, 0.5)
    n_points = int((1 / density) / 4)

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    dx = src.x + radius * np.cos(angles)
    dy = src.y + radius * np.sin(angles)

    label = src.label
    dfeatures = np.repeat(src.features.reshape((1, -1)), n_points, axis=0)

    return [
        DetectionNode(x, y, f, label=label)
        for x, y, f in zip(dx, dy, dfeatures)
    ]


def _random_node(
    features: npt.NDArray, label: int, *, scale: float = 1.0
) -> DetectionNode:
    node = DetectionNode(
        x=RNG.uniform(0, 1 * scale),
        y=RNG.uniform(0, 1 * scale),
        features=features,
        label=label,
    )
    return node


def random_graph(
    *,
    n_motifs: int = 3,
    n_chaff: int = 100,
    n_features: int = 3,
    scale: float = 1.0,
    density: float = 0.02,
    motif: str = "lines",
) -> nx.Graph:
    """Create a random graph with line objects.

    Parameters
    ----------
    n_motifs : int
        The number of random lines to add.
    n_chaff : int
        The number of false positive detections.
    n_features : int
        The number of feature for each detection.
    scale : float
        A scaling factor. The default detections are generated in a 2D box in
        the range of (0.0, 1.0), i.e. a scaling factor of 1.0
    density : float
        The density of detections when forming a motif.
    motif: str
        The type of object to draw. Defaults to "lines"

    Returns
    -------
    graph : graph
        A networkx graph.
    """
    chaff = [
        _random_node(
            RNG.uniform(0.0, 1.0, size=(n_features,)), label=0, scale=scale
        )
        for _ in range(n_chaff)
    ]

    # Identify which motif to draw:
    motif_type = f"{motif}s" if not motif.endswith("s") else motif

    motifs = []
    ground_truth = []

    for n in range(n_motifs):
        src = _random_node(np.ones((n_features,)), label=1, scale=scale)

        if motif_type == "lines":
            # Straight line:
            dst = _random_node(np.ones((n_features,)), label=1, scale=scale)
            motif = _line_motif(src, dst, density=density * scale)

        elif motif_type == "curves":
            # Curved line:
            dst = _random_node(np.ones((n_features,)), label=1, scale=scale)
            motif = _curve_motif(src, dst, density=density * scale)

        elif motif_type == "spirals":
            # Spiral curve:
            # curvature = RNG.uniform(1, 10) if RNG.integers(0, 2) < 0.5 else -RNG.uniform(1, 10)
            motif = _spiral_motif(src=src, density=scale * density)

        elif motif_type == "circles":
            # 2D circle:
            # curvature = RNG.uniform(1, 3)
            motif = _circle_motif(src=src, density=scale * density)

        else:
            raise ValueError(
                f"Invalid 'motif_type' string specified: '{motif}'"
            )

        for node in motif:
            node.object_idx = n + 1

        motifs += motif
        ground_truth.append(motif)

    all_nodes = chaff + motifs

    points = np.asarray([[node.x, node.y] for node in all_nodes])
    tri = Delaunay(points)

    graph = nx.Graph()

    for idx, node in enumerate(all_nodes):
        graph.add_node(idx, **node.asdict())

    edges = edges_from_delaunay(tri)
    for edge in edges:
        graph.add_edge(*edge)

    return graph


def random_graph_mixed_motifs(
    *,
    n_motifs: int = 3,
    n_chaff: int = 100,
    n_features: int = 3,
    scale: float = 1.0,
    density: float = 0.02,
    motifs: list = ["line", "curve", "spiral", "circle"],
) -> nx.Graph:
    """Create a random graph with line objects.

    Parameters
    ----------
    n_motifs : int
        The number of random lines to add.
    n_chaff : int
        The number of false positive detections.
    n_features : int
        The number of feature for each detection.
    scale : float
        A scaling factor. The default detections are generated in a 2D box in
        the range of (0.0, 1.0), i.e. a scaling factor of 1.0
    density : float
        The density of detections when forming a motif.
    motifs: list[str]
        All types of object motifs to draw. Defaults to cover all options.

    Returns
    -------
    graph : graph
        A networkx graph.
    """
    chaff = [
        _random_node(
            RNG.uniform(0.0, 1.0, size=(n_features,)), label=0, scale=scale
        )
        for _ in range(n_chaff)
    ]

    # Identify which motif to draw:
    motifs = []
    ground_truth = []

    for n in range(n_motifs):
        # Straight line:
        src = _random_node(np.ones((n_features,)), label=1, scale=scale)
        dst = _random_node(np.ones((n_features,)), label=1, scale=scale)
        motif = _line_motif(src, dst, density=density * scale)

        for node in motif:
            node.object_idx = n + 1

        motifs += motif
        ground_truth.append(motif)

        # Curved line:
        src = _random_node(np.ones((n_features,)), label=1, scale=scale)
        dst = _random_node(np.ones((n_features,)), label=1, scale=scale)
        motif = _curve_motif(src, dst, density=density * scale)

        for node in motif:
            node.object_idx = n + 1

        motifs += motif
        ground_truth.append(motif)

        # Spiral curve:
        src = _random_node(np.ones((n_features,)), label=1, scale=scale)
        dst = _random_node(np.ones((n_features,)), label=1, scale=scale)
        motif = _spiral_motif(src=src, density=scale * density)

        for node in motif:
            node.object_idx = n + 1

        motifs += motif
        ground_truth.append(motif)

        # 2D circle:
        src = _random_node(np.ones((n_features,)), label=1, scale=scale)
        dst = _random_node(np.ones((n_features,)), label=1, scale=scale)
        motif = _circle_motif(src=src, density=scale * density)

        for node in motif:
            node.object_idx = n + 1

        motifs += motif
        ground_truth.append(motif)

    # Done drawing:
    all_nodes = chaff + motifs

    points = np.asarray([[node.x, node.y] for node in all_nodes])
    tri = Delaunay(points)

    graph = nx.Graph()

    for idx, node in enumerate(all_nodes):
        graph.add_node(idx, **node.asdict())

    edges = edges_from_delaunay(tri)
    for edge in edges:
        graph.add_edge(*edge)

    return graph


def update_graph_with_dummy_predictions(G: nx.Graph) -> None:
    """Create a random graph with line objects.

    Parameters
    ----------
    G : nx.Graph
        The graph which node & edge predictions are to be updated with synthetic values.

    Returns
    -------
    None
    - modifies the graph in place
    """
    nodes = list(G.nodes.data())

    for _, node in nodes:
        pd = np.random.random() * 0.5
        if node["label"] > 0:
            node[GraphAttrs.NODE_PREDICTION] = pd
        else:
            node[GraphAttrs.NODE_PREDICTION] = 1 - pd

    for edge in G.edges.data():
        pd = np.random.random() * 0.1
        _, e_i = nodes[edge[0]]
        _, e_j = nodes[edge[1]]

        if e_i["object_idx"] == e_j["object_idx"] and e_i["label"] > 0:
            edge[2][GraphAttrs.EDGE_PREDICTION] = 1 - pd
        else:
            edge[2][GraphAttrs.EDGE_PREDICTION] = pd
