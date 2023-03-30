from __future__ import annotations

from typing import List

import dataclasses
import networkx as nx
import numpy as np
import numpy.typing as npt

from scipy.interpolate import interp1d
from scipy.spatial import Delaunay

from .base import edges_from_delaunay

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
    n_lines: int = 3,
    n_chaff: int = 100,
    n_features: int = 3,
    scale: float = 1.0,
    density: float = 0.02,
) -> nx.Graph:
    """Create a random graph with line objects.

    Parameters
    ----------
    n_lines : int
        The number of random lines to add.
    n_chaff : int
        The number of false positive detections.
    n_features : int
        The number of feature for each detection.
    scale : float
        A scaling factor. The default detections are generated in a 2D box in
        the range of (0.0, 1.0), i.e. a scaling factor of 1.0
    density : float
        The density of detections when forming a line.

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
    lines = []
    ground_truth = []

    for n in range(n_lines):
        src = _random_node(np.ones((n_features,)), label=1, scale=scale)
        dst = _random_node(np.ones((n_features,)), label=1, scale=scale)
        line = _line_motif(src, dst, density=density * scale)

        for node in line:
            node.object_idx = n + 1

        lines += line
        ground_truth.append(line)

    all_nodes = chaff + lines

    points = np.asarray([[node.x, node.y] for node in all_nodes])
    tri = Delaunay(points)

    graph = nx.Graph()

    for idx, node in enumerate(all_nodes):
        graph.add_node(idx, **node.asdict())

    edges = edges_from_delaunay(tri)
    for edge in edges:
        graph.add_edge(*edge)

    return graph
