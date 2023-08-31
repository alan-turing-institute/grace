import dataclasses
import networkx as nx
import numpy as np
import numpy.typing as npt

from scipy.interpolate import interp1d
from scipy.spatial import Delaunay

from grace.base import edges_from_delaunay


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

    # def asdict(self) -> DetectionNode:
    def asdict(self) -> dict:
        return dataclasses.asdict(self)


def _line_motif(
    src: DetectionNode, dst: DetectionNode, *, density: float
) -> list[DetectionNode]:
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
) -> list[DetectionNode]:
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
) -> list[DetectionNode]:
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
) -> list[DetectionNode]:
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
    motifs: str | list[str] = "lines",
) -> nx.Graph:
    """Create a random graph with motif objects.

    Parameters
    ----------
    n_motifs : int
        The number of each motif to add from 'motifs'.
    n_chaff : int
        The number of false positive detections.
    n_features : int
        The number of feature for each detection.
    scale : float
        A scaling factor. The default detections are generated in a 2D box in
        the range of (0.0, 1.0), i.e. a scaling factor of 1.0
    density : float
        The density of detections when forming a motif.
    motifs: str | list[str]
        The type of object(s) to draw. Defaults to "lines"

    Returns
    -------
    graph : graph
        A networkx graph.
    """
    # Check which motifs to draw:
    assert isinstance(motifs, str) or isinstance(motifs, list)

    if isinstance(motifs, str):
        motifs = [
            motifs,
        ]

    for m in range(len(motifs)):
        if not motifs[m].endswith("s"):
            motifs[m] += "s"

    # Sample some noisy nodes across the graph (GT label = 0):
    chaff_nodes = [
        _random_node(
            RNG.uniform(0.0, 1.0, size=(n_features,)), label=0, scale=scale
        )
        for _ in range(n_chaff)
    ]

    # Iterate through motifs to draw (GT label = 0):
    motif_nodes = []
    for motif in motifs:
        for n in range(n_motifs):
            src = _random_node(np.ones((n_features,)), label=1, scale=scale)
            dst = _random_node(np.ones((n_features,)), label=1, scale=scale)

            if motif == "lines":  # straight line:
                motif_obj = _line_motif(src, dst, density=density * scale)

            elif motif == "curves":  # curved line:
                motif_obj = _curve_motif(src, dst, density=density * scale)

            elif motif == "spirals":  # spiral curve:
                motif_obj = _spiral_motif(src=src, density=scale * density)

            elif motif == "circles":  # 2D circle:
                motif_obj = _circle_motif(src=src, density=scale * density)

            else:
                raise ValueError(f"Invalid 'motif' string: '{motif}'")

            # Assign object identity index to the nodes of the new motif:
            for node in motif_obj:
                node.object_idx = n + 1
            motif_nodes += motif_obj

    # Combine noisy (TN) and real (TP) nodes & triangulate:
    all_nodes = chaff_nodes + motif_nodes
    points = np.asarray([[node.x, node.y] for node in all_nodes])
    tri = Delaunay(points)

    # Store the node & edge information in graph structure:
    graph = nx.Graph()

    for idx, node in enumerate(all_nodes):
        graph.add_node(idx, **node.asdict())

    edges = edges_from_delaunay(tri)
    for edge in edges:
        graph.add_edge(*edge)

    return graph
