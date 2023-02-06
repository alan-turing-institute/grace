from __future__ import annotations

from typing import List, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch_geometric
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay

from .base import DetectionNode

RNG = np.random.default_rng()


def _line_motif(src: DetectionNode, dst: DetectionNode) -> List[DetectionNode]:
    n_points = int(np.sqrt((dst.x - src.x) ** 2 + (dst.y - src.y) ** 2) / 0.02)

    dx = np.linspace(src.x, dst.x, n_points)
    dy = np.linspace(src.y, dst.y, n_points)

    label = src.label

    linfit = interp1d([0, 1], np.vstack([src.features, dst.features]), axis=0)
    dfeatures = linfit(np.linspace(0, 1, n_points))

    return [
        DetectionNode(x, y, f, label=label)
        for x, y, f in zip(dx, dy, dfeatures)
    ]


def _random_node(features: np.ndarray, label: int) -> DetectionNode:
    node = DetectionNode(
        x=RNG.uniform(0, 1),
        y=RNG.uniform(0, 1),
        features=features,
        label=label,
    )
    return node


def _sorted_vertices(vertices: np.array) -> Set[Tuple[int, int]]:
    ndim = len(vertices)
    edges = []
    for idx in range(ndim):
        edge = tuple(sorted([vertices[idx], vertices[(idx + 1) % ndim]]))
        edges.append(edge)
    return set(edges)


def _edges_from_delaunay(tri: Delaunay) -> Set[Tuple[int, int]]:
    edges = set()
    for idx in range(tri.nsimplex):
        edges.update(_sorted_vertices(tri.simplices[idx, ...]))
    return edges


def random_graph(
    n_lines: int = 3, n_chaff: int = 100, n_features: int = 3
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

    Returns
    -------
    graph : graph
        A networkx graph.

    """
    chaff = [
        _random_node(RNG.uniform(0.0, 1.0, size=(n_features,)), label=0)
        for _ in range(n_chaff)
    ]
    lines = []
    ground_truth = []

    for n in range(n_lines):
        src = _random_node(np.ones((n_features,)), label=1)
        dst = _random_node(np.ones((n_features,)), label=1)
        line = _line_motif(src, dst)
        lines += line
        ground_truth.append(line)

    all_nodes = chaff + lines

    points = np.asarray([[node.x, node.y] for node in all_nodes])
    tri = Delaunay(points)

    graph = nx.Graph()

    for idx, node in enumerate(all_nodes):
        graph.add_node(idx, **node.asdict())

    edges = _edges_from_delaunay(tri)
    for edge in edges:
        graph.add_edge(*edge)

    return graph


def dataset_from_graph(
    graph: nx.Graph, n_hop: int = 1
) -> List[torch_geometric.data.Data]:
    """Create a pytorch geometric dataset from a give networkx graph.

    Parameters
    ----------
    graph : graph
        A networkx graph.
    n_hop : int
        The number of hops from the central node when creating the subgraphs.


    Returns
    -------
    dataset : list
        A list of pytorch geometric data objects representing the extracted
        subgraphs.
    """

    dataset = []

    for node, values in graph.nodes(data=True):

        sub_graph = nx.ego_graph(graph, node, radius=n_hop)

        x = np.stack(
            [node["features"] for _, node in sub_graph.nodes(data=True)],
            axis=0,
        )

        pos = np.stack(
            [(node["x"], node["y"]) for _, node in sub_graph.nodes(data=True)],
            axis=0,
        )

        central_node = np.array([values["x"], values["y"]])
        edge_attr = pos - central_node
        # print(edge_attr)

        item = nx.convert_node_labels_to_integers(sub_graph)
        edges = list(item.edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = torch_geometric.data.Data(
            x=torch.Tensor(x),
            edge_index=edge_index,
            edge_attr=torch.Tensor(edge_attr),
            pos=torch.Tensor(pos),
            # y=F.one_hot(torch.as_tensor(values["label"]), num_classes=2),
            y=torch.as_tensor([values["label"]]),
        )

        dataset.append(data)

    return dataset
