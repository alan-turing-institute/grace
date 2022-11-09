import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from dataclasses import dataclass
import networkx as nx


@dataclass(frozen=True)
class Node:
    x: float
    y: float
    features: np.ndarray

    def as_dict(self):
        return {'x': self.x, 'y': self.y, 'features': self.features}


def line_motif(src, dst):
    n_points = int(np.sqrt((dst.x - src.x) ** 2 + (dst.y - src.y) ** 2) / 0.02)

    dx = np.linspace(src.x, dst.x, n_points)
    dy = np.linspace(src.y, dst.y, n_points)

    linfit = interp1d([0, 1], np.vstack([src.features, dst.features]), axis=0)
    dfeatures = linfit(np.linspace(0, 1, n_points))

    return [Node(x, y, f) for x, y, f in zip(dx, dy, dfeatures)]


def random_node(rng, features):
    node = Node(
        x=rng.uniform(0, 1),
        y=rng.uniform(0, 1),
        features=features,
    )
    return node


def normal_node(point, features):
    node = Node(
        x=point[0],
        y=point[1],
        features=features,
    )
    return node


def make_spatial_nodes(n_lines, n_features):
    rng = np.random.default_rng()
    chaff = [random_node(rng, rng.uniform(0., 1., size=(n_features,))) for _ in range(100)]
    lines = []
    ground_truth = []

    for n in range(n_lines):
        src = random_node(rng, np.ones((n_features,)))
        dst = random_node(rng, np.ones((n_features,)))
        line = line_motif(src, dst)
        lines += line
        ground_truth.append(line)

    all_nodes = chaff + lines

    return all_nodes, ground_truth, line


def make_nodes_from_points(points, features):
    nodes = []
    for i in range(0, len(points)):
        nodes.append(normal_node(points[i], features[i]))

    return nodes


def make_graph(nodes):
    points = np.asarray([[n.x, n.y] for n in nodes])
    tri = Delaunay(points)
    edges = []

    nodes_as_dict = [node.as_dict() for node in nodes]
    m = dict(enumerate(nodes_as_dict))  # mapping from vertices to nodes

    for i in range(tri.nsimplex):
        edges.append([tri.simplices[i, 0], tri.simplices[i, 1]])
        edges.append([tri.simplices[i, 1], tri.simplices[i, 2]])
        edges.append([tri.simplices[i, 2], tri.simplices[i, 0]])

    G = nx.Graph(edges)
    pos = dict(zip(m.keys(), points))

    # maps attributes to the nodes
    nx.set_node_attributes(G, m)

    return G, pos
