from typing import Any

import enum
import torch

import numpy as np
import networkx as nx

from grace.base import GraphAttrs, Annotation


def find_average_annotation(
    edge: tuple[int],
    graph: nx.Graph,
) -> Annotation:
    """Finds the average annotation of edges connected to
    the nodes defining an input edge.

    Parameters
    ----------
    edge : tuple[int]
        Tuple of nodes that define the edge

    Returns
    -------
    annotation : GraphAttrs.EDGE_GROUND_TRUTH
        Calculated annotation for the edge
    """

    annotation_list = [
        connection[GraphAttrs.EDGE_GROUND_TRUTH]
        for node in edge
        for connection in graph.adj[node].values()
    ]
    num_positive = annotation_list.count(Annotation.TRUE_POSITIVE)
    num_negative = annotation_list.count(Annotation.TRUE_NEGATIVE)

    if num_positive > num_negative:
        return Annotation.TRUE_POSITIVE
    elif num_negative > num_positive:
        return Annotation.TRUE_NEGATIVE
    else:
        return Annotation.UNKNOWN


@enum.unique
class AnnotationAugmentationModes(str, enum.Enum):
    """Names of annotation modes used for graph augmentation."""

    RANDOM = "random"
    UNKNOWN = "unknown"
    AVERAGE = "average"


class RandomEdgeAdditionAndRemoval:
    """Randomly adds and removes edges to the graph.

    Accepts an image stack of size (C,W,H) or (B,C,W,H) and a dictionary
    which includes the graph object.

    Parameters
    ----------
    p_add : float
        Number of new edges to add, as a fraction of previous edge count
    p_remove : float
        Number of new edges to remove, as a fraction of previous edge count
    rng : numpy.random.Generator
        Random number generator
    annotation_mode : str
        The scheme by which annotations will be assigned to newly addded edges;
        random: chosen by random choice; unknown: will always be GraphAttrs.UNKNOWN,
        average: an average of all edge annotations associated with the nodes of the
        new edge.
    """

    def __init__(
        self,
        p_add: float = 0.1,
        p_remove: float = 0.1,
        rng: np.random.Generator = np.random.default_rng(),
        annotation_mode: str = "random",
    ):
        self.rng = rng
        self.p_add = p_add
        self.p_remove = p_remove

        if annotation_mode not in iter(AnnotationAugmentationModes):
            raise ValueError(
                f"annotation_mode must be one of {list(AnnotationAugmentationModes)}"
            )

        if annotation_mode == "random":
            self.assign_annotation = lambda edge, graph: Annotation(
                rng.choice(Annotation)
            )
        elif annotation_mode == "average":
            self.assign_annotation = (
                lambda edge, graph: find_average_annotation(edge, graph)
            )
        else:
            self.assign_annotation = lambda edge, graph: Annotation.UNKNOWN

    def __call__(
        self, x: torch.Tensor, graph: dict[str, Any]
    ) -> tuple[torch.Tensor, dict]:
        new_graph = graph["graph"].copy()
        graph = graph.copy()

        edges_list = list(graph["graph"].edges)
        max_node = graph["graph"].number_of_nodes()
        n_add = int(self.p_add * len(edges_list))
        n_remove = int(self.p_remove * len(edges_list))

        edges_to_add = [
            tuple(self.rng.integers(low=0, high=max_node, size=(2,)))
            for n in range(n_add)
        ]
        edge_annotations = [
            self.assign_annotation(e, graph["graph"]) for e in edges_to_add
        ]
        edges_to_add = [
            e + ({GraphAttrs.EDGE_GROUND_TRUTH: edge_annotations[n]},)
            for n, e in enumerate(edges_to_add)
        ]

        edges_to_remove = [
            edges_list[e]
            for e in self.rng.integers(0, len(edges_list), (n_remove,))
        ]

        new_graph.add_edges_from(edges_to_add)
        new_graph.remove_edges_from(edges_to_remove)

        graph["graph"] = new_graph

        return x, graph


class RandomXYTranslation:
    """Randomly shifts the X and Y coordinates of each node.

    Parameters
    ----------
    max_shift : float
        Maximum coordinate shift.
    """

    def __init__(self, max_shift: float = 10.0):
        self.max_shift = max_shift

    def __call__(
        self, img: torch.Tensor, graph: dict[str, Any]
    ) -> tuple[torch.Tensor, dict]:
        new_graph = graph["graph"].copy()
        graph = graph.copy()

        new_nodes = []

        for node, values in new_graph.nodes(data=True):
            x, y = values[GraphAttrs.NODE_X], values[GraphAttrs.NODE_Y]
            shift = np.random.uniform(
                low=-self.max_shift, high=self.max_shift, size=(2,)
            )

            new_nodes.append(
                (
                    node,
                    {
                        GraphAttrs.NODE_X: x + shift[0],
                        GraphAttrs.NODE_Y: y + shift[1],
                    },
                )
            )

        new_graph.update(nodes=new_nodes)
        graph["graph"] = new_graph

        return img, graph
