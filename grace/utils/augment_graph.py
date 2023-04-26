from typing import Any, Dict, Union, Tuple

import torch

import numpy as np

from grace.base import GraphAttrs, Annotation


class RandomEdgeAdditionAndRemoval:
    """Rotate the image and graph in tandem; i.e., the graph x-y coordinates
    will be transformed to reflect the image rotation.

    Accepts an image stack of size (C,W,H) or (B,C,W,H) and a dictionary
    which includes the graph object.

    Parameters
    ----------
    n_add : int
        Number of new edges to add
    n_remove : Tuple[int]
        Number of existing edges to remove
    rng : numpy.random.Generator
        Random number generator
    """

    def __init__(
        self,
        n_add: int = 10,
        n_remove: int = 10,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.rng = rng
        self.n_add = n_add
        self.n_remove = n_remove

    def __call__(
        self, x: torch.Tensor, graph: Dict[Union[str, GraphAttrs], Any]
    ) -> Tuple[torch.Tensor, dict]:
        edges_list = list(graph["graph"].edges)
        max_node = graph["graph"].number_of_nodes()
        edges_to_add = [
            tuple(self.rng.integers(low=0, high=max_node, size=(2,)))
            + ({GraphAttrs.EDGE_GROUND_TRUTH: Annotation.UNKNOWN},)
            for n in range(self.n_add)
        ]
        edges_to_remove = [
            edges_list[e]
            for e in self.rng.integers(0, len(edges_list), (self.n_remove,))
        ]

        graph["graph"].add_edges_from(edges_to_add)
        graph["graph"].remove_edges_from(edges_to_remove)

        return x, graph
