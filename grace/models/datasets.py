from typing import List

import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Data
from grace.base import GraphAttrs, Annotation


def dataset_from_graph(
    graph: nx.Graph,
    *,
    mode: str = "full",
    n_hop: int = 1,
    in_train_mode: bool = True,
) -> List[Data]:
    """Create a pytorch geometric dataset from a given networkx graph.

    Parameters
    ----------
    graph : graph
        A networkx graph.
    mode : str
        "sub" or "full".
    n_hop : int
        The number of hops from the central node when creating the subgraphs.
    in_train_mode : bool
        Traverses & checks sub-graphs to generate training dataset. Default = True

    Returns
    -------
    dataset : List[Data] or Data
        A (list of) pytorch geometric data object(s) representing the extracted
        subgraphs or full graph.

    TODO:
        - currently doesn't work on 'corner' nodes i.e. nodes which have
        patches cropped at the boundary of the image - need to pad the image beforehand
    """

    assert mode in ["sub", "full"]

    if mode == "sub":
        dataset = []

        for node, values in graph.nodes(data=True):
            if (
                in_train_mode
                and values[GraphAttrs.NODE_GROUND_TRUTH] is Annotation.UNKNOWN
            ):
                continue

            sub_graph = nx.ego_graph(graph, node, radius=n_hop)
            edge_label = [
                edge[GraphAttrs.EDGE_GROUND_TRUTH]
                for _, _, edge in sub_graph.edges(data=True)
            ]

            if in_train_mode and all(
                [e == Annotation.UNKNOWN for e in edge_label]
            ):
                continue

            pos = np.stack(
                [
                    (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
                    for _, node in graph.nodes(data=True)
                ],
                axis=0,
            )
            central_node = np.array(
                [values[GraphAttrs.NODE_X], values[GraphAttrs.NODE_Y]]
            )
            edge_attr = pos - central_node

            data = _info_from_graph(
                sub_graph,
                pos=torch.Tensor(pos),
                edge_attr=torch.Tensor(edge_attr),
                y=torch.as_tensor([values[GraphAttrs.NODE_GROUND_TRUTH]]),
            )

            dataset.append(data)

    elif mode == "full":
        edge_label = [
            edge[GraphAttrs.EDGE_GROUND_TRUTH]
            for _, _, edge in graph.edges(data=True)
        ]

        pos = np.stack(
            [
                (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
                for _, node in graph.nodes(data=True)
            ],
            axis=0,
        )
        data = _info_from_graph(
            sub_graph,
            pos=torch.Tensor(pos),
            edge_attr=torch.Tensor(edge_attr),
            y=torch.as_tensor([values[GraphAttrs.NODE_GROUND_TRUTH]]),
            edge_label=torch.Tensor(edge_label).long(),
        )

        return data


def _info_from_graph(
    graph: nx.Graph,
    **kwargs,
) -> Data:
    x = np.stack(
        [node[GraphAttrs.NODE_FEATURES] for _, node in graph.nodes(data=True)],
        axis=0,
    )

    item = nx.convert_node_labels_to_integers(graph)
    edges = list(item.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(
        x=torch.Tensor(x),
        edge_index=edge_index,
        **kwargs,
    )

    return data
