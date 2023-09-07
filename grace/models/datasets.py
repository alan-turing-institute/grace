from typing import List, Union

import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Data
from grace.base import GraphAttrs, Annotation


def dataset_from_graph(
    graph: nx.Graph,
    *,
    mode: str = "whole",
    n_hop: int = 1,
    in_train_mode: bool = True,
) -> Union[Data, List[Data]]:
    """Create a pytorch geometric dataset from a given networkx graph.

    Parameters
    ----------
    graph : graph
        A networkx graph.
    mode : str
        "sub" or "whole".
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

    assert mode in ["sub", "whole"]

    if mode == "sub":
        dataset = []

        for node, values in graph.nodes(data=True):
            if (
                in_train_mode
                and values[GraphAttrs.NODE_GROUND_TRUTH] is Annotation.UNKNOWN
            ):
                continue

            sub_graph = nx.ego_graph(graph, node, radius=n_hop)
            edge_label = _edge_label(sub_graph)

            if in_train_mode and all(
                [e == Annotation.UNKNOWN for e in edge_label]
            ):
                continue

            pos = _pos(sub_graph)
            central_node = np.array(
                [values[GraphAttrs.NODE_X], values[GraphAttrs.NODE_Y]]
            )
            edge_attr = pos - central_node

            data = Data(
                x=_x(sub_graph),
                y=_y(sub_graph),
                pos=pos,
                edge_attr=edge_attr,
                edge_index=_edge_index(sub_graph),
                edge_label=edge_label,
            )

            dataset.append(data)

        return dataset

    elif mode == "whole":
        data = Data(
            x=_x(graph),
            y=_y(graph),
            pos=_pos(graph),
            edge_index=_edge_index(graph),
            edge_label=_edge_label(graph),
        )

        return data


def _edge_label(graph: nx.Graph):
    edge_label = [
        edge[GraphAttrs.EDGE_GROUND_TRUTH]
        for _, _, edge in graph.edges(data=True)
    ]
    return torch.Tensor(edge_label).long()


def _pos(graph: nx.Graph):
    pos = np.stack(
        [
            (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
            for _, node in graph.nodes(data=True)
        ],
        axis=0,
    )
    return torch.Tensor(pos)


def _x(graph: nx.Graph):
    x = np.stack(
        [node[GraphAttrs.NODE_FEATURES] for _, node in graph.nodes(data=True)],
        axis=0,
    )
    return torch.Tensor(x)


def _y(graph: nx.Graph):
    y = np.stack(
        [
            node[GraphAttrs.NODE_GROUND_TRUTH]
            for _, node in graph.nodes(data=True)
        ],
        axis=0,
    )
    return torch.Tensor(y).long()


def _edge_index(graph: nx.Graph):
    item = nx.convert_node_labels_to_integers(graph)
    edges = list(item.edges)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
