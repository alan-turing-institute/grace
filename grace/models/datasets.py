from typing import List

import networkx as nx
import numpy as np
import torch
import torch_geometric

from grace.base import GraphAttrs, Annotation


def dataset_from_subgraphs(
    graph: nx.Graph,
    *,
    n_hop: int = 1,
    in_train_mode: bool = True,
) -> List[torch_geometric.data.Data]:
    """Create a pytorch geometric dataset from a given networkx graph.

    Parameters
    ----------
    graph : graph
        A networkx graph.
    n_hop : int
        The number of hops from the central node when creating the subgraphs.
    in_train_mode : bool
        Traverses & checks sub-graphs to generate training dataset. Default = True

    Returns
    -------
    dataset : list
        A list of pytorch geometric data objects representing the extracted
        subgraphs.

    TODO:
        - currently doesn't work on 'corner' nodes i.e. nodes which have
        patches cropped at the boundary of the image - need to pad the image beforehand
    """

    dataset = []

    for node, values in graph.nodes(data=True):
        # Define a subgraph - n_hop subgraph at train time, whole graph otherwise:
        sub_graph = nx.ego_graph(graph, node, radius=n_hop)

        # Constraint: exclusion of unknown nodes at the centre of subgraph:
        if in_train_mode is True:
            if values[GraphAttrs.NODE_GROUND_TRUTH] is Annotation.UNKNOWN:
                continue

        edge_label = [
            edge[GraphAttrs.EDGE_GROUND_TRUTH]
            for _, _, edge in sub_graph.edges(data=True)
        ]

        # Constraint: exclusion of all unknown edges forming the subgraph:
        if in_train_mode is True:
            if all([e == Annotation.UNKNOWN for e in edge_label]):
                continue

        x = np.stack(
            [
                node[GraphAttrs.NODE_FEATURES]
                for _, node in sub_graph.nodes(data=True)
            ],
            axis=0,
        )

        pos = np.stack(
            [
                (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
                for _, node in sub_graph.nodes(data=True)
            ],
            axis=0,
        )

        # TODO: edge attributes
        central_node = np.array(
            [values[GraphAttrs.NODE_X], values[GraphAttrs.NODE_Y]]
        )
        edge_attr = pos - central_node

        item = nx.convert_node_labels_to_integers(sub_graph)
        edges = list(item.edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = torch_geometric.data.Data(
            x=torch.Tensor(x),
            edge_index=edge_index,
            edge_attr=torch.Tensor(edge_attr),
            edge_label=torch.Tensor(edge_label).long(),
            pos=torch.Tensor(pos),
            y=torch.as_tensor([values[GraphAttrs.NODE_GROUND_TRUTH]]),
        )

        dataset.append(data)

    return dataset


def dataset_from_whole_graph(graph: nx.Graph) -> torch_geometric.data.Data:
    """Create a single pytorch geometric dataset from an entire give networkx graph.

    Parameters
    ----------
    graph : graph
        A networkx graph.

    Returns
    -------
    dataset : list
        A single pytorch geometric data objects representing the extracted graph.
    """

    edge_label = [
        edge[GraphAttrs.EDGE_GROUND_TRUTH]
        for _, _, edge in graph.edges(data=True)
    ]

    x = np.stack(
        [node[GraphAttrs.NODE_FEATURES] for _, node in graph.nodes(data=True)],
        axis=0,
    )

    pos = np.stack(
        [
            (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
            for _, node in graph.nodes(data=True)
        ],
        axis=0,
    )

    item = nx.convert_node_labels_to_integers(graph)
    edges = list(item.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = torch_geometric.data.Data(
        x=torch.Tensor(x),
        edge_index=edge_index,
        edge_label=torch.Tensor(edge_label).long(),
        pos=torch.Tensor(pos),
    )

    return data
