from typing import List

import networkx as nx
import numpy as np
import torch
import torch_geometric

from grace.base import GraphAttrs, Annotation


def dataset_from_graph(
    graph: nx.Graph, *, n_hop: int = 1
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
        if values[GraphAttrs.NODE_GROUND_TRUTH] is Annotation.UNKNOWN:
            continue

        sub_graph = nx.ego_graph(graph, node, radius=n_hop)

        edge_label = [
            edge[GraphAttrs.EDGE_GROUND_TRUTH]
            for _, _, edge in sub_graph.edges(data=True)
        ]

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
            edge_label=torch.Tensor(edge_label),
            pos=torch.Tensor(pos),
            y=torch.as_tensor([values[GraphAttrs.NODE_GROUND_TRUTH]]),
        )

        dataset.append(data)

    return dataset
