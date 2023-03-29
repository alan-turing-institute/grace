from typing import List

import networkx as nx
import numpy as np
import torch
import torch_geometric


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

        # TODO: edge attributes
        central_node = np.array([values["x"], values["y"]])
        edge_attr = pos - central_node

        item = nx.convert_node_labels_to_integers(sub_graph)
        edges = list(item.edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = torch_geometric.data.Data(
            x=torch.Tensor(x),
            edge_index=edge_index,
            edge_attr=torch.Tensor(edge_attr),
            pos=torch.Tensor(pos),
            y=torch.as_tensor([values["label"]]),
        )

        dataset.append(data)

    return dataset
