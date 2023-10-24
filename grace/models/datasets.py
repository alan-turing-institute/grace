import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Data
from grace.base import GraphAttrs


def dataset_from_graph(
    graph: nx.Graph,
    *,
    num_hops: int | str = 1,
    connection: str = "spiderweb",
    include_coords: bool = False,
    ordered_nodes: bool = False,
) -> list[Data]:
    """Create a pytorch geometric dataset from a given networkx graph.

    Parameters
    ----------
    graph : graph
        A networkx graph.
    num_hops : int | str
        The number of hops from the central node when creating the subgraphs.
        Entire graph is returned when num_hops = "whole" (no subgraphs made).
    connection : str
        Whether central node-formed edges are considered in the subgraph:
        Options: "spiderweb" or "fireworks" (defaults to spiderweb)
                      O              O
                    / | \            |
                  O---O---O  vs. O---O---O
                    \ | /            |
                      O              O
        Ignored if num_hops = "whole" as all edges must be considered here.
        TODO: Fix the edge logic if num_hops > 1!
    include_coords : bool
        Whether to include the node / points coordinates in edge_properties.
        Defaults to False. TODO: Implement properly.
    ordered_nodes : bool
        Whether nodes in the subgraph are returned in particular order.
        If False, nodes are listed from north to south (Cartesian coords).
        If True, central node is first, followed by angle-ordered neighbours.
        Defaults to False. TODO: Implement properly.

    Returns
    -------
    dataset : list[Data]
        A list of pytorch geometric data object(s) representing the extracted
        subgraphs or full graph.
    """
    if isinstance(num_hops, int):
        dataset = []

        for node_idx, _ in graph.nodes(data=True):
            # Isolate a small subgraph:
            sub_graph = nx.ego_graph(graph, node_idx, radius=num_hops)

            # Release spider web - only edges in contact with central node:
            if connection == "fireworks":
                if num_hops == 1:
                    sub_graph = _release_non_central_edges(sub_graph, node_idx)
                    assert len(sub_graph.edges()) == len(sub_graph.nodes()) - 1
            elif connection == "spiderweb":
                pass
            else:
                raise ValueError(
                    f"Graph connectivity type '{connection}' not implemented"
                )

            # Store the data as pytorch.geometric object:
            data = Data(
                x=_x(sub_graph),
                y=_y(sub_graph),
                node_pos=_node_pos_coords(sub_graph),
                edge_label=_edge_label(sub_graph),
                edge_index=_edge_index(sub_graph),
                edge_properties=_edge_properties(sub_graph),
            )
            dataset.append(data)

        return dataset

    elif isinstance(num_hops, str) and num_hops == "whole":
        return [
            Data(
                x=_x(graph),
                y=_y(graph),
                node_pos=_node_pos_coords(graph),
                edge_label=_edge_label(graph),
                edge_index=_edge_index(graph),
                edge_properties=_edge_properties(graph),
            ),
        ]


def _release_non_central_edges(
    sub_graph: nx.Graph, central_node_idx: int
) -> nx.Graph:
    edges_to_remove = []
    for src, dst, _ in sub_graph.edges(data=True):
        if not (central_node_idx == src or central_node_idx == dst):
            edges_to_remove.append((src, dst))
    sub_graph.remove_edges_from(edges_to_remove)
    return sub_graph


def _sort_neighbors_by_angle(neighbors) -> tuple[list[torch.Tensor]]:
    # Calculate the angles of the neighbors relative to the anchor
    angles = torch.atan2(neighbors[:, 1], neighbors[:, 0])

    # Convert angles to degrees
    angles_degrees = angles * (180.0 / torch.pi)

    # Combine angles, neighbors, and indices into a tuple
    neighbor_data = [
        (angle, neighbor, i)
        for i, (angle, neighbor) in enumerate(zip(angles_degrees, neighbors))
    ]

    # Sort the neighbor data based on angles
    sorted_neighbor_data = sorted(neighbor_data, key=lambda x: x[0])

    # Extract the sorted neighbors and indices
    sorted_neighbors = torch.stack([data[1] for data in sorted_neighbor_data])
    sorted_indices = [data[2] for data in sorted_neighbor_data]

    # Find the index of the anchor in the sorted list
    anchor_index = sorted_indices.index(neighbors.shape[0] - 1)

    # Reorder the sorted neighbors to place the anchor at the beginning
    sorted_neighbors = torch.cat(
        [sorted_neighbors[anchor_index:], sorted_neighbors[:anchor_index]]
    )

    # Reorder the sorted indices accordingly
    sorted_indices = (
        sorted_indices[anchor_index:] + sorted_indices[:anchor_index]
    )

    return sorted_neighbors, sorted_indices


def _node_degree(
    graph: nx.Graph, ordered_node_idx_list: list[int]
) -> torch.Tensor:
    deg = np.stack(
        [graph.degree[idx] for idx in ordered_node_idx_list], axis=0
    )
    return torch.Tensor(deg).long()


def _x(graph: nx.Graph) -> torch.Tensor:
    x = np.stack(
        [
            np.concatenate(
                [
                    graph.nodes[idx][GraphAttrs.NODE_FEATURES],
                    graph.nodes[idx][GraphAttrs.NODE_EMBEDDINGS],
                ],
                axis=-1,
            )
            for idx in graph.nodes()
        ],
        axis=0,
    )
    return torch.Tensor(x).float()


def _y(graph: nx.Graph) -> torch.Tensor:
    y = np.stack(
        [
            graph.nodes[idx][GraphAttrs.NODE_GROUND_TRUTH]
            for idx in graph.nodes()
        ],
        axis=0,
    )
    return torch.Tensor(y).long()


def _node_pos_coords(graph: nx.Graph) -> torch.Tensor:
    pos = np.stack(
        [
            (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
            for _, node in graph.nodes(data=True)
        ],
        axis=0,
    )
    return torch.Tensor(pos).float()


def _edge_label(graph: nx.Graph) -> torch.Tensor:
    ground_truth_labels = np.stack(
        [
            edge[GraphAttrs.EDGE_GROUND_TRUTH]
            for _, _, edge in graph.edges(data=True)
        ]
    )
    return torch.Tensor(ground_truth_labels).long()


def _edge_index(graph: nx.Graph) -> torch.Tensor:
    item = nx.convert_node_labels_to_integers(graph)
    edges = list(item.edges)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _edge_properties(graph: nx.Graph) -> torch.Tensor:
    edge_properties = np.stack(
        [
            edge[GraphAttrs.EDGE_PROPERTIES].property_training_data
            for _, _, edge in graph.edges(data=True)
        ],
        axis=0,
    )
    return torch.Tensor(edge_properties).float()
