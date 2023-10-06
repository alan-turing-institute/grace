import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Data
from grace.base import GraphAttrs, EdgeProps

# from dataclasses import dataclass, field

# @dataclass
# class NormalisedProperties:
#     keys: list[str] = field(
#         default_factory=lambda: NormalisedProperties.ordered_keys
#     )

#     # Define the class-level ordered keys
#     ordered_keys = [
#         "edge_length_nrm",
#         "edge_orientation_radians",
#         "east_to_mid_length_nrm",
#         "west_to_mid_length_nrm",
#         "east_to_mid_orient_raw",
#         "west_to_mid_orient_raw",
#         "east_triangle_area_nrm",
#         "west_triangle_area_nrm",
#     ]


# @dataclass
# class NormalisedProperties:
#     """TODO: """
#     # keys: list[str] = [
#     #     NrmProperties.EDGE_LENGTH,
#     #     NrmProperties.EDGE_ORIENT,
#     #     NrmProperties.EAST_NEIGHBOUR_LENGTH,
#     #     NrmProperties.WEST_NEIGHBOUR_LENGTH,
#     #     NrmProperties.EAST_NEIGHBOUR_ORIENT,
#     #     NrmProperties.WEST_NEIGHBOUR_ORIENT,
#     #     NrmProperties.EAST_TRIANGLE_AREA,
#     #     NrmProperties.WEST_TRIANGLE_AREA,
#     # ]
#     keys: list[str] = field(
#         default_factory = [
#             "edge_length_nrm",
#             "edge_orientation_radians",
#             "east_to_mid_length_nrm",
#             "west_to_mid_length_nrm",
#             "east_to_mid_orient_raw",
#             "west_to_mid_orient_raw",
#             "east_triangle_area_nrm",
#             "west_triangle_area_nrm",
#         ]
#     )

#     @property
#     def get_properties(self) -> list[str]:
#         return self.keys


# @dataclass
# class KeyExtractor:
#     keys: list[str] = field(
#         default_factory=lambda: KeyExtractor.ordered_keys
#     )

#     # Define the class-level ordered keys
#     ordered_keys = ["key1", "key2", "key3"]


def dataset_from_graph(
    graph: nx.Graph,
    *,
    num_hops: int = 1,
    mode: str = "whole",
    connection: str = "spiderweb",
    node_order: bool = False,
) -> list[Data]:
    """Create a pytorch geometric dataset from a given networkx graph.

    Parameters
    ----------
    graph : graph
        A networkx graph.
    num_hops : int
        The number of hops from the central node when creating the subgraphs.
    mode : str
        "sub" or "whole".


    Returns
    -------
    dataset : list[Data]
        A (list of) pytorch geometric data object(s) representing the extracted
        subgraphs or full graph.

    TODO:
        - currently doesn't work on 'corner' nodes i.e. nodes which have
        patches cropped at the boundary of the image - need to pad the image beforehand
    """

    assert mode in {"sub", "whole"}

    if mode == "sub":
        dataset = []

        for node_idx, _ in graph.nodes(data=True):
            # Isolate a small subgraph:
            sub_graph = nx.ego_graph(graph, node_idx, radius=num_hops)

            # Release spider web - only edges in contact with central node:
            if connection == "fireworks":
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
                edge_label=_edge_label(sub_graph),
                edge_index=_edge_index(sub_graph),
                edge_properties=_edge_properties(sub_graph),
            )
            dataset.append(data)

        return dataset

    elif mode == "whole":
        data = Data(
            x=_x(graph),
            y=_y(graph),
            edge_label=_edge_label(graph),
            edge_index=_edge_index(graph),
            edge_properties=_edge_properties(graph),
        )

        return [
            data,
        ]


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


def _release_non_central_edges(
    sub_graph: nx.Graph, central_node_idx: int
) -> nx.Graph:
    edges_to_remove = []
    for src, dst, _ in sub_graph.edges(data=True):
        if not (central_node_idx == src or central_node_idx == dst):
            edges_to_remove.append((src, dst))
    sub_graph.remove_edges_from(edges_to_remove)
    return sub_graph


def _x(graph: nx.Graph) -> torch.Tensor:
    x = np.stack(
        [graph.nodes[idx][GraphAttrs.NODE_FEATURES] for idx in graph.nodes()],
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


def _node_degree(graph: nx.Graph, ordered_node_idx_list: list[int]):
    deg = np.stack(
        [graph.degree[idx] for idx in ordered_node_idx_list], axis=0
    )
    return torch.Tensor(deg).long()


def _node_pos_coords(graph: nx.Graph):
    pos = np.stack(
        [
            (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
            for _, node in graph.nodes(data=True)
        ],
        axis=0,
    )
    return torch.Tensor(pos)


def _edge_label(graph: nx.Graph) -> None:
    ground_truth_labels = np.stack(
        [
            edge[GraphAttrs.EDGE_GROUND_TRUTH]
            for _, _, edge in graph.edges(data=True)
        ]
    )
    return torch.Tensor(ground_truth_labels).long()


def _edge_index(graph: nx.Graph):
    item = nx.convert_node_labels_to_integers(graph)
    edges = list(item.edges)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _edge_properties(graph: nx.Graph):
    edge_properties = []
    for _, _, edge in graph.edges(data=True):
        # Check if all EdgeProps exist as an edge attribute:
        edge_props_dict = edge[GraphAttrs.EDGE_PROPERTIES].properties_dict
        assert all(prop in edge_props_dict for prop in EdgeProps)

        # Extract the float values:
        single_edge_props = [edge_props_dict[prop] for prop in EdgeProps]
        edge_properties.append(single_edge_props)

    edge_properties = np.stack(edge_properties, axis=0)
    return torch.Tensor(edge_properties).float()
