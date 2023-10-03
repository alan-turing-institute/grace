import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Data
from grace.base import GraphAttrs

# type properly!


def dataset_from_graph(
    graph: nx.Graph,
    *,
    mode: str = "whole",
    n_hop: int = 1,
) -> list[Data]:
    """Create a pytorch geometric dataset from a given networkx graph.

    Parameters
    ----------
    graph : graph
        A networkx graph.
    mode : str
        "sub" or "whole".
    n_hop : int
        The number of hops from the central node when creating the subgraphs.

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

        for node_idx, node in graph.nodes(data=True):
            # Isolate a small subgraph:
            sub_graph = nx.ego_graph(graph, node_idx, radius=n_hop)
            sub_graph_nodes = list(sub_graph.nodes())
            # print (f"Created a subgraph: {sub_graph} around node index = {node_idx}")

            # NODES:
            # Order nodes by locating neighbours angle:
            absolute_pos = _node_pos_coords(sub_graph)
            central_node = torch.Tensor(
                [node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y]]
            )
            neighbours_relative = absolute_pos - central_node

            # Sort the neighbours:
            relative_pos, neighbour_idx = _sort_neighbors_by_angle(
                neighbors=neighbours_relative
            )
            sorted_neighbour_idx_list = [
                sub_graph_nodes[g] for g in neighbour_idx
            ]

            # Sorted node features:
            x = _x(sub_graph, sorted_neighbour_idx_list)
            y = _y(sub_graph, sorted_neighbour_idx_list)
            degree = _node_degree(sub_graph, sorted_neighbour_idx_list)

            # EDGES:
            # Release spider web - only edges in contact with central node:
            sub_graph = _release_non_central_edges(sub_graph, node_idx)
            sub_graph_edges = list(sub_graph.edges(data=True))

            # Sanity check for a fireworks edge subgraph:
            assert len(sub_graph_edges) == len(sub_graph_nodes) - 1

            # Read the labels of the edges the central node forms:
            # edge_labels = _edge_label(graph, sub_graph_edges)
            edge_labels = _edge_label(sub_graph)

            # Create a list of edge indices, as simple as it gets:
            edge_indices = _edge_index(sub_graph)

            # Calculate edge lenghts:
            edge_length = _edge_length(graph)
            mean_length = torch.mean(edge_length)
            edge_length = _edge_length(sub_graph, mean_length=mean_length)

            # Calculate the edge angle:
            edge_orient = _edge_orientation(sub_graph)

            # Store the data as pytorch.geometric object:
            data = Data(
                x=x,
                y=y,
                degree=degree,
                pos_abs=absolute_pos,
                pos_rel=relative_pos,
                edge_label=edge_labels,
                edge_index=edge_indices,
                edge_length=edge_length,
                edge_orient=edge_orient,
            )
            # print (data)
            dataset.append(data)

        return dataset

    elif mode == "whole":
        pos_abs = _node_pos_coords(graph)
        pos_rel = torch.zeros_like(pos_abs)
        nodes = range(len(graph.nodes()))

        edge_length = _edge_length(graph)
        mean_length = torch.mean(edge_length)
        edge_length = _edge_length(graph, mean_length=mean_length)

        edge_orient = _edge_orientation(graph)

        data = Data(
            x=_x(graph, nodes),
            y=_y(graph, nodes),
            degree=_node_degree(graph, nodes),
            pos_abs=pos_abs,
            pos_rel=pos_rel,
            edge_label=_edge_label(graph),
            edge_index=_edge_index(graph),
            edge_length=edge_length,
            edge_orient=edge_orient,
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


def _x(graph: nx.Graph, ordered_node_idx_list: list[int]) -> torch.Tensor:
    x = np.stack(
        [
            graph.nodes[idx][GraphAttrs.NODE_FEATURES]
            for idx in ordered_node_idx_list
        ],
        axis=0,
    )
    return torch.Tensor(x).float()


def _y(graph: nx.Graph, ordered_node_idx_list: list[int]) -> torch.Tensor:
    y = np.stack(
        [
            graph.nodes[idx][GraphAttrs.NODE_GROUND_TRUTH]
            for idx in ordered_node_idx_list
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


def _edge_length(graph: nx.Graph, mean_length: float = None):
    edge_lengths = []
    for src, dst, _ in list(graph.edges(data=True)):
        src_coords = np.array(
            [
                graph.nodes[src][GraphAttrs.NODE_X],
                graph.nodes[src][GraphAttrs.NODE_Y],
            ]
        )
        dst_coords = np.array(
            [
                graph.nodes[dst][GraphAttrs.NODE_X],
                graph.nodes[dst][GraphAttrs.NODE_Y],
            ]
        )
        distance = np.linalg.norm(dst_coords - src_coords)
        edge_lengths.append(distance)

    # Normalise if mean edge length is supplied:
    edge_lengths = np.stack(edge_lengths, axis=0)
    if mean_length is not None:
        edge_lengths = np.divide(edge_lengths, mean_length)

    edge_lengths = torch.Tensor(edge_lengths).float()
    # edge_lengths = torch.unsqueeze(edge_lengths, dim=0)
    return edge_lengths


def _edge_orientation(graph: nx.Graph):
    edge_angles = []
    for src, dst, _ in list(graph.edges(data=True)):
        src_coords = np.array(
            [
                graph.nodes[src][GraphAttrs.NODE_X],
                graph.nodes[src][GraphAttrs.NODE_Y],
            ]
        )
        dst_coords = np.array(
            [
                graph.nodes[dst][GraphAttrs.NODE_X],
                graph.nodes[dst][GraphAttrs.NODE_Y],
            ]
        )
        angle = calculate_angle_with_vertical(src_coords, dst_coords)
        edge_angles.append(angle)

    # Normalise if mean edge length is supplied:
    edge_angles = np.stack(edge_angles, axis=0)
    edge_angles = torch.Tensor(edge_angles).float()
    # edge_angles = torch.unsqueeze(edge_angles, dim=0)
    return edge_angles


def calculate_angle_with_vertical(src_point, dst_point):
    """
    Calculate the angle between a line segment and
    the vertical plane (measured from the vertical axis).

    Args:
    x_src (float): x-coordinate of the source point.
    y_src (float): y-coordinate of the source point.
    x_dst (float): x-coordinate of the destination point.
    y_dst (float): y-coordinate of the destination point.

    Returns:
    float: The angle (in radians) between line segment & vertical plane.
    """
    # Calculate the midpoint of the line
    x_src, y_src = src_point
    x_dst, y_dst = dst_point

    x_mid = (x_src + x_dst) / 2.0
    y_mid = (y_src + y_dst) / 2.0

    # Calculate the angle using arctan2, measured from the vertical axis
    angle_rad = np.arctan2(x_mid, y_mid)

    return angle_rad
