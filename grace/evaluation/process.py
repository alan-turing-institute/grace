import networkx as nx
import numpy as np

from grace.base import GraphAttrs


def add_and_remove_random_edges(
    G: nx.Graph, num_edges_to_add: int, num_edges_to_remove: int
) -> nx.Graph:
    """Adds random edges and removes random edges from the given graph.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    num_edges_to_add : int
        The number of random edges to add.
    num_edges_to_remove : int
        The number of random edges to remove.

    Returns
    -------
        nx.Graph: The graph with added and removed random edges.
    """
    # Create a copy, not to modify the original graph in-place:
    modified_graph = G.copy()

    if num_edges_to_add >= 1:
        modified_graph = _add_random_edges(modified_graph, num_edges_to_add)

    if num_edges_to_remove >= 1:
        modified_graph = _remove_random_edges(
            modified_graph, num_edges_to_remove
        )

    return modified_graph


def _add_random_edges(G: nx.Graph, num_edges_to_add: int) -> nx.Graph:
    nodes = list(G.nodes())
    node_pairs = np.random.choice(
        nodes, size=(num_edges_to_add, 2), replace=True
    )

    for node1, node2 in node_pairs:
        if not G.has_edge(node1, node2) and node1 != node2:
            G.add_edge(node1, node2)
    return G


def _remove_random_edges(G: nx.Graph, num_edges_to_remove: int) -> nx.Graph:
    edges = list(G.edges())
    edges_to_remove = np.random.choice(
        edges, size=num_edges_to_remove, replace=False
    )
    G.remove_edges_from(edges_to_remove)
    return G


def update_graph_with_dummy_predictions(
    G: nx.Graph,
    node_confidence: float = 0.5,
    edge_confidence: float = 0.1,
) -> None:
    """Create a random graph with line objects.

    Parameters
    ----------
    G : nx.Graph
        The graph which node & edge predictions are to be updated with synthetic values.

    Returns
    -------
    None

    Notes
    -----
    Modifies the graph in place.

    """
    # Make sure all objects are labelled according to object identity:
    nodes = list(G.nodes.data())
    obj_identity = ["object_idx" in node for _, node in nodes]
    assert all(obj_identity)

    for _, node in nodes:
        pd = np.random.random() * node_confidence
        if node["label"] > 0:
            node[GraphAttrs.NODE_PREDICTION] = pd
        else:
            node[GraphAttrs.NODE_PREDICTION] = 1 - pd

    for edge in G.edges.data():
        pd = np.random.random() * edge_confidence
        _, e_i = nodes[edge[0]]
        _, e_j = nodes[edge[1]]

        if e_i["object_idx"] == e_j["object_idx"] and e_i["label"] > 0:
            edge[2][GraphAttrs.EDGE_PREDICTION] = 1 - pd
        else:
            edge[2][GraphAttrs.EDGE_PREDICTION] = pd


def generate_ground_truth_graph(random_graph: nx.Graph):
    """Generate a ground truth graph from the graph annotation.

    Parameters
    ----------
    random_graph: nx.Graph
        Annotated graph with all objects assigned an identity
        Each node must contain an ['object_idx'] key, Error otherwise.

    Returns
    -------
    GT_graph: nx.Graph
        Ground truth graph with all maintained properties.
    """
    # Make sure all objects are labelled according to object identity:
    nodes = list(random_graph.nodes.data())
    obj_identity = ["object_idx" in node for _, node in nodes]
    assert all(obj_identity)

    # Copy graph information:
    GT_graph = random_graph.copy()

    # Shorlist edges to delete from GT, maintaining the edge properties:
    edges_to_remove = []
    for src, dst in random_graph.edges(data=False):
        _, node_st = nodes[src]
        _, node_en = nodes[dst]

        # If identities do no match, it's not the same object:
        if node_st["object_idx"] != node_en["object_idx"]:
            edges_to_remove.append((src, dst))
        else:
            # If they match, but between two non-object nodes:
            if node_st["label"] < 1:
                edges_to_remove.append((src, dst))

    # Delete the shorlisted edges:
    GT_graph.remove_edges_from(edges_to_remove)

    return GT_graph


# def imply_annotations_from_dummy_predictions(G: nx.Graph) -> None:
#     """TODO: Fill in. HACK: This code doesn't account for UNKNOWN annotations."""

#     for _, node in G.nodes(data=True):
#         if node[GraphAttrs.NODE_PREDICTION] >= 0.5:
#             node[GraphAttrs.NODE_GROUND_TRUTH] = 1
#         else:
#             node[GraphAttrs.NODE_GROUND_TRUTH] = 0

#     for _, _, edge in G.edges(data=True):
#         if edge[GraphAttrs.EDGE_PREDICTION] >= 0.5:
#             edge[GraphAttrs.EDGE_GROUND_TRUTH] = 1
#         else:
#             edge[GraphAttrs.EDGE_GROUND_TRUTH] = 0


# def imply_dummy_predictions_from_annotations(G: nx.Graph) -> None:
#     """TODO: Fill in.
#     HACK: This code doesn't account for UNKNOWN annotations.
#     TODO: This code doesn't distinguish individual objects.
#           Account for an object_index where possible!
#     """

#     for _, node in G.nodes(data=True):
#         pd = np.random.random() * 0.5
#         if node[GraphAttrs.NODE_GROUND_TRUTH] == 1:
#             node[GraphAttrs.NODE_PREDICTION] = 1 - pd
#         else:
#             node[GraphAttrs.NODE_PREDICTION] = pd

#     for _, _, edge in G.edges(data=True):
#         pd = np.random.random() * 0.1
#         if node[GraphAttrs.EDGE_GROUND_TRUTH] == 1:
#             node[GraphAttrs.EDGE_PREDICTION] = 1 - pd
#         else:
#             node[GraphAttrs.EDGE_PREDICTION] = pd
