import numpy as np
import networkx as nx

# Proof of concept - adding / removing random edges & recalculating exact metrics:

# 1. randomly add some edges
# 2. randomly remove some edges
# 3. randomly add & remove some edges


def add_random_edges(G: nx.Graph, num_edges_to_add: int) -> nx.Graph:
    nodes = list(G.nodes())
    node_pairs = np.random.choice(
        nodes, size=(num_edges_to_add, 2), replace=True
    )

    for node1, node2 in node_pairs:
        if not G.has_edge(node1, node2) and node1 != node2:
            G.add_edge(node1, node2)
    return G


def remove_random_edges(G: nx.Graph, num_edges_to_remove: int) -> nx.Graph:
    edges = list(G.edges())
    edges_to_remove = np.random.choice(
        edges, size=num_edges_to_remove, replace=False
    )
    G.remove_edges_from(edges_to_remove)
    return G


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
        modified_graph = add_random_edges(modified_graph, num_edges_to_add)

    if num_edges_to_remove >= 1:
        modified_graph = remove_random_edges(
            modified_graph, num_edges_to_remove
        )

    return modified_graph


# Call the modifier fn:
# graph_new_edges = add_and_remove_random_edges(
#     pred_graph, num_edges_to_add=5, num_edges_to_remove=0
# )
# pred_graph.number_of_edges(), graph_new_edges.number_of_edges()
