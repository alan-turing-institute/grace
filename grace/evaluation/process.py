import networkx as nx
import numpy as np

from grace.base import GraphAttrs, Annotation, Prediction


def update_graph_with_dummy_predictions(
    G: nx.Graph,
    node_uncertainty: float = 0.5,
    edge_uncertainty: float = 0.1,
) -> None:
    """Updates dummy graph prediction instead of node / edge classifier.
        Uses random values but assignes all predictions correctly (default).

    Parameters
    ----------
    G : nx.Graph
        The graph which node & edge predictions to updated with random values
    node_uncertainty : float
        Confidence limit for assigning the probability values.
        Accepts values ranging from (0.0, 0.5]. Defaults to 0.5
    edge_uncertainty : float
        Confidence limit for assigning the probability values.
        Accepts values ranging from (0.0, 0.5]. Defaults to 0.1

    Returns
    -------
    None (modifies the input graph in-place)

    Notes
    -----
    - Modifies the graph in place.
    - The 'uncertainty' parameter separates the distributions of predictions
        for the respective classes, e.g. 0.5 will produce a distribution of
        attribute values for TN between [0.0, 0.5) & TP between (0.5, 1.0]
        which are very close, whilst 0.1 will force larger separation between
        them, i.e. TN between [0.0, 0.1) & TP between (0.9, 1.0].
    - Only motifs with the same object identity should be linked (TODO)
    """
    # Make sure that the uncertainties range within the expected threshold:
    mn, mx = 0.0, 1.0
    if not (mn <= node_uncertainty <= mx):
        raise ValueError(
            "Node uncertainty value must be between "
            f"{mn} and {mx}, but got {node_uncertainty}"
        )
    if not (mn <= edge_uncertainty <= mx):
        raise ValueError(
            "Edge uncertainty value must be between "
            f"{mn} and {mx}, but got {edge_uncertainty}"
        )

    # Make sure all objects have their GT label in some form:
    nodes = list(G.nodes.data())
    obj_identity = ["object_idx" in n and "label" in n for _, n in nodes]
    lbl_identity = [GraphAttrs.NODE_GROUND_TRUTH in n for _, n in nodes]
    assert all(obj_identity) or all(lbl_identity)

    # Iterate through all the nodes:
    for _, node in nodes:
        pd = np.random.random() * node_uncertainty

        if all(obj_identity):
            # true positive node:
            if node["label"] > 0:
                pred = Prediction(np.array([pd, 1 - pd, 0.0]))
            else:
                pred = Prediction(np.array([1 - pd, pd, 0.0]))
            node[GraphAttrs.NODE_PREDICTION] = pred

        if all(lbl_identity):
            # true positive node:
            if node[GraphAttrs.NODE_GROUND_TRUTH] == Annotation.TRUE_POSITIVE:
                pred = Prediction(np.array([pd, 1 - pd, 0.0]))
            else:
                pred = Prediction(np.array([1 - pd, pd, 0.0]))
            node[GraphAttrs.NODE_PREDICTION] = pred

    # Iterate through all the edges:
    for edge in G.edges.data():
        pd = np.random.random() * edge_uncertainty
        _, e_i = nodes[edge[0]]
        _, e_j = nodes[edge[1]]

        if all(obj_identity):
            # true positive edge:
            if e_i["object_idx"] == e_j["object_idx"] and e_i["label"] > 0:
                pred = Prediction(np.array([pd, 1 - pd, 0.0]))
            else:
                pred = Prediction(np.array([1 - pd, pd, 0.0]))
            edge[-1][GraphAttrs.EDGE_PREDICTION] = pred

        if all(lbl_identity):
            # true positive edge:
            if (
                edge[-1][GraphAttrs.EDGE_GROUND_TRUTH]
                == Annotation.TRUE_POSITIVE
            ):
                pred = Prediction(np.array([pd, 1 - pd, 0.0]))
            else:
                pred = Prediction(np.array([1 - pd, pd, 0.0]))
            edge[-1][GraphAttrs.EDGE_PREDICTION] = pred


def _remove_edges_from_ObjectIndex(GT_graph: nx.Graph) -> None:
    """Modifies graph in-place."""
    nodes = list(GT_graph.nodes.data())

    # Shorlist edges to delete from GT, maintaining the edge properties:
    edges_to_remove = []
    for src, dst in GT_graph.edges(data=False):
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


def _remove_edges_from_GraphAttrs(GT_graph: nx.Graph) -> None:
    """Modifies graph in-place."""

    # Shorlist edges to delete from GT, maintaining the edge properties:
    edges_to_remove = []
    for src, dst, edge in GT_graph.edges(data=True):
        # If edges are not labelled as TP, remove them:
        if edge[GraphAttrs.EDGE_GROUND_TRUTH] != Annotation.TRUE_POSITIVE:
            edges_to_remove.append((src, dst))

    # Delete the shorlisted edges:
    GT_graph.remove_edges_from(edges_to_remove)


def generate_ground_truth_graph(graph: nx.Graph):
    """Generate a ground truth graph from the graph annotation.

    Parameters
    ----------
    graph: nx.Graph
        Annotated graph with all objects assigned an identity

    Returns
    -------
    GT_graph: nx.Graph
        Ground truth graph with all maintained properties.

    Notes
    -----
    - Needs either "object_idx" or GraphAttrs.NODE_GROUND_TRUTH labels.
    """
    # Copy graph information:
    GT_graph = graph.copy()

    # List all nodes & their attributes:
    nodes = list(graph.nodes.data())
    _, single_node = nodes[0]

    # Make sure all objects are somehow labelled:

    # according to object identity:
    if "object_idx" in single_node:
        _remove_edges_from_ObjectIndex(GT_graph)

        # according to GraphAttrs from annotation:
    elif GraphAttrs.NODE_GROUND_TRUTH in single_node:
        _remove_edges_from_GraphAttrs(GT_graph)

        # if not, you cannot continue...
    else:
        raise KeyError("There is no ground truth information")

    return GT_graph


def assume_annotations_from_dummy_predictions(G: nx.Graph) -> None:
    """Translates between dummy predictions & GT labels."""

    for _, node in G.nodes(data=True):
        pred_label = node[GraphAttrs.NODE_PREDICTION].label
        node[GraphAttrs.NODE_GROUND_TRUTH] = Annotation(pred_label)

    for _, _, edge in G.edges(data=True):
        pred_label = node[GraphAttrs.NODE_PREDICTION].label
        edge[GraphAttrs.EDGE_GROUND_TRUTH] = Annotation(pred_label)


def add_and_remove_random_edges(
    graph_to_modify: nx.Graph,
    Delaunay_graph: nx.Graph,
    num_edges_to_add: int,
    num_edges_to_remove: int,
) -> nx.Graph:
    """Adds and/or removes random edges from the given graph.

    Parameters
    ----------
    graph_to_modify : nx.Graph
        The input graph which will be modified & returned.
    Delaunay_graph : nx.Graph
        The graph with all possible edges, as triangulated.
    num_edges_to_add : int
        The number of random edges to add.
    num_edges_to_remove : int
        The number of random edges to remove.

    Returns
    -------
    modified_graph : nx.Graph
        The modified graph with added and/or removed random edges.
    """
    # Create a copy, not to modify the original graph in-place:
    modified_graph = graph_to_modify.copy()

    if num_edges_to_add >= 1:
        all_edges = set(Delaunay_graph.edges(data=False))
        modifiable_edges = set(graph_to_modify.edges(data=False))
        possible_edges = list(all_edges - modifiable_edges)

        modified_graph = _add_random_edges(
            modified_graph, num_edges_to_add, possible_edges
        )
    if num_edges_to_remove >= 1:
        modified_graph = _remove_random_edges(
            modified_graph, num_edges_to_remove
        )

    return modified_graph


def _add_random_edges(
    G: nx.Graph, num_edges_to_add: int, possible_edges: list[tuple[int, int]]
) -> nx.Graph:
    edges_to_add = np.random.choice(
        range(len(possible_edges)), size=num_edges_to_add, replace=False
    )
    edges_to_add = [possible_edges[e] for e in edges_to_add]
    for node1, node2 in edges_to_add:
        G.add_edge(node1, node2)
        G[node1][node2][
            GraphAttrs.EDGE_GROUND_TRUTH
        ] = Annotation.TRUE_POSITIVE
    return G


def _remove_random_edges(G: nx.Graph, num_edges_to_remove: int) -> nx.Graph:
    edges = list(G.edges(data=False))
    edges_to_remove = np.random.choice(
        range(len(edges)), size=num_edges_to_remove, replace=False
    )
    edges_to_remove = [edges[e] for e in edges_to_remove]
    G.remove_edges_from(edges_to_remove)
    return G
