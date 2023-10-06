import pytest

from grace.base import GraphAttrs, Annotation
from grace.models.datasets import dataset_from_graph

from conftest import random_image_and_graph


def test_dataset_ignores_subgraph_if_all_edges_unknown(default_rng):
    _, graph = random_image_and_graph(default_rng)

    edge_update = [
        (src, dst, {GraphAttrs.EDGE_GROUND_TRUTH: Annotation.UNKNOWN})
        for src, dst in graph.edges
    ]
    graph.update(edges=edge_update)
    # this action is not currently required since edges are by default UNKNOWN;
    # however it enables testing of this condition should the default label be changed

    assert dataset_from_graph(graph, mode="sub") == []


@pytest.mark.parametrize("num_unknown", [7, 17])
def test_dataset_ignores_subgraph_if_central_node_unknown(
    num_unknown, default_rng
):
    num_nodes_total = 20
    _, graph = random_image_and_graph(
        default_rng,
        num_nodes=num_nodes_total,
    )

    edge_update = [
        (src, dst, {GraphAttrs.EDGE_GROUND_TRUTH: Annotation.TRUE_POSITIVE})
        for src, dst in graph.edges
    ]
    graph.update(edges=edge_update)

    node_update = [
        (node, {GraphAttrs.NODE_GROUND_TRUTH: Annotation.UNKNOWN})
        for node in list(graph.nodes)[:num_unknown]
    ]
    node_update += [
        (node, {GraphAttrs.NODE_GROUND_TRUTH: Annotation.TRUE_POSITIVE})
        for node in list(graph.nodes)[num_unknown:]
    ]
    graph.update(nodes=node_update)

    assert (
        len(dataset_from_graph(graph, mode="sub"))
        == num_nodes_total - num_unknown
    )
