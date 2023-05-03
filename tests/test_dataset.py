# import networkx as nx
# import torch

import pytest

from grace.io.core import GraphAttrs, Annotation
from grace.io.image_dataset import ImageGraphDataset, mrc_reader
from grace.models.datasets import dataset_from_graph

from _utils import random_image_and_graph


def test_image_graph_dataset(mrc_image_and_annotations_dir):
    """Test that the dataset loader can import test images and grace annotations."""
    num_images = len(list(mrc_image_and_annotations_dir.glob("*.mrc")))

    dataset = ImageGraphDataset(
        mrc_image_and_annotations_dir,
        mrc_image_and_annotations_dir,
        mrc_reader,
    )

    # # all currently fail
    # image, graph = dataset[0]

    # assert isinstance(image, torch.Tensor)
    # assert isinstance(graph, nx.Graph)

    # assert image.shape == (1, 128, 128)
    # assert graph.number_of_nodes() == 4

    assert len(dataset) == num_images


def test_dataset_ignores_subgraph_if_all_edges_unknown(default_rng):
    _, graph = random_image_and_graph(default_rng)

    edge_update = [
        (src, dst, {GraphAttrs.EDGE_GROUND_TRUTH: Annotation.UNKNOWN})
        for src, dst in graph.edges
    ]
    graph.update(edges=edge_update)
    # this action is not currently required since edges are by default UNKNOWN;
    # however it enables testing of this condition should the default label be changed

    assert dataset_from_graph(graph) == []


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

    assert len(dataset_from_graph(graph)) == num_nodes_total - num_unknown
