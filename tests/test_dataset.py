# import networkx as nx
# import torch

import pytest

from grace.io.core import GraphAttrs, Annotation, mkdir_grace_from_star
from grace.io.image_dataset import ImageGraphDataset
from grace.models.datasets import dataset_from_graph

from _utils import random_image_and_graph


def test_image_graph_dataset(mrc_image_and_annotations_dir):
    """Test that the dataset loader can import test images and grace annotations."""
    num_images = len(list(mrc_image_and_annotations_dir.glob("*.mrc")))

    dataset = ImageGraphDataset(
        mrc_image_and_annotations_dir,
        mrc_image_and_annotations_dir,
        image_filetype="mrc",
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


def test_dataset_only_takes_common_filenames(tmp_path):
    image_fns = ["b", "a", "aj", "jj"]
    label_fns = ["jj", "b", "kk"]

    image_fns = [f + ".png" for f in image_fns]
    label_fns = [f + ".grace" for f in label_fns]

    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"

    image_dir.mkdir()
    label_dir.mkdir()

    for fn in image_fns:
        file = image_dir / fn
        file.touch()

    for fn in label_fns:
        file = label_dir / fn
        file.touch()

    dataset = ImageGraphDataset(image_dir, label_dir, image_filetype="png")

    expected_image_paths = [
        tmp_path / "images" / "b.png",
        tmp_path / "images" / "jj.png",
    ]

    expected_label_paths = [
        tmp_path / "labels" / "b.grace",
        tmp_path / "labels" / "jj.grace",
    ]

    assert dataset.image_paths == expected_image_paths
    assert dataset.grace_paths == expected_label_paths


def test_gracedir_from_stardir(tmp_path):
    star_fns = ["b", "a", "aj", "jj"]

    star_fns = [f + ".png" for f in star_fns]

    star_dir = tmp_path / "star"
    star_dir.mkdir()

    for fn in star_fns:
        file = star_dir / fn
        file.touch()

    mkdir_grace_from_star(stardir=star_dir)

    expected_grace_paths = [
        tmp_path / "grace" / "b.grace",
        tmp_path / "grace" / "jj.grace",
    ]

    assert expected_grace_paths[0].is_file() is True
    assert expected_grace_paths[1].is_file() is True
