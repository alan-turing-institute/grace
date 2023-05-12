import os

from grace.base import GraphAttrs, Annotation
from grace.models.train import train_model
from grace.models.datasets import dataset_from_graph
from grace.models.classifier import GCN

from _utils import random_image_and_graph


def test_logger_file(tmpdir, default_rng):
    gcn = GCN(
        input_channels=2,
        hidden_channels=4,
    )

    _, graph = random_image_and_graph(
        default_rng, num_nodes=16, feature_ndim=2
    )
    graph.update(
        edges=[
            (
                src,
                dst,
                {GraphAttrs.EDGE_GROUND_TRUTH: Annotation.TRUE_POSITIVE},
            )
            for src, dst in graph.edges
        ]
    )
    dataset = dataset_from_graph(graph)

    train_model(
        model=gcn,
        dataset=dataset,
        batch_size=5,
        log_dir=tmpdir,
    )

    assert os.path.exists(tmpdir)
