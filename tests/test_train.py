import os
import pytest

from grace.base import GraphAttrs, Annotation
from grace.utils.metrics import accuracy_metric, confusion_matrix_metric
from grace.models.train import train_model
from grace.models.datasets import dataset_from_graph
from grace.models.classifier import GCN

from _utils import random_image_and_graph


class TestTraining:
    @pytest.fixture
    def data_and_model(self, default_rng):
        model = GCN(
            input_channels=2,
            hidden_channels=4,
        )

        _, graph = random_image_and_graph(
            default_rng, num_nodes=10, feature_ndim=2
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

        return dataset, model

    @pytest.mark.parametrize(
        "metrics",
        [
            ["accuracy", "confusion_matrix"],
            [accuracy_metric, confusion_matrix_metric],
        ],
    )
    def test_logger_file_exists(self, tmpdir, metrics, data_and_model):
        dataset, model = data_and_model

        train_model(
            model=model,
            dataset=dataset,
            batch_size=5,
            metrics=metrics,
            log_dir=tmpdir,
            epochs=1,
        )

        assert os.path.exists(tmpdir)
