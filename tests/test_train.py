import pytest

from grace.base import GraphAttrs, Annotation
from grace.evaluation.metrics_classifier import (
    accuracy_metric,
    confusion_matrix_metric,
)
from grace.training.train import train_model
from grace.models.datasets import dataset_from_graph
from grace.models.classifier import GCNModel

from conftest import random_image_and_graph


class TestTraining:
    @pytest.fixture
    def data_and_model(self, default_rng):
        model = GCNModel(
            input_channels=2,
            hidden_graph_channels=[16, 8],
            hidden_dense_channels=[4, 2],
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
        target = {"graph": graph, "metadata": {"image_filename": "filename"}}
        dataset = dataset_from_graph(graph, mode="sub")

        return (
            dataset,
            model,
            [
                target,
            ],
        )

    @pytest.mark.parametrize(
        "metrics",
        [
            ["accuracy", "confusion_matrix"],
            [accuracy_metric, confusion_matrix_metric],
        ],
    )
    def test_logger_file_exists(self, tmpdir, metrics, data_and_model):
        dataset, model, graph_list = data_and_model

        train_model(
            model=model,
            train_dataset=dataset,
            valid_dataset=dataset,
            valid_target_list=graph_list,
            batch_size=5,
            metrics=metrics,
            log_dir=tmpdir,
            epochs=1,
            valid_graph_ploter_frequency=5,
        )

        assert tmpdir.exists()
