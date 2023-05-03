from grace.models.classifier import GCN
from grace.models.feature_extractor import FeatureExtractor

import torch
import pytest

import networkx as nx

from _utils import random_image_and_graph

from grace.base import GraphAttrs, Annotation
from grace.models.datasets import dataset_from_graph


@pytest.mark.parametrize("input_channels", [1, 2, 4])
@pytest.mark.parametrize("hidden_channels", [16, 32, 64])
@pytest.mark.parametrize("output_classes", [1, 2, 4])
class TestGCN:
    @pytest.fixture
    def gcn(self, input_channels, hidden_channels, output_classes):
        return GCN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_classes=output_classes,
        )

    def test_model_building(
        self, input_channels, hidden_channels, output_classes, gcn
    ):
        """Test building the model with different dimensions."""

        assert gcn.conv1.in_channels == input_channels
        assert gcn.linear.in_features == hidden_channels
        assert gcn.linear.out_features == output_classes

    @pytest.mark.parametrize("num_nodes", [4, 5])
    def test_output_sizes(
        self,
        input_channels,
        hidden_channels,
        output_classes,
        gcn,
        num_nodes,
        default_rng,
    ):
        _, graph = random_image_and_graph(
            default_rng, num_nodes=num_nodes, feature_ndim=input_channels
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
        data = dataset_from_graph(graph)[0]

        subgraph = nx.ego_graph(graph, 0)
        num_nodes = subgraph.number_of_nodes()
        x, embedding = gcn(x=data.x, edge_index=data.edge_index)

        assert x.size() == (1, output_classes)
        assert embedding.size() == (num_nodes, hidden_channels)


@pytest.mark.parametrize(
    "bbox_size, model",
    [
        ((2, 2), lambda x: torch.sum(x)),
        ((3, 3), lambda x: torch.mean(x)),
    ],
)
class TestFeatureExtractor:
    @pytest.fixture
    def vars(self, bbox_size, model, default_rng):
        image, graph = random_image_and_graph(default_rng)
        return {
            "extractor": FeatureExtractor(
                bbox_size=bbox_size,
                model=model,
                transforms=lambda x: x,
                augmentations=lambda x: x,
            ),
            "image": torch.tensor(image.astype("float32")),
            "graph": graph,
        }

    def test_feature_extractor_forward(self, bbox_size, model, vars):
        image_out, target_out = vars["extractor"](
            vars["image"], {"graph": vars["graph"]}
        )
        graph_out = target_out["graph"]

        assert torch.equal(vars["image"], image_out)

        for node_id, node_attrs in graph_out.nodes.data():
            x, y = node_attrs[GraphAttrs.NODE_X], node_attrs[GraphAttrs.NODE_Y]
            features = node_attrs[GraphAttrs.NODE_FEATURES]

            x_low = int(x - bbox_size[0] / 2)
            x_box = slice(x_low, x_low + bbox_size[0])

            y_low = int(y - bbox_size[1] / 2)
            y_box = slice(y_low, y_low + bbox_size[1])

            bbox_image = vars["image"][..., y_box, x_box]

            assert features == model(bbox_image)
