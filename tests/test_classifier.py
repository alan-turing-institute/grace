import pytest

import networkx as nx

from grace.base import GraphAttrs, Annotation
from grace.models.datasets import dataset_from_graph
from grace.models.classifier import GCNModel

from conftest import random_image_and_graph


@pytest.mark.parametrize("input_channels", [1, 2])
@pytest.mark.parametrize("node_output_classes", [2, 4])
@pytest.mark.parametrize("edge_output_classes", [2, 4])
@pytest.mark.parametrize("hidden_graph_channels", [[16, 4], [128], []])
@pytest.mark.parametrize("hidden_dense_channels", [[16, 4], [128], []])
class TestGCN:
    @pytest.fixture
    def gcn(
        self,
        input_channels,
        hidden_graph_channels,
        hidden_dense_channels,
        node_output_classes,
        edge_output_classes,
    ):
        return GCNModel(
            input_channels=input_channels,
            hidden_graph_channels=hidden_graph_channels,
            hidden_dense_channels=hidden_dense_channels,
            node_output_classes=node_output_classes,
            edge_output_classes=edge_output_classes,
        )

    def test_model_building(
        self,
        input_channels,
        hidden_graph_channels,
        hidden_dense_channels,
        node_output_classes,
        edge_output_classes,
        gcn,
    ):
        """Test building the model with different dimensions."""
        # torch.nn.ModuleList objects are created with no hidden layers:
        if not hidden_graph_channels:
            assert gcn.conv_layer_list is None
        else:
            assert gcn.conv_layer_list is not None

        if not hidden_dense_channels:
            assert gcn.node_dense_list is None
        else:
            assert gcn.node_dense_list is not None

        # match shape of first list items based on hidden features:
        if gcn.conv_layer_list is not None:
            assert gcn.conv_layer_list[0].in_channels == input_channels

        if gcn.conv_layer_list is None and gcn.node_dense_list is not None:
            assert gcn.node_dense_list[0].in_features == input_channels

        # control final classifier layers based on hidden features:
        if hidden_dense_channels:
            assert gcn.node_classifier.in_features == hidden_dense_channels[-1]
            assert gcn.node_classifier.out_features == node_output_classes

            assert (
                gcn.edge_classifier.in_features
                == hidden_dense_channels[-1] * 2
            )
            assert gcn.edge_classifier.out_features == edge_output_classes

        elif hidden_graph_channels:
            assert gcn.node_classifier.in_features == hidden_graph_channels[-1]
            assert gcn.node_classifier.out_features == node_output_classes

            assert (
                gcn.edge_classifier.in_features
                == hidden_graph_channels[-1] * 2
            )
            assert gcn.edge_classifier.out_features == edge_output_classes

        else:
            assert gcn.node_classifier.in_features == input_channels
            assert gcn.node_classifier.out_features == node_output_classes

            assert gcn.edge_classifier.in_features == input_channels * 2
            assert gcn.edge_classifier.out_features == edge_output_classes

    @pytest.mark.parametrize("num_nodes", [4, 5, 8, 10])
    def test_output_sizes(
        self,
        input_channels,
        node_output_classes,
        edge_output_classes,
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
        data = dataset_from_graph(graph, mode="sub")[0]

        subgraph = nx.ego_graph(graph, 0)
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        node_x, edge_x = gcn(x=data.x, edge_index=data.edge_index)

        assert node_x.size() == (num_nodes, node_output_classes)
        assert edge_x.size() == (num_edges, edge_output_classes)
