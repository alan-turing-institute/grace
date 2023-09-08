import networkx as nx
import numpy as np

import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

from grace.base import GraphAttrs
from grace.models.datasets import dataset_from_graph

from grace.evaluation.utils import plot_confusion_matrix_tiles
from grace.evaluation.metrics_classifier import (
    accuracy_metric,
    areas_under_curves_metrics,
)


class GraphLabelPredictor(object):
    """TODO: Fill in."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()

        self.pretrained_gcn = model

    def set_node_and_edge_probabilities(self, G: nx.Graph):
        # Process graph into torch_geometric.data:
        data_batch = dataset_from_graph(
            graph=G, mode="whole", in_train_mode=False
        )
        results = infer_graph_predictions(self.pretrained_gcn, data_batch)
        n_probabs, e_probabs, n_pred, e_pred, _, _ = results

        # Iterate through each node, storing the variables:
        for idx, node in G.nodes(data=True):
            prediction = [int(n_pred[idx].item()), n_probabs[idx].numpy()]
            node[GraphAttrs.NODE_PREDICTION] = prediction

        for e_idx, edge in enumerate(G.edges(data=True)):
            prediction = (int(e_pred[e_idx].item()), e_probabs[e_idx].numpy())
            edge[-1][GraphAttrs.EDGE_PREDICTION] = prediction

    def visualise_performance(self, G: nx.Graph):
        # Prep the data & plot them:
        node_true = [
            node[GraphAttrs.NODE_GROUND_TRUTH]
            for _, node in G.nodes(data=True)
        ]
        node_pred = [
            node[GraphAttrs.NODE_PREDICTION][0]
            for _, node in G.nodes(data=True)
        ]
        node_probabs = np.array(
            [
                node[GraphAttrs.NODE_PREDICTION][1]
                for _, node in G.nodes(data=True)
            ]
        )

        edge_true = [
            edge[GraphAttrs.EDGE_GROUND_TRUTH]
            for _, _, edge in G.edges(data=True)
        ]
        edge_pred = [
            edge[GraphAttrs.EDGE_PREDICTION][0]
            for _, _, edge in G.edges(data=True)
        ]
        edge_probabs = np.array(
            [
                edge[GraphAttrs.EDGE_PREDICTION][1]
                for _, _, edge in G.edges(data=True)
            ]
        )

        node_acc, edge_acc = accuracy_metric(
            node_pred, edge_pred, node_true, edge_true
        )
        areas_under_curves_metrics(
            node_probabs, edge_probabs, node_true, edge_true, figsize=(10, 4)
        )
        plot_confusion_matrix_tiles(node_pred, edge_pred, node_true, edge_true)
        return node_acc, edge_acc


def infer_graph_predictions(
    model: torch.nn.Module,
    data_batches: list[Data],
) -> tuple[torch.Tensor]:
    """TODO: Clean this fn."""

    # Instantiate the vectors to return:
    node_softmax_preds = []
    edge_softmax_preds = []
    node_argmax_preds = []
    edge_argmax_preds = []
    node_labels = []
    edge_labels = []

    # Predict labels from sub-graph:
    for data in tqdm(data_batches, desc="Predicting for the entire graph: "):
        # Get the ground truth labels:
        node_labels.extend(data.y)
        edge_labels.extend(data.edge_label)

        # Get the model predictions:
        node_x, edge_x = model.predict(x=data.x, edge_index=data.edge_index)

        # Process node probs into classes predictions:
        node_soft = node_x.softmax(dim=1)
        node_softmax_preds.extend(node_soft)
        node_arg = node_soft.argmax(dim=1).long()
        node_argmax_preds.extend(node_arg)

        # Process edge probs into classes predictions:
        edge_soft = edge_x.softmax(dim=1)
        edge_softmax_preds.extend(edge_soft)
        edge_arg = edge_soft.argmax(dim=1).long()
        edge_argmax_preds.extend(edge_arg)

    # Stack the results:
    node_softmax_preds = torch.stack(node_softmax_preds, axis=0)
    edge_softmax_preds = torch.stack(edge_softmax_preds, axis=0)
    node_argmax_preds = torch.stack(node_argmax_preds, axis=0)
    edge_argmax_preds = torch.stack(edge_argmax_preds, axis=0)
    node_labels = torch.stack(node_labels, axis=0)
    edge_labels = torch.stack(edge_labels, axis=0)

    return (
        node_softmax_preds,
        edge_softmax_preds,
        node_argmax_preds,
        edge_argmax_preds,
        node_labels,
        edge_labels,
    )
