import networkx as nx

import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

from grace.base import GraphAttrs
from grace.models.datasets import dataset_from_graph


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
        for node_idx, node in G.nodes(data=True):
            prediction_vector = (n_pred[node_idx], n_probabs[node_idx])
            node[GraphAttrs.NODE_PREDICTION] = prediction_vector

        for e_idx, edge in enumerate(G.edges(data=True)):
            prediction_vector = (e_pred[e_idx], e_probabs[e_idx])
            edge[GraphAttrs.EDGE_PREDICTION] = prediction_vector


def infer_graph_predictions(
    model: torch.nn.Module,
    data_batches: list[Data],
) -> tuple[torch.Tensor]:
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
