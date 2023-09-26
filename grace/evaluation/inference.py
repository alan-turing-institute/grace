import networkx as nx
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch_geometric.data import Data

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

from grace.base import GraphAttrs, Annotation, Prediction
from grace.models.datasets import dataset_from_graph
from grace.visualisation.plotting import (
    plot_confusion_matrix_tiles,
    plot_areas_under_curves,
    plot_prediction_probabilities_hist,
    visualise_node_and_edge_probabilities,
)


class GraphLabelPredictor(object):
    """Loader for pre-trained classifier model & update
    predictions probabilities for nodes & edges of an
    individual graph, or entire inference batch dataset.

    Parameters
    ----------
    model : str | torch.nn.Module
        (Path to) Pre-trained classifier model.

    Notes
    -----
    - Updates the graph node / edge prediction probabilities
    via the `set_node_and_edge_probabilities` method.
    - Modifies the input graph in place (returns None)

    Parameters
    ----------
    model : str | torch.nn.Module
        (Path to) Pre-trained classifier model.

    """

    def __init__(
        self,
        model: str | torch.nn.Module,
    ) -> None:
        super().__init__()

        if isinstance(model, str):
            assert Path(model).is_file()
        elif isinstance(model, Path):
            assert model.is_file()

        self.pretrained_gcn = torch.load(model)
        self.pretrained_gcn.eval()

    def infer_graph_predictions(
        self, data_batches: list[Data], verbose: bool = False
    ) -> tuple[npt.NDArray]:
        """Returns stacks of true & pred labels from graph dataset(s)."""

        # Instantiate the vectors to return:
        node_softmax_preds = []
        edge_softmax_preds = []
        node_labels = []
        edge_labels = []

        # Predict labels from sub-graph:
        desc = "Inferring predictions from graph data batches: "
        for data in tqdm(data_batches, desc=desc, disable=not verbose):
            # Get the ground truth labels:
            node_labels.extend(data.y)
            edge_labels.extend(data.edge_label)

            # Get the model predictions:
            node_x, edge_x = self.pretrained_gcn.predict(
                x=data.x, edge_index=data.edge_index
            )

            # Process node probs into classes predictions:
            node_soft = node_x.softmax(dim=1).numpy()
            node_softmax_preds.extend(node_soft)

            # Process edge probs into classes predictions:
            edge_soft = edge_x.softmax(dim=1).numpy()
            edge_softmax_preds.extend(edge_soft)

        # Stack the results & return:
        return (
            np.stack(node_softmax_preds, axis=0),
            np.stack(edge_softmax_preds, axis=0),
            np.stack(node_labels, axis=0),
            np.stack(edge_labels, axis=0),
        )

    def set_node_and_edge_probabilities(self, G: nx.Graph) -> None:
        """Update predicted node / edge probs by Prediction in-place."""

        data_batch = dataset_from_graph(
            graph=G, mode="whole", in_train_mode=False
        )
        # Process graph into torch_geometric.data:
        n_probs, e_probs, _, _ = self.infer_graph_predictions(data_batch)

        # Iterate through each node, storing the 3 class predictions:
        for n_idx, node in G.nodes(data=True):
            prediction = Prediction(np.append(n_probs[n_idx], 0.0))
            node[GraphAttrs.NODE_PREDICTION] = prediction

        for e_idx, edge in enumerate(G.edges(data=True)):
            prediction = Prediction(np.append(e_probs[e_idx], 0.0))
            edge[-1][GraphAttrs.EDGE_PREDICTION] = prediction

    def visualise_prediction_probs_on_graph(self, G: nx.Graph) -> None:
        """Visualise predictions as shaded points (nodes) / lines (edges)."""
        assert all(GraphAttrs.NODE_PREDICTION in G.nodes[n] for n in G.nodes)
        assert all(GraphAttrs.EDGE_PREDICTION in G.edges[e] for e in G.edges)
        fig = visualise_node_and_edge_probabilities(G=G)
        return fig

    def get_predictions_for_entire_batch(
        self,
        infer_target_list: list[dict[str]],
        verbose: bool = False,
    ) -> tuple[npt.NDArray]:
        """Processes predictions for list of graph targets."""

        data = [[] for _ in range(6)]

        for target in tqdm(infer_target_list, disable=not verbose):
            # Read & update the graph:
            G = target["graph"]
            self.set_node_and_edge_probabilities(G=G)

            # Store the node & edge data into vectors:
            for _, n in G.nodes(data=True):
                label = n[GraphAttrs.NODE_GROUND_TRUTH]
                if label == Annotation.UNKNOWN:
                    continue
                data[0].append(label)
                data[1].append(n[GraphAttrs.NODE_PREDICTION].label)
                data[2].append(n[GraphAttrs.NODE_PREDICTION].prob_TP)

            for _, _, e in G.edges(data=True):
                label = e[GraphAttrs.EDGE_GROUND_TRUTH]
                if label == Annotation.UNKNOWN:
                    continue
                data[3].append(label)
                data[4].append(e[GraphAttrs.EDGE_PREDICTION].label)
                data[5].append(e[GraphAttrs.EDGE_PREDICTION].prob_TP)

        # Sanity checks:
        assert len(data[0]) == len(data[1]) == len(data[2])
        assert len(data[3]) == len(data[4]) == len(data[5])
        data = [np.array(item) for item in data]

        n_true, n_pred, n_prob, e_true, e_pred, e_prob = data
        return n_true, n_pred, n_prob, e_true, e_pred, e_prob

    def calculate_numerical_results(
        self,
        infer_target_list: list[dict[str]],
        verbose: bool = False,
    ) -> dict[str, float]:
        """Calculates & return dictionary of numerical metrics for batch."""
        # Get the batch probs & predictions:
        data = self.get_predictions_for_entire_batch(
            infer_target_list=infer_target_list, verbose=verbose
        )
        n_true, n_pred, n_prob, e_true, e_pred, e_prob = data

        inference_batch_metrics = {}

        # Accuracy:
        inference_batch_metrics["Batch accuracy (nodes)"] = accuracy_score(
            y_pred=n_pred, y_true=n_true, normalize=True
        )
        inference_batch_metrics["Batch accuracy (edges)"] = accuracy_score(
            y_pred=e_pred, y_true=e_true, normalize=True
        )

        # PRF1:
        prf1_nodes = precision_recall_fscore_support(
            y_pred=n_pred, y_true=n_true, average="binary"
        )
        prf1_edges = precision_recall_fscore_support(
            y_pred=e_pred, y_true=e_true, average="binary"
        )
        inference_batch_metrics["Batch precision (nodes)"] = prf1_nodes[0]
        inference_batch_metrics["Batch precision (edges)"] = prf1_edges[0]
        inference_batch_metrics["Batch recall (nodes)"] = prf1_nodes[1]
        inference_batch_metrics["Batch recall (edges)"] = prf1_edges[1]
        inference_batch_metrics["Batch f1-score (nodes)"] = prf1_nodes[2]
        inference_batch_metrics["Batch f1-score (edges)"] = prf1_edges[2]

        # AUC scores:
        inference_batch_metrics["Batch AUROC (nodes)"] = roc_auc_score(
            y_true=n_true, y_score=n_prob
        )
        inference_batch_metrics["Batch AUROC (edges)"] = roc_auc_score(
            y_true=e_true, y_score=e_prob
        )

        # AP scores:
        inference_batch_metrics[
            "Batch avg precision (nodes)"
        ] = average_precision_score(y_true=n_true, y_score=n_prob)
        inference_batch_metrics[
            "Batch avg precision (edges)"
        ] = average_precision_score(y_true=e_true, y_score=e_prob)

        # Check all metric outputs are floats & return
        assert all(
            isinstance(item, float)
            for item in inference_batch_metrics.values()
        )
        return inference_batch_metrics

    def visualise_model_performance_on_entire_batch(
        self,
        infer_target_list: list[dict[str]],
        *,
        save_figures: str | Path = None,
        show_figures: bool = False,
        verbose: bool = False,
    ) -> None:
        """Visualise and/or saves out performance plots for batch."""
        # Create the save path:
        if save_figures is not None:
            save_path = save_figures
            if isinstance(save_path, str):
                save_path = Path(save_path)
            assert save_path.is_dir()
            save_figures = True
        else:
            save_figures = False

        # Process batch data predictions:
        data = self.get_predictions_for_entire_batch(
            infer_target_list=infer_target_list, verbose=verbose
        )
        n_true, n_pred, n_prob, e_true, e_pred, e_prob = data

        # Confusion matrix tiles:
        plot_confusion_matrix_tiles(n_pred, e_pred, n_true, e_true)
        if save_figures is True:
            plt.savefig(save_path / "Batch_Dataset-Confusion_Matrix_Tiles.png")
        if show_figures is True:
            plt.show()
        plt.close()

        # Areas under curves:
        plot_areas_under_curves(n_prob, e_prob, n_true, e_true)
        if save_figures is True:
            plt.savefig(save_path / "Batch_Dataset-Areas_Under_Curves.png")
        if show_figures is True:
            plt.show()
        plt.close()

        # Predicted probs hist:
        plot_prediction_probabilities_hist(n_pred, e_pred, n_true, e_true)
        if save_figures is True:
            plt.savefig(save_path / "Batch_Dataset-Prediction_Probs_Hist.png")
        if show_figures is True:
            plt.show()
        plt.close()
