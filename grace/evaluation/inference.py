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
        model: str | Path | torch.nn.Module,
    ) -> None:
        super().__init__()

        if isinstance(model, str):
            assert Path(model).is_file()
            model = torch.load(model)
        elif isinstance(model, Path):
            assert model.is_file()
            model = torch.load(model)

        self.pretrained_model = model
        self.pretrained_model.eval()

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
            node_x, edge_x = self.pretrained_model.predict(
                x=data.x,
                edge_index=data.edge_index,
                edge_length=data.edge_length,
                edge_orient=data.edge_orient,
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
        # Process graph into torch_geometric.data:
        data_batch = dataset_from_graph(graph=G, mode="whole")

        # Process graph into torch_geometric.data:
        n_probs, e_probs, _, _ = self.infer_graph_predictions(data_batch)

        # Iterate through each node, storing the 3 class predictions:
        for n_idx, node in G.nodes(data=True):
            prediction = Prediction(np.append(n_probs[n_idx], 0.0))
            node[GraphAttrs.NODE_PREDICTION] = prediction

        for e_idx, edge in enumerate(G.edges(data=True)):
            prediction = Prediction(np.append(e_probs[e_idx], 0.0))
            edge[-1][GraphAttrs.EDGE_PREDICTION] = prediction

    def visualise_prediction_probs_on_graph(
        self,
        G: nx.Graph,
        *,
        graph_filename: str = "Graph",
        save_figure: str | Path = None,
        show_figure: bool = False,
    ) -> None:
        """Visualise predictions as shaded points (nodes) / lines (edges)."""
        assert all(GraphAttrs.NODE_PREDICTION in G.nodes[n] for n in G.nodes)
        assert all(GraphAttrs.EDGE_PREDICTION in G.edges[e] for e in G.edges)

        # Create the save path:
        if save_figure is not None:
            save_path = save_figure
            if isinstance(save_path, str):
                save_path = Path(save_path)
            assert save_path.is_dir()
            save_figure = True
        else:
            save_figure = False

        # Plot the thing:
        visualise_node_and_edge_probabilities(G=G)
        if save_figure is True:
            plt.savefig(
                save_path / f"{graph_filename}-Whole_Graph_Visualisation.png"
            )
        if show_figure is True:
            plt.show()
        plt.close()

    def get_predictions_for_entire_batch(
        self,
        infer_target_list: list[dict[str]],
        verbose: bool = False,
    ) -> dict[str, npt.NDArray]:
        """Processes predictions for list of graph targets."""

        predictions_data = {
            "n_true": [],
            "n_pred": [],
            "n_prob": [],
            "e_true": [],
            "e_pred": [],
            "e_prob": [],
        }

        for target in tqdm(infer_target_list, disable=not verbose):
            # Read & update the graph:
            G = target["graph"]
            self.set_node_and_edge_probabilities(G=G)

            # Store the node & edge data into vectors:
            for _, n in G.nodes(data=True):
                label = n[GraphAttrs.NODE_GROUND_TRUTH]
                if label == Annotation.UNKNOWN:
                    continue
                predictions_data["n_true"].append(label)
                predictions_data["n_pred"].append(
                    n[GraphAttrs.NODE_PREDICTION].label
                )
                predictions_data["n_prob"].append(
                    n[GraphAttrs.NODE_PREDICTION].prob_TP
                )

            for _, _, e in G.edges(data=True):
                label = e[GraphAttrs.EDGE_GROUND_TRUTH]
                if label == Annotation.UNKNOWN:
                    continue
                predictions_data["e_true"].append(label)
                predictions_data["e_pred"].append(
                    e[GraphAttrs.EDGE_PREDICTION].label
                )
                predictions_data["e_prob"].append(
                    e[GraphAttrs.EDGE_PREDICTION].prob_TP
                )

        # Sanity checks:
        assert (
            len(predictions_data["n_true"])
            == len(predictions_data["n_pred"])
            == len(predictions_data["n_prob"])
        )
        assert (
            len(predictions_data["e_true"])
            == len(predictions_data["e_pred"])
            == len(predictions_data["e_prob"])
        )

        # Return as dict[str, np.NDArray]:
        predictions_data = {
            key: np.array(value) for key, value in predictions_data.items()
        }
        return predictions_data

    def calculate_numerical_results_on_entire_batch(
        self,
        infer_target_list: list[dict[str]],
        verbose: bool = False,
    ) -> dict[str, float]:
        """Calculates & return dictionary of numerical metrics for batch."""
        # Get the batch probs & predictions:
        predictions_data = self.get_predictions_for_entire_batch(
            infer_target_list=infer_target_list, verbose=verbose
        )

        inference_batch_metrics = {}

        # Accuracy:
        inference_batch_metrics["Batch accuracy (nodes)"] = accuracy_score(
            y_pred=predictions_data["n_pred"],
            y_true=predictions_data["n_true"],
            normalize=True,
        )
        inference_batch_metrics["Batch accuracy (edges)"] = accuracy_score(
            y_pred=predictions_data["e_pred"],
            y_true=predictions_data["e_true"],
            normalize=True,
        )

        # PRF1:
        prf1_nodes = precision_recall_fscore_support(
            y_pred=predictions_data["n_pred"],
            y_true=predictions_data["n_true"],
            average="weighted",
            beta=1.0,
            zero_division=0.0,
        )
        prf1_edges = precision_recall_fscore_support(
            y_pred=predictions_data["e_pred"],
            y_true=predictions_data["e_true"],
            average="weighted",
            beta=1.0,
            zero_division=0.0,
        )
        inference_batch_metrics["Batch precision (nodes)"] = prf1_nodes[0]
        inference_batch_metrics["Batch precision (edges)"] = prf1_edges[0]
        inference_batch_metrics["Batch recall (nodes)"] = prf1_nodes[1]
        inference_batch_metrics["Batch recall (edges)"] = prf1_edges[1]
        inference_batch_metrics["Batch F1-score (nodes)"] = prf1_nodes[2]
        inference_batch_metrics["Batch F1-score (edges)"] = prf1_edges[2]

        # AUC scores:
        inference_batch_metrics["Batch AUROC (nodes)"] = roc_auc_score(
            y_true=predictions_data["n_true"],
            y_score=predictions_data["n_prob"],
        )
        inference_batch_metrics["Batch AUROC (edges)"] = roc_auc_score(
            y_true=predictions_data["e_true"],
            y_score=predictions_data["e_prob"],
        )

        # AP scores:
        inference_batch_metrics[
            "Batch avg precision (nodes)"
        ] = average_precision_score(
            y_true=predictions_data["n_true"],
            y_score=predictions_data["n_prob"],
        )
        inference_batch_metrics[
            "Batch avg precision (edges)"
        ] = average_precision_score(
            y_true=predictions_data["e_true"],
            y_score=predictions_data["e_prob"],
        )

        # Check all metric outputs are floats & return
        inference_batch_metrics = {
            key: float(value) for key, value in inference_batch_metrics.items()
        }
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
        predictions_data = self.get_predictions_for_entire_batch(
            infer_target_list=infer_target_list, verbose=verbose
        )

        # Confusion matrix tiles:
        plot_confusion_matrix_tiles(
            node_pred=predictions_data["n_pred"],
            edge_pred=predictions_data["e_pred"],
            node_true=predictions_data["n_true"],
            edge_true=predictions_data["e_true"],
        )
        if save_figures is True:
            plt.savefig(save_path / "Batch_Dataset-Confusion_Matrix_Tiles.png")
        if show_figures is True:
            plt.show()
        plt.close()

        # Areas under curves:
        plot_areas_under_curves(
            node_pred=predictions_data["n_pred"],
            edge_pred=predictions_data["e_pred"],
            node_true=predictions_data["n_true"],
            edge_true=predictions_data["e_true"],
        )
        if save_figures is True:
            plt.savefig(save_path / "Batch_Dataset-Areas_Under_Curves.png")
        if show_figures is True:
            plt.show()
        plt.close()

        # Predicted probs hist:
        plot_prediction_probabilities_hist(
            node_pred=predictions_data["n_pred"],
            edge_pred=predictions_data["e_pred"],
            node_true=predictions_data["n_true"],
            edge_true=predictions_data["e_true"],
        )
        if save_figures is True:
            plt.savefig(
                save_path / "Batch_Dataset-Histogram_Prediction_Probs.png"
            )
        if show_figures is True:
            plt.show()
        plt.close()
