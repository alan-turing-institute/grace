from typing import List, Tuple, Callable

import torch

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix


def accuracy_metric(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    node_true: torch.Tensor,
    edge_true: torch.Tensor,
) -> Tuple[float]:
    # Calculate class weighting:
    num_classes = np.unique(node_true)

    node_frequency = [np.sum([n == c for n in node_true]) / len(node_true) for c in num_classes]
    node_weights = node_frequency[::-1]

    edge_frequency = [np.sum([n == c for n in edge_true]) / len(edge_true) for c in num_classes]
    edge_weights = edge_frequency[::-1]

    # Calculate the accuracy, with/out weights:
    node_acc = accuracy_score(
        y_pred=node_pred, 
        y_true=node_true, 
        sample_weight=None,
        # sample_weight=node_weights
    )
    edge_acc = accuracy_score(
        y_pred=edge_pred, 
        y_true=edge_true, 
        sample_weight=None,
        # sample_weight=edge_weights
    )

    return float(node_acc), float(edge_acc)


def confusion_matrix_metric(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    node_true: torch.Tensor,
    edge_true: torch.Tensor,
    node_classes: List[str] = ["TN_node", "TP_node"],
    edge_classes: List[str] = ["TN_edge", "TP_edge"],
    figsize: Tuple[int] = (6, 5),
) -> Tuple[plt.Figure]:
    # Calculate confusion matrices:
    cm_node = confusion_matrix(
        node_true.detach().numpy(),
        node_pred.detach().numpy(),
        labels=np.arange(len(node_classes)),
        normalize="true",
    )
    cm_edge = confusion_matrix(
        edge_true.detach().numpy(),
        edge_pred.detach().numpy(),
        labels=np.arange(len(edge_classes)),
        normalize="true",
    )

    # Store confusion matrices:
    df_node = pd.DataFrame(
        cm_node,
        index=node_classes,
        columns=node_classes,
    )
    df_edge = pd.DataFrame(
        cm_edge,
        index=edge_classes,
        columns=edge_classes,
    )

    # Visualise confusion matrices:
    sn.set_theme(font="Helvetica", font_scale=2)
    fig_node = plt.figure(figsize=figsize)
    sn.heatmap(df_node, annot=True, vmin=0.0, vmax=1.0)
    fig_edge = plt.figure(figsize=figsize)
    sn.heatmap(df_edge, annot=True, vmin=0.0, vmax=1.0)

    for fig in (fig_node, fig_edge):
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

    return fig_node, fig_edge


METRICS = {
    "accuracy": accuracy_metric,
    "confusion_matrix": confusion_matrix_metric,
}


def get_metric(name: str) -> Callable:
    return METRICS[name]
