from typing import List, Tuple, Callable

import torch

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def accuracy_metric(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    node_true: torch.Tensor,
    edge_true: torch.Tensor,
) -> Tuple[float]:
    node_pred_labels = node_pred.argmax(dim=-1)
    edge_pred_labels = edge_pred.argmax(dim=-1)

    correct_nodes = (node_pred_labels == node_true).sum()
    correct_edges = (edge_pred_labels == edge_true).sum()

    node_acc = correct_nodes / node_pred.size(-2)
    edge_acc = correct_edges / edge_pred.size(-2)

    return float(node_acc), float(edge_acc)


def confusion_matrix_metric(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    node_true: torch.Tensor,
    edge_true: torch.Tensor,
    node_classes: List[str] = ["TP", "TN"],
    edge_classes: List[str] = ["TP", "TN"],
    figsize: Tuple[int] = (6, 5),
) -> Tuple[plt.Figure]:
    node_pred_labels = node_pred.argmax(dim=-1)
    edge_pred_labels = edge_pred.argmax(dim=-1)

    cm_node = confusion_matrix(
        node_true.detach().numpy(),
        node_pred_labels.detach().numpy(),
        labels=np.arange(len(node_classes)),
        normalize="true",
    )
    cm_edge = confusion_matrix(
        edge_true.detach().numpy(),
        edge_pred_labels.detach().numpy(),
        labels=np.arange(len(edge_classes)),
        normalize="true",
    )

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
