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
) -> tuple[float]:
    # Calculate the accuracy, with/out weights:
    # TODO: Implement / calculate class weighting:
    node_acc = accuracy_score(
        y_pred=node_pred,
        y_true=node_true,
        sample_weight=None,
    )
    edge_acc = accuracy_score(
        y_pred=edge_pred,
        y_true=edge_true,
        sample_weight=None,
    )

    return float(node_acc), float(edge_acc)


def confusion_matrix_metric(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    node_true: torch.Tensor,
    edge_true: torch.Tensor,
    node_classes: list[str] = ["TN_node", "TP_node"],
    edge_classes: list[str] = ["TN_edge", "TP_edge"],
    normalize: str = "true",
    figsize: tuple[int] = (6, 5),
) -> tuple[plt.Figure]:
    # Calculate confusion matrices:
    cm_node = confusion_matrix(
        node_true.detach().numpy(),
        node_pred.detach().numpy(),
        labels=np.arange(len(node_classes)),
        normalize=normalize,
    )
    cm_edge = confusion_matrix(
        edge_true.detach().numpy(),
        edge_pred.detach().numpy(),
        labels=np.arange(len(edge_classes)),
        normalize=normalize,
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


def get_metric(name: str) -> callable:
    return METRICS[name]
