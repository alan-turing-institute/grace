import torch

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    average_precision_score,
    PrecisionRecallDisplay,
)


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


def areas_under_curves_metrics(
    node_pred: torch.tensor,
    edge_pred: torch.tensor,
    node_true: torch.tensor,
    edge_true: torch.tensor,
    figsize: tuple[int] = (20, 7),
) -> tuple[plt.figure]:
    # Unify the inputs - get the predictions scores for TP class:
    if node_pred.shape[-1] == 2:
        node_pred = node_pred[:, 1]
    if edge_pred.shape[-1] == 2:
        edge_pred = edge_pred[:, 1]

    # Instantiate the figure
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Area under ROC:
    roc_score_nodes = roc_auc_score(y_true=node_true, y_score=node_pred)
    # rcd_nodes = RocCurveDisplay.from_predictions(
    RocCurveDisplay.from_predictions(
        y_true=node_true,
        y_pred=node_pred,
        color="dodgerblue",
        lw=3,
        label=f"Nodes = {roc_score_nodes:.4f}",
        ax=axes[0],
    )

    roc_score_edges = roc_auc_score(y_true=edge_true, y_score=edge_pred)
    # rcd_edges = RocCurveDisplay.from_predictions(
    RocCurveDisplay.from_predictions(
        y_true=edge_true,
        y_pred=edge_pred,
        color="forestgreen",
        lw=3,
        label=f"Edges = {roc_score_edges:.4f}",
        ax=axes[0],
    )

    # Average Precision:
    prc_score_nodes = average_precision_score(
        y_true=node_true, y_score=node_pred
    )
    # prc_nodes = PrecisionRecallDisplay.from_predictions(
    PrecisionRecallDisplay.from_predictions(
        y_true=node_true,
        y_pred=node_pred,
        color="dodgerblue",
        lw=3,
        label=f"Nodes = {prc_score_nodes:.4f}",
        ax=axes[1],
    )

    prc_score_edges = average_precision_score(
        y_true=edge_true, y_score=edge_pred
    )
    # prc_edges = PrecisionRecallDisplay.from_predictions(
    PrecisionRecallDisplay.from_predictions(
        y_true=edge_true,
        y_pred=edge_pred,
        color="forestgreen",
        lw=3,
        label=f"Edges = {prc_score_edges:.4f}",
        ax=axes[1],
    )

    # Annotate the figure:
    axes[0].plot([0, 0], [1, 1], ls="dashed", lw=1, color="lightgrey")
    axes[1].plot([0, 1], [0.5, 0.5], ls="dashed", lw=1, color="lightgrey")
    axes[1].plot([0.5, 0.5], [0, 1], ls="dashed", lw=1, color="lightgrey")

    axes[0].set_title("Area under ROC")
    axes[1].set_title("Average Precision Score")
    plt.tight_layout()
    # plt.show()

    return axes


METRICS = {
    "accuracy": accuracy_metric,
    "confusion_matrix": confusion_matrix_metric,
}


def get_metric(name: str) -> callable:
    return METRICS[name]
