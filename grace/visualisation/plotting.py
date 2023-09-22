from grace.base import GraphAttrs, Annotation

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib

import numpy.typing as npt

from skimage.util import montage
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
    average_precision_score,
    PrecisionRecallDisplay,
)


def plot_simple_graph(
    G: nx.Graph,
    title: str = "",
    *,
    ax: matplotlib.axes = None,
    **kwargs,
) -> matplotlib.axes:
    """Plots a simple graph with black nodes and edges."""
    # Make sure some axis to plot onto is defined:
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    # Read node positions:
    pos = {
        idx: (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
        for idx, node in G.nodes(data=True)
    }

    # Draw all nodes/vertices in the graph, including noisy nodes:
    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_size=32,
        edge_color="k",
        node_color="k",
    )
    ax.invert_yaxis()
    ax.set_title(f"{title}")
    return ax


def plot_connected_components(
    G: nx.Graph,
    title: str = "",
    *,
    ax: matplotlib.axes = None,
    **kwargs,
) -> matplotlib.axes:
    """Colour-codes the connected components (individual objects)
    & plots them onto a simple graph with black nodes & edges.
    Connected component (subgraph) must contain at least one edge.
    """
    # Make sure some axis to plot onto is defined:
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    # Read node positions:
    pos = {
        idx: (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
        for idx, node in G.nodes(data=True)
    }

    # Draw all nodes/vertices in the graph, including noisy nodes:
    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_color="k",
        node_size=32,
    )

    # get each connected subgraph and draw it with a different colour
    cc = nx.connected_components(G)
    for index, sg in enumerate(cc):
        # Ignore 1-node components:
        if len(sg) <= 1:
            continue

        # Color-code proper objects
        c_idx = np.array(plt.cm.tab20((index % 20) / 20)).reshape(1, -1)
        sg = G.subgraph(sg).copy()

        nx.draw_networkx(
            sg,
            ax=ax,
            pos=pos,
            edge_color=c_idx,
            node_color=c_idx,
        )

    ax.invert_yaxis()
    ax.set_title(f"{title}")
    return ax


def plot_confusion_matrix_tiles(
    node_pred: npt.NDArray,
    edge_pred: npt.NDArray,
    node_true: npt.NDArray,
    edge_true: npt.NDArray,
    *,
    figsize: tuple[int, int] = (10, 10),
    cmap: str = "copper",
) -> None:
    # Prep:
    confusion_matrix_plotting_data = [
        [node_pred, node_true, "nodes"],
        [edge_pred, edge_true, "edges"],
    ]

    # Plot:
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    for d, matrix_data in enumerate(confusion_matrix_plotting_data):
        if len(np.unique(matrix_data[1])) < 2:
            continue

        for n, nrm in enumerate([None, "true"]):
            ConfusionMatrixDisplay.from_predictions(
                y_pred=matrix_data[0],
                y_true=matrix_data[1],
                normalize=nrm,
                ax=axs[d, n],
                cmap=cmap,
                display_labels=["TN", "TP"],
                text_kw={"fontsize": "large"},
            )

            flag = "Raw Counts" if nrm is None else "Normalised"
            text = f"{matrix_data[2].capitalize()} | {flag} Values"
            axs[d, n].set_title(text)

    plt.tight_layout()
    return fig


def plot_areas_under_curves(
    node_pred: npt.NDArray,
    edge_pred: npt.NDArray,
    node_true: npt.NDArray,
    edge_true: npt.NDArray,
    figsize: tuple[int] = (10, 4),
) -> plt.figure:
    # Instantiate the figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Area under ROC:
    roc_score_nodes = roc_auc_score(y_true=node_true, y_score=node_pred)
    RocCurveDisplay.from_predictions(
        y_true=node_true,
        y_pred=node_pred,
        color="dodgerblue",
        lw=3,
        label=f"Nodes = {roc_score_nodes:.4f}",
        ax=axes[0],
    )

    roc_score_edges = roc_auc_score(y_true=edge_true, y_score=edge_pred)
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
    PrecisionRecallDisplay.from_predictions(
        y_true=edge_true,
        y_pred=edge_pred,
        color="forestgreen",
        lw=3,
        label=f"Edges = {prc_score_edges:.4f}",
        ax=axes[1],
    )

    # Annotate the figure:
    axes[0].plot([0, 1], [0, 1], ls="dashed", lw=1, color="lightgrey")
    axes[1].plot([0, 1], [0.5, 0.5], ls="dashed", lw=1, color="lightgrey")
    axes[1].plot([0.5, 0.5], [0, 1], ls="dashed", lw=1, color="lightgrey")

    axes[0].set_title("Area under ROC")
    axes[1].set_title("Average Precision Score")

    plt.tight_layout()
    return fig


def plot_prediction_probabilities_hist(
    node_pred: npt.NDArray,
    edge_pred: npt.NDArray,
    node_true: npt.NDArray,
    edge_true: npt.NDArray,
    *,
    figsize: tuple[int] = (10, 4),
) -> None:
    """Plot the prediction probabilities colour-coded by their GT label."""

    # Plot the node & edge histogram by label:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    for i, (pred, true, att) in enumerate(
        zip([node_pred, edge_pred], [node_true, edge_true], ["nodes", "edges"])
    ):
        for lab_idx in np.unique(true):
            preds = [p for p, t in zip(pred, true) if t == lab_idx]
            axes[i].hist(
                preds, alpha=0.7, label=f"GT = {lab_idx} | {len(preds)} {att}"
            )
            axes[i].set_title(f"Inferred predictions -> TP {att}")
            axes[i].set_xlabel("Predicted softmax probability")
            axes[i].legend()

    axes[0].set_ylabel("Attribute count")
    plt.tight_layout()
    return fig


def visualise_node_and_edge_probabilities(G: nx.Graph) -> plt.figure:
    """Visualise per-node & per-edge predictions on color-coded
    graph of TP attribute probabilities independently for
    nodes, independently for edges & in overlay of both.
    """

    # Create a figure and axes
    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, ncols + 1))
    cmap = plt.cm.ScalarMappable(cmap="coolwarm")

    # JUST THE NODES:
    nodes = list(G.nodes(data=True))
    x_coords = [node[GraphAttrs.NODE_X] for _, node in nodes]
    y_coords = [node[GraphAttrs.NODE_Y] for _, node in nodes]
    node_preds = [
        node[GraphAttrs.NODE_PREDICTION].prob_TP for _, node in nodes
    ]

    # Plot nodes:
    axes[0].scatter(
        x=x_coords,
        y=y_coords,
        c=node_preds,
        # cmap="coolwarm",
        cmap="RdPu",
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].scatter(
        x=x_coords,
        y=y_coords,
        c=node_preds,
        # cmap="coolwarm",
        cmap="RdPu",
        vmin=0.0,
        vmax=1.0,
    )

    # Add colorbar:
    cbar = plt.colorbar(cmap, ax=axes[0])
    cbar.set_label("Node Probability")

    # JUST THE EDGES:
    for src, dst, edge in G.edges(data=True):
        e_st_x, e_st_y = (
            nodes[src][1][GraphAttrs.NODE_X],
            nodes[src][1][GraphAttrs.NODE_Y],
        )
        e_en_x, e_en_y = (
            nodes[dst][1][GraphAttrs.NODE_X],
            nodes[dst][1][GraphAttrs.NODE_Y],
        )
        edge_pred = edge[GraphAttrs.EDGE_PREDICTION].prob_TP

        axes[1].plot(
            [e_st_x, e_en_x],
            [e_st_y, e_en_y],
            color=cmap.to_rgba(edge_pred),
            marker="",
        )
        axes[2].plot(
            [e_st_x, e_en_x],
            [e_st_y, e_en_y],
            color=cmap.to_rgba(edge_pred),
            marker="",
        )

    # Add colorbar
    cbar = plt.colorbar(cmap, ax=axes[1])
    cbar.set_label("Edge Probability")

    # Annotate & display:
    cbar = plt.colorbar(cmap, ax=axes[2])
    cbar.set_label("TP Probability")

    axes[0].set_title("Probability of 'nodeness'")
    axes[1].set_title("Probability of 'edgeness'")
    axes[2].set_title("Merged graph predictions")

    [axes[i].get_xaxis().set_visible(False) for i in range(ncols)]
    [axes[i].get_yaxis().set_visible(False) for i in range(ncols)]

    # Format & return:
    plt.tight_layout()
    return fig


def read_patch_stack_by_label(
    G: nx.Graph,
    image: npt.NDArray,
    crop_shape: tuple[int, int] = (224, 224),
) -> list[npt.NDArray]:
    """Reads the image & crops patches of specified at node locations.

    Parameters
    ----------
    G : nx.Graph
        The annotated graph.
    image : npt.NDArray
        Raw image array.
    crop_shape : tuple[int, int]
        Shape of the patches. Defaults to (224, 224).

    Returns
    -------
    crops : list[npt.NDArray]
        List of image stacks, divided by annotation class.
    """
    # Prepare the crops:
    crops = [[] for _ in range(len(Annotation))]

    for _, node in G.nodes.data():
        # Node coords:
        coords = node[GraphAttrs.NODE_Y], node[GraphAttrs.NODE_X]

        # Locate the patch:
        st_x, en_x = (
            int(coords[0]) - crop_shape[0] // 2,
            int(coords[0]) + crop_shape[0] // 2,
        )
        st_y, en_y = (
            int(coords[1]) - crop_shape[1] // 2,
            int(coords[1]) + crop_shape[1] // 2,
        )

        # Crop & sort based on labels:
        crop = image[st_x:en_x, st_y:en_y]
        label = node[GraphAttrs.NODE_GROUND_TRUTH]
        crops[label].append(crop)

    # Stack the crops:
    for i in range(len(crops)):
        if len(crops[i]) > 0:
            crops[i] = np.stack(crops[i])

    return crops


def montage_from_image_patches(crops: list[npt.NDArray]) -> None:
    """Visualise the few random patches per class as a montage.

    Parameters
    ----------
    crops : list[npt.NDArray]
        List of image stacks, divided by annotation class.

    """
    # Value extrema on all crops:
    mn = np.min([np.min(c) for c in crops if isinstance(c, np.ndarray)])
    mx = np.max([np.max(c) for c in crops if isinstance(c, np.ndarray)])

    # Value extrema on all crops:
    plt.figure(figsize=(15, 5))

    for c, patches in enumerate(crops):
        if not isinstance(patches, np.ndarray):
            continue

        # Randomise
        np.random.shuffle(patches)
        mont = montage(
            patches[:49],
            grid_shape=(7, 7),
            padding_width=10,
            fill=mx,
        )
        # Plot a few patches
        plt.subplot(1, len(crops), c + 1)
        plt.imshow(mont, cmap="binary_r", vmin=mn, vmax=mx)
        plt.colorbar(fraction=0.045)
        plt.title(f"Montage of patches\nwith 'node_label' = {c}")
        plt.axis("off")
    plt.show()
    plt.close()


def overlay_from_image_patches(crops: list[npt.NDArray]) -> None:
    """Visualise the average patch from stack per each class.

    Parameters
    ----------
    crops : list[npt.NDArray]
        List of image stacks, divided by annotation class.

    """
    plt.figure(figsize=(15, 5))
    for c, patches in enumerate(crops):
        if not isinstance(patches, np.ndarray):
            continue
        stack = np.mean(patches, axis=0)
        plt.subplot(1, len(crops), c + 1)
        plt.imshow(stack, cmap="binary_r")
        plt.colorbar(fraction=0.045)
        plt.title(f"Montage of patches\nwith 'node_label' = {c}")
        plt.axis("off")
    plt.show()
    plt.close()
