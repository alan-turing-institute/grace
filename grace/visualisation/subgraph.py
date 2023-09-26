import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib
import torch_geometric
import numpy.typing as npt

from grace.base import GraphAttrs
from grace.models.datasets import _pos, _edge_index


COLOR_MAPPING = {0: "royalblue", 1: "firebrick", 2: "grey"}
STYLE_MAPPING = {0: "dashed", 1: "solid", 2: "dashed"}


def plot_subgraph_geometry(
    sub_graph: torch_geometric.data.Data, **kwargs
) -> tuple[npt.NDArray]:
    # Extract data from subgraph:
    node_positions = sub_graph.edge_attr
    edge_indices = sub_graph.edge_index
    node_true_labels = sub_graph.y
    edge_true_labels = sub_graph.edge_label

    # Perform dimensionality checks:
    assert node_positions.shape[0] == len(node_true_labels)
    assert edge_indices.shape[-1] == len(edge_true_labels)

    # Plot the subgraph:
    ax = plot_subgraph_coordinates(
        node_positions,
        edge_indices,
        node_true_labels,
        edge_true_labels,
        **kwargs,
    )
    return ax


def plot_local_node_geometry(
    G: nx.Graph, node_idx: int, **kwargs
) -> tuple[npt.NDArray]:
    # Define a subgraph with 1-hop connectivity:
    sub_graph = nx.ego_graph(G, node_idx, radius=1)
    central_node = np.array(
        [
            G.nodes[node_idx][GraphAttrs.NODE_X],
            G.nodes[node_idx][GraphAttrs.NODE_Y],
        ]
    )
    node_positions = _pos(sub_graph).numpy() - central_node
    edge_indices = _edge_index(sub_graph)

    # Extract data from subgraph:
    node_true_labels = np.stack(
        [
            n[GraphAttrs.NODE_GROUND_TRUTH]
            for _, n in sub_graph.nodes(data=True)
        ]
    )
    edge_true_labels = np.stack(
        [
            e[-1][GraphAttrs.EDGE_GROUND_TRUTH]
            for e in sub_graph.edges(data=True)
        ]
    )
    # Perform dimensionality checks:
    assert node_positions.shape[0] == len(node_true_labels)
    assert edge_indices.shape[-1] == len(edge_true_labels)

    # Optional: node & edge predictions:
    sample_node_data = list(sub_graph.nodes(data=True))[0][1]

    if GraphAttrs.NODE_PREDICTION not in sample_node_data:
        # Assume graph doesn't hold any predictions:
        node_pred_labels = None
        edge_pred_labels = None
    else:
        # Process node predictions:
        node_pred_labels = np.stack(
            [
                node[GraphAttrs.NODE_PREDICTION].label
                for _, node in sub_graph.nodes(data=True)
            ]
        )
        assert node_pred_labels.shape == node_true_labels.shape

        edge_pred_labels = np.stack(
            [
                edge[GraphAttrs.EDGE_PREDICTION].label
                for _, _, edge in sub_graph.edges(data=True)
            ]
        )
        assert edge_pred_labels.shape == edge_true_labels.shape

    # Plot the subgraph:
    ax = plot_subgraph_coordinates(
        node_positions,
        edge_indices,
        node_true_labels,
        edge_true_labels,
        node_pred_labels,
        edge_pred_labels,
        **kwargs,
    )
    return ax


def plot_subgraph_coordinates(
    node_positions: npt.NDArray,
    edge_indices: npt.NDArray,
    node_true_labels: npt.NDArray,
    edge_true_labels: npt.NDArray,
    node_pred_labels: npt.NDArray = None,
    edge_pred_labels: npt.NDArray = None,
    *,
    title: str = "",
    color_mapping: dict[int, str] = None,
    style_mapping: dict[int, str] = None,
    ax: matplotlib.axes = None,
    **kwargs,
) -> matplotlib.axes:
    # Make sure some axis to plot onto is defined:
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    # Define color mapping & plot:
    color_mapping = (
        color_mapping if color_mapping is not None else COLOR_MAPPING
    )
    style_mapping = (
        style_mapping if style_mapping is not None else STYLE_MAPPING
    )

    # Plot the edges underneath:
    for e_idx in range(edge_indices.shape[-1]):
        src, dst = edge_indices[:, e_idx]
        st, en = node_positions[src], node_positions[dst]
        true_color = color_mapping[edge_true_labels[e_idx].item()]
        ax.plot(
            [st[0], en[0]], [st[1], en[1]], lw=5, ls="solid", c=true_color
        )  # true

        # Visualise GCN predictions for edges, if any:
        if edge_pred_labels is not None:
            pred_style = "solid" if edge_pred_labels[e_idx] == 1 else "dashed"
            ax.plot(
                [st[0], en[0]], [st[1], en[1]], lw=2, ls=pred_style, c="black"
            )  # pred

    # Plot the scatter on top:
    for n_idx in range(node_positions.shape[0]):
        node_color = color_mapping[node_true_labels[n_idx].item()]
        ax.scatter(
            x=node_positions[n_idx, 0],
            y=node_positions[n_idx, 1],
            s=200,
            c=node_color,
            zorder=edge_indices.shape[-1] + n_idx + 1,
        )

        # Visualise GCN predictions for nodes, if any:
        if node_pred_labels is not None:
            node_style = "solid" if node_pred_labels[n_idx] == 1 else "dashed"
            ax.scatter(
                x=node_positions[n_idx, 0],
                y=node_positions[n_idx, 1],
                s=200,
                ls=node_style,
                linewidths=2,
                facecolors="none",
                edgecolors="k",
                zorder=edge_indices.shape[-1] + n_idx + 2,
            )

    # Fake a legend:
    for i in range(len(color_mapping)):
        ax.scatter(x=[], y=[], c=color_mapping[i], label=f"GT Label '{i}'")
        if i > 1:
            continue
        ax.scatter(
            x=[],
            y=[],
            facecolors="none",
            edgecolors="k",
            ls=style_mapping[i],
            label=f"Prediction '{i}'",
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(color_mapping),
    )

    # Plot the title:
    ax.set_title(f"Subgraph Geometry\n{title}")
    return ax
