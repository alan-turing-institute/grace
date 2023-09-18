import numpy as np
import matplotlib
import torch_geometric


COLOR_MAPPING = {
    0: "dodgerblue",  # "magenta",
    1: "firebrick",  # "green",
    2: "grey",
}


# TODO: implement a function that processes subgraph directly from G:
# def process_subgraph_data(sub_graph: nx.Graph | torch_geometric.data.Data):


def plot_subgraph_geometry(
    sub_graph: torch_geometric.data.Data,
    *,
    title: str = "",
    color_mapping: dict[int, str] = None,
    ax: matplotlib.axes = None,
) -> matplotlib.axes:
    # Extract data from subgraph:
    node_positions = sub_graph.edge_attr.numpy()
    node_GT_labels = sub_graph.y.numpy()
    edge_GT_labels = sub_graph.edge_label.tolist()
    edge_indices = sub_graph.edge_index.numpy()
    assert node_positions.shape[0] == len(node_GT_labels)
    assert edge_indices.shape[-1] == len(edge_GT_labels)

    # Define color mapping & plot:
    color_mapping = (
        color_mapping if color_mapping is not None else COLOR_MAPPING
    )

    # Plot the edges underneath:
    for e_idx in range(edge_indices.shape[-1]):
        src, dst = edge_indices[:, e_idx]
        st, en = node_positions[src], node_positions[dst]
        true_color = color_mapping[edge_GT_labels[e_idx]]
        ax.plot(
            [st[0], en[0]], [st[1], en[1]], lw=5, ls="solid", c=true_color
        )  # true

        # TODO: implement non-random predictions:
        pred_style = "dashed" if np.random.randint(2) == 0 else "solid"
        ax.plot(
            [st[0], en[0]], [st[1], en[1]], lw=2, ls=pred_style, c="black"
        )  # pred

    # Plot the scatter on top:
    node_color = [color_mapping[GT] for GT in node_GT_labels]
    # TODO: implement non-random predictions:
    node_style = [
        "solid" if np.random.randint(2) == 0 else "dashed"
        for _ in range(len(node_GT_labels))
    ]
    ax.scatter(
        x=node_positions[:, 0],
        y=node_positions[:, 1],
        s=200,
        c=node_color,
        ls=node_style,
        linewidths=2,
        edgecolors="k",
        zorder=edge_indices.shape[-1] + 1,
    )

    # Fake a legend:
    [
        ax.scatter(x=[], y=[], c=color_mapping[i], label=f"Label '{i}'")
        for i in range(len(color_mapping))
    ]
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(color_mapping),
    )

    # Plot the title:
    ax.set_title(f"Subgraph Geometry\n{title}")
    return ax
