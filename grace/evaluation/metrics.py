import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import networkx as nx
import matplotlib.patches as patches

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    ConfusionMatrixDisplay,
)

from grace.base import GraphAttrs

from grace.evaluation.annotate import (
    _find_connected_objects,
    visualise_semantic_iou_map,
    semantic_iou_from_masks,
    instance_iou_from_masks,
)

COLOR_LIST = ["limegreen", "gold", "dodgerblue"]
TITLE_LIST = [
    "perfect match",
    "only in predicted graph",
    "only in ground truth",
]

LEGEND_HANDLE = {COLOR_LIST[i]: TITLE_LIST[i] for i in range(len(COLOR_LIST))}


def _list_real_connected_components(graph: nx.Graph) -> list[set[int]]:
    """Filter through instance sets."""
    connected_comps = nx.connected_components(graph)
    con_comp_sets = []
    for comp in connected_comps:
        # Ignore 1-node components:
        if len(comp) <= 1:
            continue
        con_comp_sets.append(comp)
    return con_comp_sets


def _pool_connected_components_nodes(G: nx.Graph) -> set[int]:
    con_pool = set()
    cc = nx.connected_components(G)
    for comp in cc:
        # Ignore 1-node components:
        if len(comp) > 1:
            con_pool.update(comp)
    return con_pool


def _pool_connected_components_edges(G: nx.Graph) -> set[int]:
    con_pool = set()
    cc = nx.connected_components(G)
    for comp in cc:
        if len(comp) > 1:
            for i in range(len(list(comp)) - 1):
                src, dst = list(comp)[i : i + 2]
                con_pool.update([(src, dst), (dst, src)])
    return con_pool


def _generate_label_vector(input_data_set: set, component_pool_set: set):
    y_labels = [n in component_pool_set for n in input_data_set]
    y_labels = np.array(y_labels).astype(int)
    return y_labels


def _find_matching_pairs(
    list_of_sets1: list[set[int]] | list[set[tuple[int]]],
    list_of_sets2: list[set[int]] | list[set[tuple[int]]],
) -> list[set[int]] | list[set[tuple[int]]]:
    matching_pairs = []
    for set1 in list_of_sets1:
        for set2 in list_of_sets2:
            if len(set1.intersection(set2)) > 0:
                matching_pairs.append((set1, set2))
    return matching_pairs


def locate_rectangle_nodes(object_node_set: set[int], G: nx.Graph):
    nodes = G.nodes(data=True)
    x_coords, y_coords = [], []

    for node_idx in object_node_set:
        y_coords.append(nodes[node_idx][GraphAttrs.NODE_Y])
        x_coords.append(nodes[node_idx][GraphAttrs.NODE_X])

    x_mn, x_mx = np.min(x_coords), np.max(x_coords)
    y_mn, y_mx = np.min(y_coords), np.max(y_coords)

    return (x_mn, y_mn), x_mx - x_mn, y_mx - y_mn  # anchor, height, width


def locate_rectangle_points(
    object_node_coords_set: set[tuple[int]], G: nx.Graph = None
) -> tuple[tuple[float], float, float]:
    x_coords, y_coords = [], []

    for coo_pair in object_node_coords_set:
        x_coords.append(coo_pair[0])
        y_coords.append(coo_pair[1])

    x_mn, x_mx = np.min(x_coords), np.max(x_coords)
    y_mn, y_mx = np.min(y_coords), np.max(y_coords)

    return (x_mn, y_mn), x_mx - x_mn, y_mx - y_mn  # anchor, height, width


def extract_rectangle_info(
    matching_pairs: list[set[int]] | list[set[tuple[int]]], G: nx.Graph = None
) -> list[tuple[tuple[float], float, float]]:
    # Nominate which function to use:
    if G is None:
        locate_function = locate_rectangle_points
    else:
        locate_function = locate_rectangle_nodes

    # Extract rectangle plotting data:
    rectangles = []
    for pair in matching_pairs:
        # Check if objects are equal:
        if pair[0] == pair[1]:
            anchor, height, width = locate_function(pair[0], G)
            rectangles.append((anchor, height, width, COLOR_LIST[0]))

        else:
            # Under/over-detected objects - unwrap both + assign correct color:
            for o, _ in enumerate(pair):
                anchor, height, width = locate_function(pair[o], G)
                rectangles.append((anchor, height, width, COLOR_LIST[o + 1]))

    return rectangles


def _legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, lab)
        for i, (h, lab) in enumerate(zip(handles, labels))
        if lab not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def intersection_over_union(set_1: set, set_2: set):
    # Calculate the intersection of sets
    intersection = len(set_1.intersection(set_2))

    # Calculate the union of sets
    union = len(set_1.union(set_2))

    if union < 1:
        return 0
    else:
        return intersection / union


def plot_confusion_matrices(
    y_pred_nodes: npt.NDArray,
    y_true_nodes: npt.NDArray,
    y_pred_edges: npt.NDArray,
    y_true_edges: npt.NDArray,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    _, axs = plt.subplots(2, 2, figsize=figsize)

    for c, comp_mat in enumerate(
        [
            [
                [y_pred_nodes, y_true_nodes, None, "Nodes | Raw count values"],
                [
                    y_pred_nodes,
                    y_true_nodes,
                    "true",
                    "Nodes | Normalised values",
                ],
            ],
            [
                [y_pred_edges, y_true_edges, None, "Edges | Raw count values"],
                [
                    y_pred_edges,
                    y_true_edges,
                    "true",
                    "Edges | Normalised values",
                ],
            ],
        ]
    ):
        for n, norm_mat in enumerate(comp_mat):
            ConfusionMatrixDisplay.from_predictions(
                y_pred=norm_mat[0],
                y_true=norm_mat[1],
                normalize=norm_mat[2],
                ax=axs[c, n],
                cmap="copper",
                display_labels=["TN", "TP"],
                text_kw={"fontsize": "large"},
            )
            axs[c, n].set_title(norm_mat[3])
    plt.show()
    plt.close()


def plot_iou_histogram(
    iou_per_object: npt.NDArray,
    iou_semantic: float,
    figsize: tuple[int, int] = (10, 3),
) -> None:
    # Instantiate a figure
    plt.figure(figsize=figsize)

    # Plot the histogram + boundaries
    plt.hist(iou_per_object, color="grey", label="Individual objects IoU")
    plt.xlim(-0.15, 1.15)
    mn, std = np.mean(iou_per_object), np.std(iou_per_object)

    # Add vertical lines
    plt.axvline(
        x=iou_semantic,
        color="cyan",
        linestyle="dashed",
        linewidth=2,
        label=f"IoU semantic: {iou_semantic:.4f}",
    )
    plt.axvline(
        x=mn,
        color="purple",
        linestyle="dashed",
        linewidth=2,
        label=f"IoU instance {mn:.4f}",
    )
    plt.axvline(x=mn - std, color="purple", linestyle="dashed", linewidth=1)
    plt.axvline(x=mn + std, color="purple", linestyle="dashed", linewidth=1)

    # Fill the area between the vertical lines
    plt.fill_betweenx(
        [0, plt.gca().get_ylim()[1]],
        mn - std,
        mn + std,
        color="purple",
        alpha=0.3,
        label="IoU instance | mean ± std",
    )

    plt.xlabel("Object instance IoU")
    plt.ylabel("Object count")
    plt.title(
        f"Intersection over union across {len(iou_per_object)} overlapping objects"
    )
    plt.legend()
    plt.show()
    plt.close()


def visualise_object_bounding_boxes_on_graph(
    G: nx.Graph,
    rectangles: list[tuple[float]],
    legend_handle: dict[str, str],
    annotation: npt.NDArray = None,
    figsize: tuple[int] = (10, 10),
) -> None:
    """TODO: Fill in."""

    # Create figure and axes
    _, ax = plt.subplots(figsize=figsize)

    # Plot the faded annotation under the graph
    if annotation is not None:
        ax.imshow(annotation, alpha=0.5, cmap="gray", interpolation="none")

    # Display the graph node positions
    pos = {
        idx: (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
        for idx, node in G.nodes(data=True)
    }

    # draw all nodes/vertices in the graph
    nx.draw_networkx(
        G, ax=ax, pos=pos, with_labels=False, node_color="k", node_size=32
    )

    # plot the rectangles one by one
    for rectangle in rectangles:
        anchor, height, width, color = rectangle
        handle = legend_handle[color]
        rect = patches.Rectangle(
            anchor,
            height,
            width,
            label=handle,
            linewidth=3,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        _legend_without_duplicate_labels(ax)

    ax.set_axis_on()
    ax.set_title("IoU metric illustration on per-object level")
    plt.show()


def format_object_detection_metrics(
    metrics_dict: dict[str, float],
    title: str = "Object detection metrics",
    minimum_padding: int = 0,
) -> str:
    # Calculate the maximum length of metric names for formatting
    max_metric_length = max(len(metric) for metric in metrics_dict.keys())

    # Calculate the total width of the table (including borders)
    min_table_width = max(
        max_metric_length + minimum_padding * 2,
        len(title) + minimum_padding * 2,
    )

    # Construct the table as a multiline string
    formatted_table = []
    formatted_table.append("=" * min_table_width)
    title_str = title[: min_table_width - minimum_padding * 2].center(
        min_table_width
    )
    formatted_table.append(title_str)
    formatted_table.append("=" * min_table_width)
    for metric, value in metrics_dict.items():
        formatted_table.append(
            "%-*s : %.4f" % (max_metric_length, metric, value)
        )
    formatted_table.append("=" * min_table_width)

    return "\n".join(formatted_table)


def combine_multiline_strings(
    string1: str, string2: str, column_gap: int = 10
):
    lines1 = string1.split("\n")
    lines2 = string2.split("\n")
    assert len(lines1) == len(lines2)

    max_line_length = max(len(line) for line in lines1)
    combined_lines = [
        f"{line1:<{max_line_length+column_gap}}{line2}"
        for line1, line2 in zip(lines1, lines2)
    ]
    combined_string = "\n".join(combined_lines)
    return combined_string


def compute_exact_metrics(
    G: nx.Graph,
    pred_graph: nx.Graph,
    true_graph: nx.Graph,
) -> None:
    """TODO: Fill in!
    TODO: Refactor into an object with methods!

    Parameters
    ----------
    G : nx.Graph
        Original graph with all nodes and possible edges (as triangulated)
    pred_graph : nx.Graph
        Predicted graph after the ILP optimisation step.
    true_graph : nx.Graph
        Ground truth graph deducted from napari annotation.

    Returns
    -------
    ...

    """

    # Prepare data:
    nodes, edges = G.nodes(data=False), G.edges(data=False)
    con_pool_pred_nodes = _pool_connected_components_nodes(pred_graph)
    con_pool_true_nodes = _pool_connected_components_nodes(true_graph)

    con_pool_pred_edges = _pool_connected_components_edges(pred_graph)
    con_pool_true_edges = _pool_connected_components_edges(true_graph)

    y_pred_nodes = _generate_label_vector(
        input_data_set=nodes, component_pool_set=con_pool_pred_nodes
    )
    y_true_nodes = _generate_label_vector(
        input_data_set=nodes, component_pool_set=con_pool_true_nodes
    )

    y_pred_edges = _generate_label_vector(
        input_data_set=edges, component_pool_set=con_pool_pred_edges
    )
    y_true_edges = _generate_label_vector(
        input_data_set=edges, component_pool_set=con_pool_true_edges
    )

    # Accuracy, precision, recall, f1, support scores:
    accur_nodes = accuracy_score(
        y_pred=y_pred_nodes, y_true=y_true_nodes, normalize=True
    )
    accur_edges = accuracy_score(
        y_pred=y_pred_edges, y_true=y_true_edges, normalize=True
    )

    prf1_nodes = precision_recall_fscore_support(
        y_pred=y_pred_nodes, y_true=y_true_nodes, average="binary"
    )
    prf1_edges = precision_recall_fscore_support(
        y_pred=y_pred_edges, y_true=y_true_edges, average="binary"
    )

    # Intersection over union:
    # Semantic IoU:
    iou_semantic = intersection_over_union(
        con_pool_pred_nodes, con_pool_true_nodes
    )

    # Instance IoU:
    con_comp_sets_pred = _list_real_connected_components(pred_graph)
    con_comp_sets_true = _list_real_connected_components(true_graph)

    # Find matching pairs of sets with at least one common node
    matching_pairs = _find_matching_pairs(
        con_comp_sets_pred, con_comp_sets_true
    )

    # Calculate IoU for every component pair with at least 1-node overlap:
    iou_per_object = []
    for pair in matching_pairs:
        iou_score = intersection_over_union(pair[0], pair[1])
        iou_per_object.append(iou_score)

        # Calculate mean ± st.dev. instance IoU:
    iou_instance_avg = np.mean(iou_per_object)
    iou_instance_std = np.std(iou_per_object)

    # RESULTS #1: Printable results:
    # Organise into a dictionary of object detection metrics:
    for metrics_data in [
        [accur_nodes, prf1_nodes, "Nodes"],
        [accur_edges, prf1_edges, "Edges"],
    ]:
        table_dict = {
            "Accuracy": metrics_data[0],
            "Precision": metrics_data[1][0],
            "Recall": metrics_data[1][1],
            "F1 Score": metrics_data[1][2],
        }
        # Format & print the table:
        formatted_table = format_object_detection_metrics(
            table_dict, title=f"Object detection metrics | {metrics_data[2]}"
        )
        print(formatted_table)

    # Print IoU results:
    table_dict = {
        "Semantic": iou_semantic,
        "Instance [mean]": iou_instance_avg,
        "Instance [std]": iou_instance_std,
    }
    # Format & print the table:
    formatted_table = format_object_detection_metrics(
        table_dict, title="Intersection over Union"
    )
    print(formatted_table)

    # RESULTS #2: Figures:
    # Confusion matrices
    plot_confusion_matrices(
        y_pred_nodes, y_true_nodes, y_pred_edges, y_true_edges
    )

    # IoU scores histogram:
    plot_iou_histogram(iou_per_object, iou_semantic)

    # Locate rectangle coords for every individual object:
    rectangles = extract_rectangle_info(matching_pairs=matching_pairs, G=G)
    visualise_object_bounding_boxes_on_graph(G, rectangles, LEGEND_HANDLE)


def compute_approx_metrics(
    G: nx.Graph,
    hand_annotated_mask: npt.NDArray,
    auto_annotated_mask: npt.NDArray,
) -> None:
    """TODO: Fill in.
    TODO: Create a wrapper for the metrics as above.

    Parameters
    ----------
    G : nx.Graph
        Original graph with all nodes and possible edges (as triangulated)

    """

    pixel_accuracy, object_accuracy = semantic_iou_from_masks(
        hand_annotated_mask, auto_annotated_mask
    )
    iou_per_object = instance_iou_from_masks(
        hand_annotated_mask, auto_annotated_mask
    )

    # TODO: Print numerical values:
    table_dict = {
        "Pixel Accuracy": pixel_accuracy,
        "Semantic IoU": object_accuracy,
        "Instance IoU [mean]": np.mean(iou_per_object),
        "Instance IoU [std]": np.std(iou_per_object),
    }
    # Format & print the table:
    formatted_table = format_object_detection_metrics(
        table_dict, title="Intersection over Union"
    )
    print(formatted_table)

    # Plot numerical values:
    plot_iou_histogram(iou_per_object, object_accuracy)

    # Visualise semantic IoU:
    visualise_semantic_iou_map(hand_annotated_mask, auto_annotated_mask)

    # Plot bbox overlap on mask:
    connected_objects_hand_anno = _find_connected_objects(hand_annotated_mask)
    connected_objects_auto_anno = _find_connected_objects(auto_annotated_mask)
    matching_pairs = _find_matching_pairs(
        connected_objects_hand_anno, connected_objects_auto_anno
    )
    rectangles = extract_rectangle_info(matching_pairs=matching_pairs)
    visualise_object_bounding_boxes_on_graph(
        G, rectangles, LEGEND_HANDLE, hand_annotated_mask
    )
