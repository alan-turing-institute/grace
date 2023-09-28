import numpy as np
import numpy.typing as npt
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.colors import to_rgb
from scipy.ndimage import label
from sklearn.metrics import ConfusionMatrixDisplay

from grace.base import GraphAttrs
from grace.styling import COLORMAPS


COLOR_LIST = ["limegreen", "gold", "dodgerblue"]
TITLE_LIST = [
    "perfect match",
    "only in predicted graph",
    "only in ground truth",
]

LEGEND_HANDLE = {COLOR_LIST[i]: TITLE_LIST[i] for i in range(len(COLOR_LIST))}

RGB_MAPPING = {
    "grey": "image background",
    "gold": "only in predicted graph",
    "dodgerblue": "only in ground truth",
    "forestgreen": "perfect match",
}


def list_real_connected_components(G: nx.Graph) -> list[set[int]]:
    """Filter through instance sets."""
    connected_comps = nx.connected_components(G)
    con_comp_sets = []
    for comp in connected_comps:
        # Ignore 1-node components:
        if len(comp) <= 1:
            continue
        con_comp_sets.append(comp)
    return con_comp_sets


def find_connected_objects(mask: npt.NDArray):
    labeled_mask, num_features = label(mask)

    connected_objects = []

    for label_idx in range(1, num_features + 1):
        object_indices = np.argwhere(labeled_mask == label_idx)
        object_indices = [[y, x] for x, y in object_indices]
        connected_objects.append(set(map(tuple, object_indices)))

    return connected_objects


def find_matching_pairs(
    list_of_sets1: list[set[int]] | list[set[tuple[int]]],
    list_of_sets2: list[set[int]] | list[set[tuple[int]]],
) -> list[set[int]] | list[set[tuple[int]]]:
    matching_pairs = []

    lists_of_sets = [list_of_sets1, list_of_sets2]
    setsOverlapping = [[False] * len(lst) for lst in lists_of_sets]

    for s1, set1 in enumerate(list_of_sets1):
        for s2, set2 in enumerate(list_of_sets2):
            if len(set1.intersection(set2)) > 0:
                matching_pairs.append((set1, set2))
                setsOverlapping[0][s1] = True
                setsOverlapping[1][s2] = True

    for ch, overlap_check in enumerate(setsOverlapping):
        # Check if there is an unmatched set:
        if not all(overlap_check):
            for idx in range(len(overlap_check)):
                # If that set was used, ignore:
                if overlap_check[idx]:
                    continue

                # Index the unmatched set:
                unmatched_set = lists_of_sets[ch][idx]
                unmatched = [set(), set()]

                # Position correctly:
                unmatched[ch] = unmatched_set
                matching_pairs.append(tuple(unmatched))

    return matching_pairs


# Process annotation masks:


def calculate_pixel_accuracy(mask1, mask2):
    agreement_mask = mask1 == mask2
    return np.sum(agreement_mask) / agreement_mask.size


def calculate_object_accuracy(mask1, mask2):
    overlap_mask = np.logical_and(mask1, mask2)
    agreement_mask = np.logical_or(mask1, mask2)
    return np.sum(overlap_mask) / np.sum(agreement_mask)


def create_overlay_image(mask1, mask2):
    # Create channels for red, green, and blue
    r_channel = np.where(mask1 & ~mask2, 255, 0)
    g_channel = np.where(~mask1 & mask2, 255, 0)
    b_channel = np.where(mask1 & mask2, 255, 0)

    # Combine the channels to form an RGB image
    overlay_image = np.stack((r_channel, g_channel, b_channel), axis=-1)

    return overlay_image


def map_values_to_colors(input_image):
    mapping = {
        (0, 0, 0): "grey",  # Map black (0, 0, 0)
        (255, 0, 0): "gold",  # Map red (255, 0, 0)
        (0, 255, 0): "dodgerblue",  # Map green (0, 255, 0)
        (0, 0, 255): "forestgreen",  # Map blue (0, 0, 255)
    }

    output_image = np.zeros_like(input_image, dtype=np.float32)
    legend_patches = []

    for rgb_value, mapped_color in mapping.items():
        lab = RGB_MAPPING[mapped_color]
        patch = patches.Patch(color=mapped_color, label=lab)
        legend_patches.append(patch)

        mapped_color = to_rgb(mapped_color)
        mask = np.all(input_image == rgb_value, axis=-1)
        output_image[mask] = mapped_color

    return output_image, legend_patches


def semantic_iou_from_masks(
    hand_annotated_mask: npt.NDArray, auto_annotated_mask: npt.NDArray
) -> tuple[float]:
    # Calculate accuracy metrics
    acc_pix = calculate_pixel_accuracy(
        hand_annotated_mask, auto_annotated_mask
    )
    acc_obj = calculate_object_accuracy(
        hand_annotated_mask, auto_annotated_mask
    )
    return acc_pix, acc_obj


def instance_iou_from_masks(
    hand_annotated_mask: npt.NDArray, auto_annotated_mask: npt.NDArray
) -> npt.NDArray:
    mask_sum = np.add(hand_annotated_mask, auto_annotated_mask)
    labeled_map, num_objects = label(mask_sum)

    iou_list = []
    for lab_idx in range(1, num_objects + 1):
        obj_mask = labeled_map == lab_idx
        union_lab = np.sum(obj_mask)

        agreed = hand_annotated_mask[obj_mask] == auto_annotated_mask[obj_mask]
        inter_lab = np.sum(agreed)

        iou = inter_lab / union_lab
        iou_list.append(iou)

    return iou_list


def visualise_semantic_iou_map(
    auto_annotated_mask: npt.NDArray,
    hand_annotated_mask: npt.NDArray,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    overlap = create_overlay_image(auto_annotated_mask, hand_annotated_mask)
    overlap, patches = map_values_to_colors(overlap)

    # Calculate accuracy metrics
    acc_pix, acc_obj = semantic_iou_from_masks(
        hand_annotated_mask, auto_annotated_mask
    )

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(overlap)

    # Add legend
    plt.legend(handles=patches, loc="upper right")
    plt.title(
        f"IoU Semantic | Pixel Accuracy: {acc_pix:.4f} | Object Accuracy: {acc_obj:.4f}"
    )
    plt.show()


def intersection_over_union(set_1: set, set_2: set):
    # Calculate the intersection of sets
    intersection = len(set_1.intersection(set_2))

    # Calculate the union of sets
    union = len(set_1.union(set_2))

    if intersection == 0 or union == 0:
        return 0.0
    else:
        return intersection / union


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
            if len(pair[0]) < 1:
                continue
            anchor, height, width = locate_function(pair[0], G)
            rectangles.append((anchor, height, width, COLOR_LIST[0]))

        else:
            # Under/over-detected objects - unwrap both + assign correct color:
            for o, _ in enumerate(pair):
                if len(pair[o]) < 1:
                    continue
                anchor, height, width = locate_function(pair[o], G)
                rectangles.append((anchor, height, width, COLOR_LIST[o + 1]))

    return rectangles


### PLOTTING:


def _legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, lab)
        for i, (h, lab) in enumerate(zip(handles, labels))
        if lab not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def plot_confusion_matrix_tiles(
    node_pred: npt.NDArray,
    edge_pred: npt.NDArray,
    node_true: npt.NDArray,
    edge_true: npt.NDArray,
    *,
    figsize: tuple[int, int] = (10, 10),
    cmap: str = COLORMAPS["conf_matrix"],
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


def plot_iou_histogram(
    iou_per_object: list | npt.NDArray,
    iou_semantic: float = None,
    figsize: tuple[int, int] = (10, 4),
) -> None:
    # Instantiate a figure
    fig = plt.figure(figsize=figsize)

    # Plot the histogram + boundaries
    plt.hist(iou_per_object, color="grey", label="Individual objects IoU")
    plt.xlim(-0.15, 1.15)
    mn, std = np.mean(iou_per_object), np.std(iou_per_object)

    # Add vertical lines
    if isinstance(iou_semantic, float):
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
    return fig


def show_object_bounding_boxes_on_graph(
    G: nx.Graph,
    rectangles: list[tuple[float]],
    legend_handle: dict[str, str],
    annotation: npt.NDArray = None,
    figsize: tuple[int] = (10, 10),
    cmap: str = COLORMAPS["patches"],
) -> None:
    """TODO: Fill in."""

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the faded annotation under the graph
    if annotation is not None:
        ax.imshow(annotation, alpha=0.5, cmap=cmap, interpolation="none")

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
    return fig


def visualise_bounding_boxes_on_graph(
    G: nx.Graph,
    pred: nx.Graph | npt.NDArray,
    true: nx.Graph | npt.NDArray,
    display_image: npt.NDArray = None,
) -> None:
    if isinstance(display_image, np.ndarray):
        locator_function = find_connected_objects
        graph_input = None
    else:
        locator_function = list_real_connected_components
        graph_input = G

    con_obj_pred = locator_function(pred)
    con_obj_true = locator_function(true)
    matching_pairs = find_matching_pairs(con_obj_pred, con_obj_true)

    rectangles = extract_rectangle_info(
        matching_pairs=matching_pairs, G=graph_input
    )
    show_object_bounding_boxes_on_graph(
        G, rectangles, LEGEND_HANDLE, annotation=display_image
    )


### PRINTING:


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
