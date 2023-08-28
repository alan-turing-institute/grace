import numpy as np
import numpy.typing as npt
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.colors import to_rgb
from scipy.ndimage import label

from grace.base import GraphAttrs


RGB_MAPPING = {
    "grey": "image background",
    "gold": "only in predicted graph",
    "dodgerblue": "only in ground truth",
    "forestgreen": "perfect match",
}


def calculate_overlap(mask1, mask2):
    # Perform element-wise logical AND operation
    overlap_mask = np.logical_and(mask1, mask2)
    return overlap_mask


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
    hand_annotated_mask: npt.NDArray,
    auto_annotated_mask: npt.NDArray,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    overlap = create_overlay_image(hand_annotated_mask, auto_annotated_mask)
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


def _find_connected_objects(mask):
    labeled_mask, num_features = label(mask)

    connected_objects = []

    for label_idx in range(1, num_features + 1):
        object_indices = np.argwhere(labeled_mask == label_idx)
        object_indices = [[y, x] for x, y in object_indices]
        connected_objects.append(set(map(tuple, object_indices)))

    return connected_objects


def draw_annotation_mask_from_ground_truth_graph(
    gt_graph: nx.Graph,
    shape: tuple[int, int],
    brush_size: int = 75,
) -> npt.NDArray:
    """Create a binary annotation mask from ground truth graph,
    which may contain all nodes but only contains real edges.

    Parameters
    ----------
    gt_graph : nx.Graph
        The ground truth graph.
    shape : tuple[int, int]
        The shape of the hand-annotated image.
    brush_size : int
        Thickness of the line to draw the mask.
        Adjust according to the hand-annotated mask.

    Returns
    -------
    annotation_mask : npt.NDArray
        The automatically annotated binary array.
    """

    # Create an annotation mask with the same dimensions as the image
    annotation_mask = np.zeros(shape=shape, dtype=np.uint8)

    # Iterate through GT edges:
    for edge in gt_graph.edges():
        node1_x, node1_y = int(
            gt_graph.nodes[edge[0]][GraphAttrs.NODE_X]
        ), int(gt_graph.nodes[edge[0]][GraphAttrs.NODE_Y])
        node2_x, node2_y = int(
            gt_graph.nodes[edge[1]][GraphAttrs.NODE_X]
        ), int(gt_graph.nodes[edge[1]][GraphAttrs.NODE_Y])

        # Calculate the points for the line endpoints
        endpoint1 = np.array([node1_x, node1_y])
        endpoint2 = np.array([node2_x, node2_y])

        # Calculate the line points using integer coordinates
        num_points = int(np.linalg.norm(endpoint2 - endpoint1))
        line_points = np.column_stack(
            (
                np.linspace(endpoint1[0], endpoint2[0], num_points),
                np.linspace(endpoint1[1], endpoint2[1], num_points),
            )
        )

        # Set the line points in the annotation mask
        for point in line_points:
            y, x = map(int, point)
            y_start = max(0, y - brush_size // 2)
            y_end = min(annotation_mask.shape[0], y + brush_size // 2 + 1)
            x_start = max(0, x - brush_size // 2)
            x_end = min(annotation_mask.shape[1], x + brush_size // 2 + 1)
            annotation_mask[y_start:y_end, x_start:x_end] = 1

    annotation_mask = annotation_mask.T
    return annotation_mask
