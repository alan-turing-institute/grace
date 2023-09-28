import numpy as np
import numpy.typing as npt
import networkx as nx

from grace.base import GraphAttrs


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
