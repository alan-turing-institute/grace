import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import networkx as nx

from collections import Counter
from grace.base import GraphAttrs
from grace.styling import COLORMAPS
from grace.napari.utils import EdgeColor


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


def display_image_and_grace_annotation(
    image: npt.NDArray,
    target: dict[str],
    cmap: str = COLORMAPS["annotation"],
) -> None:
    """Overlays the annotation image (binary mask) with annotated graph,
        colour-coding the true positive (TP), true negative (TN), and
        unannotated elements of the graph (nodes & edges).

    Parameters
    ----------
    image : npt.NDArray
        Raw image array
    target : dict[str]
        Dictionary containing keys:
            'graph' : nx.Graph
                annotated graph with node & edge attributes
            'annotation' : npt.NDArray
                binary annotated image mask
            'metadata' : str
                'image_filename', etc.
    """

    annotation = target["annotation"]
    assert image.shape == annotation.shape

    # Simple image data plot - side by side:
    plt.figure(figsize=(15, 7))
    names = ["Raw image data", "GRACE annotation"]

    for i, image in enumerate([image, annotation]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(image, cmap=cmap, interpolation="none")
        plt.colorbar(fraction=0.045)
        plt.title(f"{names[i]}: {target['metadata']['image_filename']}")
    plt.show()

    # Read the annotated graph & count the nodes:
    graph = target["graph"]
    node_GT_counter = Counter(
        [
            node[GraphAttrs.NODE_GROUND_TRUTH]
            for _, node in graph.nodes(data=True)
        ]
    )

    # Fancy annotation plot
    _, ax = plt.subplots(figsize=(16, 16))

    # node positions
    pos = {
        idx: (node[GraphAttrs.NODE_X], node[GraphAttrs.NODE_Y])
        for idx, node in graph.nodes(data=True)
    }
    # edge annotations
    edge_gt = [
        graph[u][v][GraphAttrs.EDGE_GROUND_TRUTH] for u, v in graph.edges
    ]
    edge_colors = [EdgeColor[gt.name].value for gt in edge_gt]

    node_colors = [
        EdgeColor[node_attrs[GraphAttrs.NODE_GROUND_TRUTH].name].value
        for _, node_attrs in graph.nodes(data=True)
    ]

    ax.imshow(annotation, cmap=cmap, interpolation="none")

    # draw all nodes/vertices in the graph:
    nx.draw_networkx(
        graph,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_size=32,
        edge_color=edge_colors,
        node_color=node_colors,
    )

    ax.set_title(f"{target['metadata']['image_filename']}\n{node_GT_counter}")
    plt.show()
    plt.close()
