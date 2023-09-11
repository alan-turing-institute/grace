from grace.base import GraphAttrs, Annotation
from grace.napari.utils import EdgeColor
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib

import numpy.typing as npt

from skimage.util import montage


def plot_simple_graph(
    G: nx.Graph,
    title: str = "",
    ax: matplotlib.axes = None,
) -> None:
    """Plots a simple graph with black nodes and edges."""

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

    ax.set_title(f"{title}")
    return ax


def plot_connected_components(
    G: nx.Graph,
    title: str = "",
    ax: matplotlib.axes = None,
) -> None:
    """Colour-codes the connected components (individual objects)
    & plots them onto a simple graph with black nodes & edges.
    Connected component (subgraph) must contain at least one edge.
    """

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

    ax.set_title(f"{title}")
    return ax


def display_image_and_grace_annotation(
    image: npt.NDArray,
    target: dict[str],
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

    Notes
    -----
    TODO: Complicated & duplex function, could break into two functions.
    """

    annotation = target["annotation"]
    assert image.shape == annotation.shape

    # Simple image data plot - side by side:
    plt.figure(figsize=(15, 7))
    names = ["Raw image data", "GRACE annotation"]

    for i, image in enumerate([image, annotation]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(image)
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

    ax.imshow(annotation, cmap=plt.cm.turbo, interpolation="none")

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
