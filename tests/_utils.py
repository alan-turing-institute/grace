import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import List, Tuple

from grace.base import GraphAttrs, graph_from_dataframe


def random_image_and_graph(
    rng,
    *,
    num_nodes: int = 4,
    image_size: Tuple[int] = (128, 128),
) -> Tuple[npt.NDArray, List[nx.Graph]]:
    """Create a random image and graph."""
    image = np.zeros(image_size, dtype=np.uint16)

    feature_ndim = 32
    features = [rng.uniform(size=(feature_ndim,)) for _ in range(num_nodes)]

    node_coords = rng.integers(0, image.shape[1], size=(num_nodes, 2))
    node_ground_truth = rng.integers(0, 2, size=(num_nodes,))
    df = pd.DataFrame(
        {
            GraphAttrs.NODE_X: node_coords[:, 1],
            GraphAttrs.NODE_Y: node_coords[:, 0],
            GraphAttrs.NODE_FEATURES: features,
            GraphAttrs.NODE_GROUND_TRUTH: node_ground_truth,
            GraphAttrs.NODE_CONFIDENCE: rng.uniform(
                size=(num_nodes),
            ),
        }
    )

    image[tuple(node_coords[:, 0]), tuple(node_coords[:, 1])] = 1
    graph = graph_from_dataframe(df, triangulate=True)
    return image, graph
