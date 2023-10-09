import pytest
import pandas as pd
import networkx as nx
import numpy as np
import numpy.typing as npt

import torch

from pathlib import Path
from grace.base import (
    GraphAttrs,
    Annotation,
    Properties,
    EdgeProps,
    graph_from_dataframe,
)


def create_nodes(
    num_nodes: int, feature_ndim: int, rng, image_size: tuple[int, int]
):
    features = [rng.uniform(size=(feature_ndim,)) for _ in range(num_nodes)]
    node_coords = rng.integers(0, image_size[1], size=(num_nodes, 2))
    node_ground_truth = rng.choice(
        [Annotation.TRUE_NEGATIVE, Annotation.TRUE_POSITIVE], size=(num_nodes,)
    )
    df = pd.DataFrame(
        {
            GraphAttrs.NODE_X: node_coords[:, 0],
            GraphAttrs.NODE_Y: node_coords[:, 1],
            GraphAttrs.NODE_FEATURES: features,
            GraphAttrs.NODE_GROUND_TRUTH: node_ground_truth,
            GraphAttrs.NODE_CONFIDENCE: rng.uniform(
                size=(num_nodes),
            ),
        }
    )
    return df, node_coords


def create_edges(
    src: int,
    dst: int,
    rng,
) -> tuple[int, int, dict]:
    keys = [item.value for item in EdgeProps]
    vals = rng.uniform(size=(len(keys),))
    print(len(keys), len(vals))
    properties = Properties()
    properties.from_keys_and_values(keys=keys, values=vals)
    annotation = rng.choice(
        [Annotation.TRUE_NEGATIVE, Annotation.TRUE_POSITIVE],
    )
    attribute_dict = {
        GraphAttrs.EDGE_GROUND_TRUTH: annotation,
        GraphAttrs.EDGE_PROPERTIES: properties,
    }
    return (src, dst, attribute_dict)


def random_image_and_graph(
    rng,
    *,
    num_nodes: int = 4,
    image_size: tuple[int] = (128, 128),
    feature_ndim: int = 32,
) -> tuple[npt.NDArray, list[nx.Graph]]:
    """Create a random image and graph."""
    # Create the graph's nodes & edges:
    df, node_coords = create_nodes(num_nodes, feature_ndim, rng, image_size)
    graph = graph_from_dataframe(df, triangulate=True)
    graph.update(
        edges=[create_edges(src, dst, rng) for src, dst in graph.edges]
    )
    # Create an accompanying image:
    image = np.zeros(image_size, dtype=np.uint16)
    image[tuple(node_coords[:, 1]), tuple(node_coords[:, 0])] = 1

    return image, graph


@pytest.fixture(scope="session")
def default_rng() -> np.random._generator.Generator:
    """RNG for tests."""
    _seed = 1234
    return np.random.default_rng(seed=_seed)


@pytest.fixture
def simple_graph_dataframe(default_rng) -> pd.DataFrame:
    """Fixture for as simple graph as a dataframe."""
    feature_ndim = 32
    features = [default_rng.uniform(size=(feature_ndim,)) for _ in range(3)]
    df = pd.DataFrame(
        {
            GraphAttrs.NODE_X: [0.0, 1.0, 2.0],
            GraphAttrs.NODE_Y: [0.0, 1.0, 0.0],
            GraphAttrs.NODE_FEATURES: features,
            GraphAttrs.NODE_GROUND_TRUTH: [1, 1, 1],
            GraphAttrs.NODE_CONFIDENCE: [0.9, 0.1, 0.8],
        }
    )
    return df


@pytest.fixture
def simple_graph(simple_graph_dataframe) -> nx.Graph:
    """Fixture for a simple graph."""
    graph = graph_from_dataframe(simple_graph_dataframe)
    return graph


@pytest.fixture(scope="session")
def mrc_image_and_annotations_dir(tmp_path_factory, default_rng) -> Path:
    """Make some MRC images and corresponding grace files."""

    import mrcfile
    from grace.io import write_graph

    tmp_data_dir = Path(tmp_path_factory.mktemp("data"))
    num_images = 10

    for idx in range(num_images):
        image, graph = random_image_and_graph(default_rng)

        image_fn = tmp_data_dir / f"image_{idx}.mrc"
        grace_fn = tmp_data_dir / f"image_{idx}.grace"
        metadata = {"image_filename": image_fn.stem}

        with mrcfile.new(image_fn) as mrc:
            mrc.set_data(image)

        write_graph(grace_fn, graph=graph, metadata=metadata)

    # check that we made the files
    assert len(list(tmp_data_dir.glob("*.mrc"))) == num_images
    assert len(list(tmp_data_dir.glob("*.grace"))) == num_images

    return tmp_data_dir


class SimpleExtractor(torch.nn.Module):
    def forward(self, x):
        return torch.rand(x.size(0), 2)


@pytest.fixture(scope="session")
def simple_extractor() -> torch.nn.Module:
    return SimpleExtractor()
