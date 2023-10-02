import pytest
import pandas as pd
import networkx as nx
import numpy as np

import torch

from grace.base import GraphAttrs, graph_from_dataframe
from pathlib import Path

from _utils import random_image_and_graph


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
