import pytest
import pandas as pd
import networkx as nx
import numpy as np

from grace.base import GraphAttrs, graph_from_dataframe

RNG_SEED = 1234
RNG = np.random.default_rng(seed=RNG_SEED)


@pytest.fixture
def default_rng() -> np.random._generator.Generator:
    return RNG


@pytest.fixture
def simple_graph_dataframe(default_rng) -> pd.DataFrame:
    """Fixture for as simple graph as a dataframe."""
    features = [default_rng.uniform(size=(32,)) for _ in range(3)]
    df = pd.DataFrame(
        {
            GraphAttrs.NODE_X: [0.0, 1.0, 2.0],
            GraphAttrs.NODE_Y: [0.0, 1.0, 0.0],
            GraphAttrs.NODE_FEATURES: features,
            GraphAttrs.NODE_GROUND_TRUTH: [1, 1, 1],
            GraphAttrs.NODE_CONFIDENCE: [0.9, 0.1, 0.8],
        }
    )
    print(df.head())
    return df


@pytest.fixture
def simple_graph(simple_graph_dataframe) -> nx.Graph:
    """Fixture for a simple graph."""
    graph = graph_from_dataframe(simple_graph_dataframe)
    return graph
