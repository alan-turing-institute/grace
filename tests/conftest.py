import pytest
import pandas as pd
import networkx as nx

from grace.base import GraphAttrs, graph_from_dataframe


@pytest.fixture
def simple_graph_dataframe() -> pd.DataFrame:
    """Fixture for as simple graph as a dataframe."""
    df = pd.DataFrame(
        {
            GraphAttrs.NODE_X: [0.0, 1.0, 2.0],
            GraphAttrs.NODE_Y: [0.0, 1.0, 0.0],
            GraphAttrs.NODE_FEATURES: [-1.0, 0.0, 1.0],
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
