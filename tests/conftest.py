import pytest
import pandas as pd

from grace.base import GraphAttrs


@pytest.fixture
def simple_graph_dataframe() -> pd.DataFrame:
    """Fixture for as simple graph as a dataframe."""
    df = pd.DataFrame(
        {
            GraphAttrs.NODE_X: [0.0, 1.0, 2.0],
            GraphAttrs.NODE_Y: [0.0, 1.0, 0.0],
            GraphAttrs.NODE_FEATURES: [-1.0, 0.0, 1.0],
            GraphAttrs.NODE_CONFIDENCE: [0.0, 0.5, 1.0],
        }
    )
    return df
