import pytest
import pandas as pd


@pytest.fixture
def simple_graph_dataframe() -> pd.DataFrame:
    """Fixture for as simple graph as a dataframe."""
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 0.0],
            "features": [-1.0, 0.0, 1.0],
        }
    )

    return df
