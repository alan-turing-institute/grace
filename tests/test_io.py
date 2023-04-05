import starfile
import pandas as pd

from grace.base import GraphAttrs
from grace.io.cbox import read_cbox_as_nodes
from pathlib import Path


def test_read_cbox(tmp_path, simple_graph_dataframe, fileformat):
    """Write out a very simple cbox file and then try to read it back."""

    filename = Path(tmp_path) / f"test.{fileformat}"

    # create the data
    wh = [100] * simple_graph_dataframe.shape[0]
    data = {
        "_CoordinateX": simple_graph_dataframe[GraphAttrs.NODE_X],
        "_CoordinateY": simple_graph_dataframe[GraphAttrs.NODE_Y],
        "_Width": wh,
        "_Height": wh,
        "_Confidence": simple_graph_dataframe[GraphAttrs.NODE_CONFIDENCE],
    }

    # write it out
    starfile.write({"cryolo": pd.DataFrame(data)}, filename)

    # now try to read it
    recovered_nodes = read_cbox_as_nodes(filename)

    for node_id, node_data in recovered_nodes:
        assert node_data[GraphAttrs.NODE_X] == data["_CoordinateX"][node_id]
        assert node_data[GraphAttrs.NODE_Y] == data["_CoordinateY"][node_id]
        assert (
            node_data[GraphAttrs.NODE_CONFIDENCE]
            == data["_Confidence"][node_id]
        )
