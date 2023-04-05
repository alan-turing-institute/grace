import os
import starfile
import pandas as pd

from grace.base import GraphAttrs
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_cbox_as_nodes(path: os.PathLike) -> List[Tuple[int, Dict[str, Any]]]:
    """Read a `.cbox` file and return a networkx graph of the bounding boxes.

    Parameters
    ----------
    filename :
        A filename for a `cbox` file to load.

    Returns
    -------
    nodes : list
        A list of networkx compatible nodes representing the bounding boxes. Can
        be conveted to a graph using `add_nodes_from`.
    """

    if not Path(path).suffix == ".cbox":
        raise IOError(f"File is not a `.cbox` file: {path.stem}")

    cbox_data = starfile.read(path, always_dict=True)

    if "cryolo" not in cbox_data.keys():
        raise IOError("File is not a valid `.cbox` file.")

    cbox_df = pd.DataFrame(cbox_data["cryolo"])
    num_nodes = cbox_df.shape[0]

    nodes = [
        (
            idx,
            {
                GraphAttrs.NODE_X: cbox_df["_CoordinateX"][idx],
                GraphAttrs.NODE_Y: cbox_df["_CoordinateY"][idx],
                GraphAttrs.NODE_CONFIDENCE: cbox_df["_Confidence"][idx],
                GraphAttrs.NODE_WIDTH: cbox_df["_Width"][idx],
                GraphAttrs.NODE_HEIGHT: cbox_df["_Height"][idx],
            },
        )
        for idx in range(num_nodes)
    ]

    return nodes
